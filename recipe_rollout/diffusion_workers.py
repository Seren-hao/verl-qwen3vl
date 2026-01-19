# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

import numpy as np
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_shard_placement_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.workers.config import FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from recipe.dance_grpo_mm.actor import DataParallelPPOActor
from recipe.dance_grpo_mm.rollout import HFRollout
from recipe.dance_grpo_mm.utils import load_repeat_data_train, init_fsdp_module
import mindspeed.megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.global_vars import get_args

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()


def set_random_seed(seed, only_rollout=False):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not only_rollout and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def create_device_mesh(world_size, fsdp_size):
    """Create device mesh for FSDP"""
    if fsdp_size <= 0 or fsdp_size > world_size:
        fsdp_size = world_size
    return init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["dp", "fsdp"])


class DiffusionActorRolloutWorker(Worker, DistProfilerExtension):
    """
    Worker for diffusion action rollout and GRPO training
    This worker encapsulates:
    1. Rollout process with diffusion model sampling and log probability calculation
    2. Reward calculation logic using EditScore reward model
    3. GRPO policy update with advantage clipping and KL divergence regularization
    """

    def __init__(self, config: DictConfig, role='hybrid', **kwargs):
        log_gpu_memory_usage("Before Diffustion Worker init", logger=logger, level=logging.INFO)
        Worker.__init__(self)

        self.config = config
        self.role = role
        self._is_actor = role in ["actor", "hybrid", "actor_rollout"]
        self._is_rollout = role in ["rollout", "hybrid"]
        from megatron.training.global_vars import set_args
        set_args("train")
        # Initialize distributed training
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            # TODO 初始化megatron组
            from megatron.core import parallel_state as mpu
            mpu.initialize_model_parallel()
        set_random_seed(seed=123, only_rollout=False)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Build device mesh for FSDP
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # Build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # Create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=rank, is_collect=True)

        self._register_dispatch_collect_info("rollout", dp_rank=rank, is_collect=True)

        # 只有当序列并行大小大于1时才创建分片管理器
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        else:
            self.ulysses_sharding_manager = None

        # Initialize models
        self.actor_module = None
        self.actor_module_fsdp = None
        self.ref_module = None
        self.ref_module_fsdp = None

        # Initialize optimizers and schedulers
        self.actor_optimizer = None
        self.actor_lr_scheduler = None
        # Setup FSDP
        self._mm_build_model_optimizer()
        log_gpu_memory_usage("After Diffustion Worker init", logger=logger, level=logging.INFO)

    def _build_model_optimizer(self):
        """Setup FSDP for distributed training"""
        # Apply monkey patches
        # Setup actor model with FSDP
        log_gpu_memory_usage("Before init_fsdp_module", logger=logger, level=logging.INFO)
        from mammothmoda2.model import Mammothmoda2ModelPartition
        from transformers import GenerationConfig
        config = GenerationConfig.from_pretrained(
            self.config.get('model_path', "./"), config_file_name=self.config.get('model_config', "./config.json")
        )
        transformer = Mammothmoda2ModelPartition(config)
        transformer_ref = Mammothmoda2ModelPartition(config)

        fsdp_config = self.config.actor.fsdp_config

        # Wrap actor model with FSDP
        self.actor_module_fsdp = init_fsdp_module(
            module=transformer,
            strategy=fsdp_config.strategy,
        )

        # Build optimizer using AdamW with parameters from train_grpo_edit.py
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.actor_module_fsdp.gen_transformer.parameters()))
        self.actor_optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.actor.optim.lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.actor.optim.weight_decay,
            eps=1e-8,
        )

        # Build LR scheduler using get_scheduler from diffusers.optimization
        from diffusers.optimization import get_scheduler
        self.actor_lr_scheduler = get_scheduler(
            self.config.actor.optim.lr_scheduler_name,
            optimizer=self.actor_optimizer,
            num_warmup_steps=self.config.actor.optim.lr_scheduler_num_warmup_steps,
            num_training_steps=self.config.actor.optim.lr_scheduler_num_training_steps,
            num_cycles=self.config.actor.optim.get("lr_scheduler_num_cycles", 1),
            power=self.config.actor.optim.get("lr_scheduler_power", 1.0),
        )

        # Setup reference model with FSDP
        if transformer_ref is not None:
            # Wrap reference model with FSDP
            self.ref_module_fsdp = init_fsdp_module(
                module=transformer_ref,
                strategy=fsdp_config.strategy,
                only_forward=True,
            )

        # Create a custom config for HFRollout that includes diffusion-specific parameters
        rollout_config = self.config.rollout
        actor_config = self.config.actor
        self.rollout = HFRollout(self.actor_module_fsdp, config=rollout_config)
        self.actor = DataParallelPPOActor(self.actor_module_fsdp, actor_config)
        self.ref_actor = DataParallelPPOActor(self.ref_module_fsdp,
                                              actor_config) if self.ref_module_fsdp is not None else None
        log_gpu_memory_usage("After init_fsdp_module", logger=logger, level=logging.INFO)

    def _mm_build_model_optimizer(self):
        """Setup FSDP for distributed training"""
        log_gpu_memory_usage("Before init_fsdp_module", logger=logger, level=logging.INFO)
        import dataclasses
        from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
        # from megatron.training.training import get_model
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.training import get_model
        from megatron.training.checkpointing import load_checkpoint
        from megatron.core.enums import ModelType
        from mindspeed_mm.models.sora_model import SoRAModel
        from mindspeed_mm.patchs.training_patches import get_dist_model_load_from_pt
        from mindspeed_mm.training import no_wd_decay_cond
        from mindspeed_mm.training import scale_lr_cond
        from copy import deepcopy

        # mm的模型提供
        def mm_model_provider(pre_process=True, post_process=True):
            """Builds the model."""
            args = get_args()
            print("building SoRA model ...")
            model = SoRAModel(args.mm.model)
            # if mpu.get_pipeline_model_parallel_world_size() > 1:
            #     if not hasattr(model.predictor, "initialize_pipeline_tensor_shapes"):
            #         raise AttributeError("The predictor should provide initialize_pipeline_tensor_shapes for PP_size>1. ")
            #     args.pipeline_tensor_shapes = model.predictor.initialize_pipeline_tensor_shapes()
            #     setattr(forward_step, 'pipeline_tensor_shapes', args.pipeline_tensor_shapes)
            #     # modification for core 0.12.1 when using multi-parameter PP
            #     model.config.pipeline_tensor_shapes = args.pipeline_tensor_shapes
            return model

        # 初始化actor模型
        self.actor_module_fsdp = get_dist_model_load_from_pt(mm_model_provider, ModelType.encoder_or_decoder)[0]

        from megatron.core.optimizer import get_megatron_optimizer
        from megatron.training.training import get_optimizer_param_scheduler
        # 初始化actor模型的优化器
        kwargs = {}
        # timers = get_timers()
        args = get_args()
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)

        self.actor_optimizer = get_megatron_optimizer(config, [self.actor_module_fsdp], no_wd_decay_cond,
                                                      scale_lr_cond, args.lr_mult,
                                                      use_gloo_process_groups=args.enable_gloo_process_groups)
        self.actor_lr_scheduler = get_optimizer_param_scheduler(self.actor_optimizer)
        torch.distributed.fsdp.register_fsdp_forward_method(self.actor_module_fsdp.module, "predictor_forward")

        # 初始化ref模型
        # self.ref_module_fsdp = get_model(mm_model_provider, ModelType.encoder_or_decoder)[0]
        # 对这个模型的predictor_forward应用DTensor -> Tensor 的转换
        # torch.distributed.fsdp.register_fsdp_forward_method(self.ref_module_fsdp.module, "predictor_forward")

        # from mammothmoda2.model import Mammothmoda2ModelPartition
        from transformers import GenerationConfig
        config = GenerationConfig.from_pretrained(
            self.config.get('model_path', "./"), config_file_name=self.config.get('model_config', "./config.json")
        )

        # Create a custom config for HFRollout that includes diffusion-specific parameters
        rollout_config = self.config.rollout
        actor_config = self.config.actor
        self.actor = DataParallelPPOActor(self.actor_module_fsdp, actor_config, config)
        # self.ref_actor = DataParallelPPOActor(self.ref_module_fsdp, actor_config, config) if self.ref_module_fsdp is not None else None
        set_args("inference")
        self.rollout = HFRollout(self.actor_module_fsdp, rollout_config, config)
        log_gpu_memory_usage("After init_fsdp_module", logger=logger, level=logging.INFO)

    def _compute_grpo_advantages(self, rewards):
        """Compute GRPO-specific advantages"""
        import random

        # Check if we should use group normalization or global normalization
        use_group = self.config.actor.get("use_group", False)
        reward_threshold = self.config.actor.get("reward_threshold", 0.0)

        # Compute advantages based on the chosen normalization method
        if use_group:
            advantages = torch.zeros_like(rewards)
            group_mean = rewards.mean()
            group_std = rewards.std() + 1e-8
            if group_mean < reward_threshold:
                advantages[:] = 0
            else:
                advantages[:] = (rewards - group_mean) / group_std
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, data: DataProto):
        set_args("inference")
        log_gpu_memory_usage("before generate_sequences", logger=logger, level=logging.INFO)
        """Generate sequences using diffusion model with asynchronous processing"""
        # Ensure data has the required attributes for diffusion model
        # Call the rollout's generate_sequence method
        output = self.rollout.generate_sequences(data)
        # Union with original data to preserve all information
        data = data.repeat(repeat_times=self.config.rollout.n, interleave=True)
        data = data.union(output)
        log_gpu_memory_usage("After generate_sequences", logger=logger, level=logging.INFO)
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        """Update actor using GRPO"""
        set_args("train")
        log_gpu_memory_usage("Before update_actor", logger=logger)
        assert self._is_actor
        logger.info(
            f"param_offload: {self.config.actor.fsdp_config.param_offload}, optimizer_offload:{self.config.actor.fsdp_config.optimizer_offload} ")
        # if self.config.actor.fsdp_config.param_offload:
        #     load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # if self.config.actor.fsdp_config.optimizer_offload:
        #     load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        # 根据是否有序列并行决定是否使用分片管理器
        if self.ulysses_sharding_manager is not None:
            context_manager = self.ulysses_sharding_manager
        else:
            # 创建一个空的上下文管理器
            from contextlib import nullcontext
            context_manager = nullcontext()

        with context_manager:
            latents = data.batch["all_latents"]
            old_log_probs = data.batch["all_log_probs"]

            assert data.batch["rewards"] is not None
            rewards = data.batch["rewards"]

            # Get configuration
            actor_config = self.config.actor
            adv_clip_max = actor_config.ppo_adv_clip_max
            kl_coeff = actor_config.ppo_kl_coeff
            timestep_fraction = actor_config.timestep_fraction
            sampling_steps = actor_config.sampling_steps

            # Get sigma schedule
            sigma_schedule = torch.linspace(0, 1, sampling_steps + 1).to(old_log_probs.device)
            sigma_schedule = omni_time_shift(actor_config.shift, sigma_schedule)
            timestep_values = [int((t.item() if torch.is_tensor(t) else float(t)) * 1000) for t in sigma_schedule][
                              :sampling_steps]

            import random
            train_timesteps = random.sample(range(len(timestep_values)), int(len(timestep_values) * timestep_fraction))

            # Initialize losses
            total_loss = 0.0
            total_policy_loss = 0.0
            total_kl_loss = 0.0

            # Perform GRPO update
            self.actor_module_fsdp.train()
            self.actor_optimizer.zero_grad()
            batch_size = latents.shape[0]
            file_path_list = data.non_tensor_batch["prompt_embed_path"]
            grpo_size = self.config.rollout.n
            batch_index = torch.chunk(torch.arange(batch_size), batch_size // grpo_size)
            for i, batch_ind in enumerate(batch_index):
                sample_latents = latents[batch_ind]  # Keep batch dimension
                sample_log_probs = old_log_probs[batch_ind]
                sample_reward = rewards[batch_ind]
                prompt, sample_text_hidden_states, sample_text_attention_mask, sample_negative_text_hidden_states, sample_negative_text_attention_mask = load_repeat_data_train(
                    data.non_tensor_batch, grpo_size, file_path_list[i * grpo_size])
                sample_advantages = self._compute_grpo_advantages(sample_reward)
                mini_batchs_size = sample_latents.shape[0]
                # TODO wan2.2训练
                # Get current and next latents
                start_time = time.time()
                num_chunks = mini_batchs_size // actor_config.micro_batch_size
                batch_indices = torch.chunk(torch.arange(mini_batchs_size), num_chunks)
                log_gpu_memory_usage(f"update_actor batch {i}", logger=logger, level=logging.INFO)
                for idx, batch_idx in enumerate(batch_indices):
                    # prompt_chunk = prompt[batch_idx]
                    prompt_chunk = [prompt[i] for i in batch_idx]
                    text_hidden_states_chunk = sample_text_hidden_states[batch_idx]
                    text_attention_mask_chunk = sample_text_attention_mask[batch_idx]
                    negative_text_hidden_state_chunk = sample_negative_text_hidden_states[
                        batch_idx] if sample_negative_text_hidden_states is not None else None
                    negative_text_attention_mask_chunk = sample_negative_text_attention_mask[
                        batch_idx] if sample_negative_text_attention_mask is not None else None

                    log_probs_chunk = sample_log_probs[batch_idx]
                    latents_chunk = sample_latents[batch_idx]
                    # current_latents = latents_chunk[:, timestep_idx]
                    # next_latents = latents_chunk[:, timestep_idx + 1]
                    # ref_latent_chunk = sample_ref_latents[ref_lat_len*idx:ref_lat_len*(idx+1)]

                    # Calculate new log probs
                    self.actor_module_fsdp.train()
                    new_log_probs, prev_sample_mean, std_dev_t = self.actor.forward_micro_batch(
                        latents_chunk,
                        prompt_chunk,
                        text_hidden_states_chunk,
                        text_attention_mask_chunk,
                        negative_text_hidden_state_chunk,
                        negative_text_attention_mask_chunk,
                    )
                    # Calculate reference log probs if reference model is available
                    if self.ref_module_fsdp is not None:
                        with torch.no_grad():
                            _, prev_sample_mean_ref, _ = self.ref_actor.forward_micro_batch(
                                latents_chunk,
                                prompt_chunk,
                                text_hidden_states_chunk,
                                text_attention_mask_chunk,
                                negative_text_hidden_state_chunk,
                                negative_text_attention_mask_chunk,
                            )

                    # Clamp advantages
                    clamped_advantages = torch.clamp(sample_advantages[batch_idx], -adv_clip_max, adv_clip_max)
                    log_probs_chunk = log_probs_chunk.npu()
                    new_log_probs = new_log_probs.npu()
                    # Calculate policy loss
                    # ratio = torch.exp(new_log_probs - log_probs_chunk[:, timestep_idx])
                    ratio = torch.exp(new_log_probs - log_probs_chunk)
                    unclipped_loss = -clamped_advantages.npu() * ratio.npu()
                    policy_loss = torch.mean(unclipped_loss.npu()) / (latents.npu().shape[0] * len(train_timesteps))

                    # Calculate KL loss if reference model is available
                    if self.ref_module_fsdp is not None:
                        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2), keepdim=True) / (
                                    2 * std_dev_t ** 2)
                        kl_loss = torch.mean(kl_loss)
                        loss = policy_loss + kl_coeff * kl_loss
                    else:
                        loss = policy_loss

                    # Backward pass
                    loss.backward()

                    # Accumulate losses
                    total_policy_loss += policy_loss.item()
                    total_loss += loss.item()
                    avg_loss = loss.detach().clone()
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                    total_loss += avg_loss.item()

                    avg_policy_loss = policy_loss.detach().clone()
                    dist.all_reduce(avg_policy_loss, op=dist.ReduceOp.AVG)
                    total_policy_loss += avg_policy_loss.item()

                    if self.ref_module_fsdp is not None:
                        avg_kl_loss = kl_loss.detach().clone()
                        dist.all_reduce(avg_kl_loss, op=dist.ReduceOp.AVG)
                        total_kl_loss += avg_kl_loss.item()

                rank = torch.distributed.get_rank()
                if rank == 0:
                    end_time = time.time()
                    logger.info(
                        f"Step {i + 1}/{len(train_timesteps)}: Loss {total_loss:.4f}, Policy Loss {total_policy_loss:.4f}, total_kl_loss {total_kl_loss:.4f}, Time {end_time - start_time:.4f}")

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor_module_fsdp.parameters(), actor_config.ppo_max_grad_norm)

            # wan2.2 optimizer
            self.actor_optimizer.step()
            # wan2.2 scheduler
            self.actor_lr_scheduler.step(batch_size)

            # Synchronize across processes
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Create metrics dictionary
            metrics = {
                "total_loss": total_loss,
                "policy_loss": total_policy_loss,
                "kl_loss": total_kl_loss,
            }
            output = DataProto(meta_info={"metrics": metrics})
            log_gpu_memory_usage(f"After update_actor", logger=logger, level=logging.INFO)
            return output


# Helper functions moved from train_grpo_edit.py
def omni_time_shift(shift, t):
    t = 1 - t
    t = (shift * t) / (1 + (shift - 1) * t)
    t = 1 - t
    return t