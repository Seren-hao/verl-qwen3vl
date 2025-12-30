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
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""
import contextlib
import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto

from verl.workers.rollout.base import BaseRollout
from verl.utils.device import get_device_name, get_torch_device
from recipe.dance_grpo_mm.utils import load_repeat_data
import os
import logging
from loguru import logger

import os

import torch
import mindspeed.megatron_adaptor
import torch.distributed as dist
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.global_vars import set_args
from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder, Tokenizer
from mindspeed_mm.utils.utils import get_device
from recipe.dance_grpo_mm.modeling_inference_wan2_2 import ModelingInferenceWan2_2
from verl.workers.config import ActorConfig

__all__ = ["HFRollout"]


class HFRollout:
    def __init__(self, module: nn.Module, config: ActorConfig, model_config: ActorConfig):
        self.config = config
        self.module = module
        # set_args('inference')
        args = get_args()
        # device = get_device(args.device)
        if not hasattr(args, "dist_train"):
            args.dist_train = False
        model_args = args.mm.model
        # 训练的tokenizer在数据集处理中
        self.tokenizer = Tokenizer(model_args.tokenizer).get_tokenizer()
        scheduler = DiffusionModel(model_args.diffusion).get_model()
        self.sora_rollout = ModelingInferenceWan2_2(module, self.tokenizer, scheduler, model_config)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        args = get_args()
        merge_mm_args(args)
        if not hasattr(args, "dist_train"):
            args.dist_train = False
        args.mm.model.micro_batch_size = args.micro_batch_size
        args = args.mm.model
        device = 'npu'
        prompt, prompt_embed, prompt_attention_mask, negative_prompt_embed, negative_prompt_attention_mask = load_repeat_data(
            prompts.non_tensor_batch, self.config.n)
        import contextlib
        param_ctx = contextlib.nullcontext()
        # is_test = os.getenv("IS_TEST", "TRUE").upper() in ["TRUE", "1"]
        # if is_test:
        #     prompt_embed = torch.randn(
        #         (prompt_embed.shape[0], prompt_embed.shape[1], 3584),
        #         device=prompt_embed.device,
        #         dtype=torch.bfloat16,
        #     )
        #     negative_prompt_embed = torch.randn(
        #         (negative_prompt_embed.shape[0], negative_prompt_embed.shape[1], 3584),
        #         device=prompt_embed.device,
        #         dtype=torch.bfloat16,
        #     )

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        imgs_list, all_latents_list, all_log_probs_list = [], [], []
        grpo_size = self.config.n
        # latent_w, latent_h = self.config.latent_w, self.config.latent_h
        # if self.config.init_same_noise:
        #     input_latents = torch.randn(
        #         (1, 3, latent_w, latent_h),
        #         device=prompt_embed.device,
        #         dtype=torch.bfloat16,
        #     )
        #     input_latents = input_latents.repeat(grpo_size, 1, 1, 1)
        # else:
        #     input_latents = torch.randn(
        #         (grpo_size, 3, latent_w, latent_h),
        #         device=prompt_embed.device,
        #         dtype=torch.bfloat16,
        #     )
        num_chunks = grpo_size // self.config.micro_batch_size
        batch_indices = torch.chunk(torch.arange(grpo_size), num_chunks)

        for index, chunk in enumerate(batch_indices):
            prompt = [prompt[i] for i in chunk]
            sample_text_hidden_states = prompt_embed[chunk]
            sample_text_attention_mask = prompt_attention_mask[chunk]
            sample_negative_text_hidden_states = negative_prompt_embed[
                chunk] if negative_prompt_embed is not None else None
            sample_negative_text_attention_mask = negative_prompt_attention_mask[
                chunk] if negative_prompt_attention_mask is not None else None

            # with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                imgs, all_latents, all_log_probs = self.sora_rollout.generate(
                    prompt,
                    sample_text_hidden_states,
                    sample_text_attention_mask,
                    sample_negative_text_hidden_states,
                    sample_negative_text_attention_mask,
                )
            imgs_list += imgs
            all_latents_list.append(all_latents)
            all_log_probs_list.append(all_log_probs)
        all_latents_list, all_log_probs_list = torch.cat(all_latents_list), torch.cat(all_log_probs_list)
        batch = DataProto.from_dict(
            tensors={
                "all_latents": all_latents_list,
                "all_log_probs": all_log_probs_list,
            },
            non_tensors={
                "all_imgs": np.array(imgs_list, dtype=object),
            },
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()
        return batch

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        file_index = prompts.non_tensor_batch['prompt_embed_path']
        batch_size = len(file_index)
        batch_prompts = prompts.chunk(chunks=batch_size)
        output = [self._generate_minibatch(p[0]) for p in batch_prompts]
        output = DataProto.concat(output)
        repeat_file_index = []
        for index in file_index:
            repeat_file_index.extend([index] * self.config.n)
        output.non_tensor_batch['file_index'] = repeat_file_index
        return output

    def release(self):
        pass

    def resume(self):
        pass

    def update_weights(self, data: DataProto) -> dict:
        pass