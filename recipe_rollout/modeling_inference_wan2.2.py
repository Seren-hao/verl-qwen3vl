# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union
import html
import math
import os

from PIL.Image import Image
import ftfy
import regex as re
import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.functional import center_crop
from transformers import CLIPVisionModel
from megatron.training import get_args
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from mindspeed_mm.utils.utils import get_device
from mindspeed_mm.models.predictor import PredictModel
from diffusers.utils.torch_utils import randn_tensor
from mindspeed_mm.models.diffusion import DiffusionModel
import imageio

# from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
# from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
# from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin

NEGATIVE_PROMOPT_DEFAULT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def omni_time_shift(shift, t):
    t = 1 - t
    t = (shift * t) / (1 + (shift - 1) * t)
    t = 1 - t
    return t


def wan2_2_step_aligned(
        model_output: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        timesteps: torch.Tensor,
        index: int,
        prev_sample: torch.Tensor,
        grpo: bool,
        sde_solver: bool,
        eps_schedule: str,
        eps_power: float,
        eps_min_ratio: float,
        generators: list | None = None,
):
    t = timesteps[index]
    t_next = timesteps[index + 1]
    dt = t_next - t
    prev_sample_mean = latents + dt * model_output
    x_hat = latents + (1.0 - t) * model_output
    pred_original_sample = x_hat
    score_estimate = -(latents - t * x_hat) / ((1.0 - t) ** 2 + 1e-12)

    eta_t = eta  # compute_eta_t(eta, t, schedule=eps_schedule, power=eps_power, min_ratio=eps_min_ratio)
    std_dev_t = eta_t * torch.sqrt(torch.abs(dt) + 1e-12)

    if sde_solver:
        prev_sample_mean = prev_sample_mean + (0.5 * (eta_t ** 2)) * score_estimate * dt

    if grpo and prev_sample is None:
        if generators is not None and len(generators) == prev_sample_mean.shape[0]:
            from diffusers.utils.torch_utils import randn_tensor
            noise = randn_tensor(
                prev_sample_mean.shape,
                generator=generators,
                device=prev_sample_mean.device,
                dtype=prev_sample_mean.dtype,
            )
        else:
            noise = torch.randn_like(prev_sample_mean)
        prev_sample = prev_sample_mean + noise * std_dev_t

    if grpo:
        diff = (prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32))
        std32 = std_dev_t.to(torch.float32)
        var_t = (std32 ** 2).clamp_min(1e-20)
        log_std = torch.log(std32.clamp_min(1e-20))
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=std_dev_t.device, dtype=torch.float32))
        log_prob = (-(diff ** 2) / (2 * var_t)) - log_std - 0.5 * log_two_pi
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t
    else:
        return prev_sample_mean, pred_original_sample


def compute_eta_t(
        eta: float,
        t: torch.Tensor,
        schedule: str = "constant",
        power: float = 2.0,
        min_ratio: float = 0.0,
):
    if schedule == "constant":
        scale = torch.ones_like(t)
    elif schedule == "linear":
        scale = 1.0 - t
    elif schedule == "cosine":
        scale = 0.5 * (1.0 + torch.cos(torch.pi * t))
    elif schedule == "poly":
        scale = torch.clamp(1.0 - t, min=0.0) ** power
    else:
        scale = torch.ones_like(t)
    if min_ratio > 0.0:
        scale = torch.clamp(scale, min=min_ratio)
    return torch.as_tensor(eta, dtype=t.dtype, device=t.device) * scale


# class ModelingInferenceWan2_2(MMPipeline, InputsCheckMixin, MMEncoderMixin):
class ModelingInferenceWan2_2():
    def __init__(self, actor_module: nn.Module, tokenizer, scheduler: DiffusionModel, config=None):
        super().__init__()
        self.config = config
        self.wan2_2_model = actor_module
        self.predictor = self.wan2_2_model.module.predictor
        self.vae_model = self.wan2_2_model.module.ae.model.model
        self.text_encoder = self.wan2_2_model.module.text_encoder.text_encoders
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.num_frames, self.height, self.width = 45, 704, 1280
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16
        self.generator = None if not hasattr(config, "seed") else torch.Generator().manual_seed(config.seed)
        self.expand_timesteps = getattr(config, "expand_timesteps", False)
        sigma_schedule = torch.linspace(0, 1, self.config.sample_steps + 1)
        self.sigma_schedule = omni_time_shift(self.config.shift, sigma_schedule)

    def prepare_latents(self, shape, generator, device, dtype, latents=None):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def generate(self, prompt, text_hidden_states, text_attention_mask, negative_text_hidden_states,
                 negative_text_attention_mask, max_sequence_length: int = 512):
        device = 'npu'
        batch_size = len(prompt)
        do_classifier_free_guidance = self.scheduler.do_classifier_free_guidance
        prompt_embeds, negative_prompt_embeds = self.encode_texts(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        shape = (
            batch_size,
            self.predictor.in_dim,
            (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            self.height // self.vae_scale_factor_spatial,
            self.width // self.vae_scale_factor_spatial,
        )
        latents = self.prepare_latents(shape, generator=self.generator, device=device, dtype=prompt_embeds.dtype)
        clip_features, vae_features = None, None
        first_frame_mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        model_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features,
        }

        # 5. Denoising to get clean latents
        num_inference_steps = self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps

        num_warmup_steps = self.scheduler.num_warmup_steps
        guidance_scale = self.scheduler.guidance_scale
        self.scheduler.diffusion.set_timesteps(num_inference_steps)  # reset timesteps

        for i, t in enumerate(timesteps):
            # dt = self.sigma_schedule[i + 1] - self.sigma_schedule[i]
            if True:
                if False:
                    latent_model_input = (1 - first_frame_mask) * vae_features + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(self.predictor.dtype)
                else:
                    latent_model_input = latents.to(self.predictor.dtype)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                # timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1).float()
                timestep = temp_ts.unsqueeze(0).expand(1, -1).float()
            else:
                latent_model_input = latents.to(self.predictor.dtype)
                timestep = t.to(device=latents.device).float()

            # curr_predict_model, curr_guidance_scale = self._prepare_predict_model(t, guidance_scale)
            curr_guidance_scale = guidance_scale[0] if isinstance(guidance_scale, (list, tuple)) else guidance_scale

            self.wan2_2_model.module.predictor.seperated_timestep = False
            # self.wan2_2_model.module.predictor.attention_async_offload=False
            noise_pred = self.wan2_2_model.module.predictor_forward(
                latent_model_input, timestep, model_kwargs.get("prompt_embeds"), **model_kwargs
            )[0]

            if do_classifier_free_guidance:
                noise_uncond = self.wan2_2_model.module.predictor_forward(
                    latent_model_input, timestep, model_kwargs.get("negative_prompt_embeds"), **model_kwargs
                )[0]
                noise_pred = noise_uncond + curr_guidance_scale * (noise_pred - noise_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # TODO 这里使用wan2.2计算下一帧视频
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            # TODO 暂时复用猛犸2的logProb计算逻辑
            if False:
                # Use SDE step aligned with mammoth2 SDE
                latent_model_input, pred_original, log_prob, prev_sample_mean, std_dev_t = wan2_2_step_aligned(
                    noise_pred,
                    latent_model_input.to(torch.float32),
                    self.config.eta,
                    self.sigma_schedule,
                    i,
                    prev_sample=None,
                    grpo=True,
                    sde_solver=True,
                    eps_schedule=self.config.eps_schedule,
                    eps_power=self.config.eps_power,
                    eps_min_ratio=self.config.eps_min_ratio,
                    generators=None,
                )
                latent_model_input = latent_model_input.to(torch.bfloat16)
                # all_latents.append(z)
                # all_log_probs.append(log_prob)
            else:
                pass
                # # Deterministic Euler step to align with SRPO sampling
                # latent_model_input = (latent_model_input.to(torch.float32) + dt * noise_pred.to(torch.float32)).to(torch.bfloat16)

                # # Compute a consistent per-step log_prob under zero-noise assumption for PPO bookkeeping
                # eta_t = compute_eta_t(self.config.eta, torch.as_tensor(t, device=z.device, dtype=torch.float32), schedule=self.config.eps_schedule, power=self.config.eps_power, min_ratio=self.config.eps_min_ratio)
                # std_dev_t = eta_t * torch.sqrt(torch.abs(torch.as_tensor(dt, device=z.device, dtype=torch.float32)) + 1e-12)
                # # prev_mean = (all_latents[-1].to(torch.float32) + dt * noise_pred.to(torch.float32))
                # prev_mean = (latent_model_input.to(torch.float32) + dt * noise_pred.to(torch.float32))
                # prev_sample = prev_mean  # zero noise path
                # diff = (prev_sample - prev_mean)
                # std32 = std_dev_t.to(torch.float32)
                # var_t = (std32 ** 2).clamp_min(1e-20)
                # log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=std_dev_t.device, dtype=torch.float32))
                # # elementwise log_prob then sum over non-batch dims
                # elem_log_prob = (-(diff ** 2) / (2 * var_t)) - torch.log(std32.clamp_min(1e-20)) - 0.5 * log_two_pi
                # log_prob = elem_log_prob.mean(dim=tuple(range(1, elem_log_prob.ndim)))

                # all_latents.append(z)
                # all_log_probs.append(log_prob)
        # latents = latent_model_input.to(self.vae_model.dtype)
        video_latents = latents[:1].to(self.vae_model.dtype)
        latents_mean = (
            torch.tensor(self.vae_model.config.latents_mean)
            .view(1, self.vae_model.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae_model.config.latents_std).view(
            1, self.vae_model.config.z_dim, 1, 1, 1
        ).to(video_latents.device, video_latents.dtype)
        video_latents = video_latents / latents_std + latents_mean
        video = self.decode_latents(video_latents)
        # TODO 保存视频
        save_videos(video, 0, '/home/r30009656/verl-mengma-fsdp-lh/result/', 16)
        # decoded_image = self.image_processor.postprocess(video)
        video = [video] * 2
        # TODO 临时随机log_prob
        log_prob = torch.tensor([0.9364, 0.9364]).to(device=latents.device)
        # log_prob=log_prob.unsqueeze(1).expand(2, latents.shape[2]-1)
        return video, latents, log_prob

    def decode_latents(self, latents, value_range=(-1, 1), normalize=True, **kwargs):
        video = self.vae_model.decode(latents, **kwargs)  # [b, c, t, h, w]
        video = video.sample
        if normalize:
            low, high = value_range
            video.clamp_(min=low, max=high)
            video.sub_(low).div_(max(high - low, 1e-5))
        # [b, c, t, h, w] --> [b, t, h, w, c]
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1).to("cpu", torch.uint8)
        return video

    def encode_texts(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            do_classifier_free_guidance: bool = True,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            max_sequence_length: int = 226,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prompt_preprocess(self, prompt):

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            return text

        return whitespace_clean(basic_clean(prompt))

    def _get_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            max_sequence_length: int = 226,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        prompt = [self.prompt_preprocess(u) for u in prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        self.text_encoder = self.text_encoder.to(device)
        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        return prompt_embeds.to(self.predictor.dtype)


def save_videos(videos, start_index, save_path, fps):
    os.makedirs(save_path, exist_ok=True)
    if isinstance(videos, (list, tuple)) or videos.ndim == 5:  # [b, t, h, w, c]
        for i, video in enumerate(videos):
            save_path_i = os.path.join(save_path, f"video_{start_index + i}.mp4")
            imageio.mimwrite(save_path_i, video, fps=fps, quality=6)
    elif videos.ndim == 4:
        save_path = os.path.join(save_path, f"video_{start_index}.mp4")
        imageio.mimwrite(save_path, video, fps=fps, quality=6)
    else:
        raise ValueError("The video must be in either [b, t, h, w, c] or [t, h, w, c] format.")