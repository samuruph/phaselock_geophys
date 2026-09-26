# Copyright 2024 The CogVideoX team, Tsinghua University & Zhipu AI, and HuggingFace Inc.
# Licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
# Input preparation and output handling follow the installed diffusers CogVideoX I2V pipeline.
"""Isolated CogVideoX I2V DDIM loop with same-timestep P&P iterations."""
from __future__ import annotations

import math
import time

import torch
from diffusers import CogVideoXDDIMScheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import (
    CogVideoXPipelineOutput,
    retrieve_timesteps,
)

from .refinement import refine_step


def run_refinement(backend, controller, refinement, **kwargs):
    """Run an isolated sampler; restore the loaded scheduler even after failure."""
    if backend.mode != "i2v" or not backend.spec.name.startswith("cogvideox"):
        raise ValueError("P&P currently supports CogVideoX I2V only")
    pipe = backend.pipe
    original = pipe.scheduler
    scheduler = CogVideoXDDIMScheduler.from_config(original.config)
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError("CogVideoX P&P requires v_prediction")
    metrics = {
        "guided_predictions": 0,
        "transformer_calls": 0,
        "sampler_class": type(scheduler).__name__,
        "sampler_config": dict(scheduler.config),
    }
    auxiliary = torch.Generator(device=backend.device).manual_seed(refinement.seed)
    started = time.perf_counter()
    try:
        pipe.scheduler = scheduler
        result = _sample(
            pipe, backend=backend, controller=controller, refinement=refinement,
            refinement_generator=auxiliary, metrics=metrics, **kwargs,
        )
        if torch.device(backend.device).type == "cuda":
            torch.cuda.synchronize(backend.device)
        metrics["wall_seconds"] = time.perf_counter() - started
        return result, metrics
    finally:
        pipe.scheduler = original
        pipe._current_timestep = None
        pipe.maybe_free_model_hooks()


@torch.no_grad()
def _sample(
    pipe, *, backend, controller, refinement, refinement_generator, metrics,
    image, prompt=None, negative_prompt=None, height=None, width=None,
    num_frames=49, num_inference_steps=50, guidance_scale=6.0,
    generator=None, output_type="pil", use_dynamic_cfg=False,
    attention_kwargs=None, eta=0.0, **unused,
):
    """Keep native CogVideoX I2V preparation and CFG; replace the denoising loop."""
    if unused:
        raise ValueError(f"unsupported P&P pipeline arguments: {sorted(unused)}")
    height = height or pipe.transformer.config.sample_height * pipe.vae_scale_factor_spatial
    width = width or pipe.transformer.config.sample_width * pipe.vae_scale_factor_spatial
    num_frames = num_frames or pipe.transformer.config.sample_frames
    pipe.check_inputs(
        image=image, prompt=prompt, height=height, width=width,
        negative_prompt=negative_prompt, callback_on_step_end_tensor_inputs=["latents"],
        latents=None, prompt_embeds=None, negative_prompt_embeds=None,
    )
    pipe._guidance_scale = guidance_scale
    pipe._current_timestep = None
    pipe._attention_kwargs = attention_kwargs
    pipe._interrupt = False
    device = pipe._execution_device
    use_cfg = guidance_scale > 1.0
    prompt_embeds, negative_embeds = pipe.encode_prompt(
        prompt=prompt, negative_prompt=negative_prompt,
        do_classifier_free_guidance=use_cfg, num_videos_per_prompt=1,
        prompt_embeds=None, negative_prompt_embeds=None,
        max_sequence_length=226, device=device,
    )
    if use_cfg:
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, None,
    )
    pipe._num_timesteps = len(timesteps)

    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    patch_size_t = pipe.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * pipe.vae_scale_factor_temporal
    image = pipe.video_processor.preprocess(image, height=height, width=width).to(
        device, dtype=prompt_embeds.dtype,
    )
    latent_channels = pipe.transformer.config.in_channels // 2
    latents, image_latents = pipe.prepare_latents(
        image, 1, latent_channels, num_frames, height, width,
        prompt_embeds.dtype, device, generator, None,
    )
    if eta != 0.0:
        raise ValueError("P&P comparison requires deterministic DDIM (eta=0)")
    rotary = (
        pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if pipe.transformer.config.use_rotary_positional_embeddings else None
    )
    ofs = (
        None if pipe.transformer.config.ofs_embed_dim is None
        else latents.new_full((1,), fill_value=2.0)
    )

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for index, timestep in enumerate(timesteps):
            if pipe.interrupt:
                continue
            pipe._current_timestep = timestep
            if use_dynamic_cfg:
                progress = ((num_inference_steps - timestep.item()) / num_inference_steps) ** 5.0
                pipe._guidance_scale = 1 + guidance_scale * (1 - math.cos(math.pi * progress)) / 2

            def predict(candidate, current_timestep):
                model_input = torch.cat([candidate] * 2) if use_cfg else candidate
                model_input = pipe.scheduler.scale_model_input(model_input, current_timestep)
                condition = torch.cat([image_latents] * 2) if use_cfg else image_latents
                model_input = torch.cat([model_input, condition], dim=2)
                model_timestep = current_timestep.expand(model_input.shape[0])
                with pipe.transformer.cache_context("cond_uncond"):
                    prediction = pipe.transformer(
                        hidden_states=model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=model_timestep, ofs=ofs, image_rotary_emb=rotary,
                        attention_kwargs=attention_kwargs, return_dict=False,
                    )[0].float()
                if use_cfg:
                    uncond, cond = prediction.chunk(2)
                    prediction = uncond + pipe.guidance_scale * (cond - uncond)
                return prediction

            active = controller.guide_start <= index < controller.guide_end
            iterations = refinement.steps_per_timestep if active else 0
            record = (
                controller.recorder is not None
                and (controller.recorder.record_steps is None
                     or index in controller.recorder.record_steps)
            )
            started = time.perf_counter()
            latents, state, confidence, extras = refine_step(
                backend, pipe.scheduler, latents, timestep, predict, iterations,
                refinement_generator, record=record,
                save_raw=record and controller.recorder.save_raw,
            )
            metrics["guided_predictions"] += iterations + 1
            metrics["transformer_calls"] += iterations + 1
            if record:
                if latents.is_cuda:
                    torch.cuda.synchronize(latents.device)
                extras["outer_step_seconds"] = time.perf_counter() - started
                extras["cumulative_guided_predictions"] = metrics["guided_predictions"]
            latents = controller.apply_step(
                pipe, index, timestep, latents, state,
                confidence=confidence if refinement.confidence_gate else None,
                extras=extras,
            )
            progress_bar.update()
    pipe._current_timestep = None
    if output_type == "latent":
        frames = latents
    else:
        decoded = pipe.decode_latents(latents[:, additional_frames:])
        frames = pipe.video_processor.postprocess_video(decoded, output_type=output_type)
    return CogVideoXPipelineOutput(frames=frames)
