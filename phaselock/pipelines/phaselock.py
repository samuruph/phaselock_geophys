"""The two-stage PhaseLock pipeline, driven by a backend rather than a fixed model.

Stage 1 runs a few-step generation and extracts its latent deltas as a motion prior.
Stage 2 re-runs the same seed at full length with those deltas as a guidance target.

This is the Wan2.1 support deliverable. Nothing in the experiment drivers uses it -- all
measurements run on plain baseline sampling -- but it is kept correct and layout-aware so
that Wan works alongside CogVideoX.
"""

from __future__ import annotations

import gc
from typing import Any, List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..backends.base import VideoBackend
from ..guidance import LatentDeltaGuidance, extract_prior


class PhaseLockPipeline:
    """Adds Latent Delta Guidance to any registered backend."""

    def __init__(
        self,
        backend: VideoBackend,
        few_steps: int = 2,
        full_steps: int = 50,
        guidance_strength: float = 0.05,
        guide_start: int = 0,
        guide_end: Optional[int] = None,
        few_step_prior_type: str = "motion",
    ):
        self.backend = backend
        self.few_steps = few_steps
        self.full_steps = full_steps
        self.guidance_strength = guidance_strength
        self.guide_start = guide_start
        self.guide_end = guide_end if guide_end is not None else full_steps // 2
        # Which quantity the few-step pass is mined for and the full pass is held to.
        # "motion" is PhaseLock as published; the rest are the ablation.
        self.few_step_prior_type = few_step_prior_type
        self.last_prior_rms: float = float("nan")

    @property
    def pipe(self) -> Any:
        return self.backend.pipe

    @classmethod
    def from_backend(cls, name: str, **kwargs: Any) -> "PhaseLockPipeline":
        """Load a registered backend and wrap it."""
        from ..backends import load_backend

        phaselock_keys = {
            "few_steps",
            "full_steps",
            "guidance_strength",
            "guide_start",
            "guide_end",
        }
        settings = {k: v for k, v in kwargs.items() if k in phaselock_keys}
        loader = {k: v for k, v in kwargs.items() if k not in phaselock_keys}
        return cls(load_backend(name, **loader), **settings)

    def _frames_to_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """PIL frames from a pipeline call -> ``(F, 3, H, W)`` in [0, 1]."""
        if isinstance(frames, torch.Tensor):
            return frames
        return torch.stack([TF.to_tensor(frame) for frame in frames])

    def __call__(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 6.0,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
        return_few_result: bool = False,
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], List[Image.Image]]]:
        """Generate with motion-prior guidance.

        Frame count, resolution and frame rate default to the backend's native settings
        -- 49 frames at 480x720 for CogVideoX, 81 at 480x832 for Wan -- which is what the
        paper used per model.
        """
        spec = self.backend.spec
        device = self.backend.device

        shared = self.backend.generation_kwargs(
            prompt=prompt,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )

        # Stage 1: capture the motion prior from a few-step pass.
        few_result = self.pipe(
            **shared,
            num_inference_steps=self.few_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).frames[0]

        few_latents = self.backend.encode(self._frames_to_tensor(few_result).to(device))
        motion_prior = extract_prior(few_latents, few_step_prior_type=self.few_step_prior_type)
        # Recorded so the ablation can answer whether one strength is comparable across
        # operators. lambda = 0.05 was tuned for first differences; each further
        # difference amplifies whatever noise the 2-step pass carries, so a higher-order
        # prior can be larger by a factor that makes the same lambda a different
        # intervention. If these differ by orders of magnitude, the arms are not yet a
        # fair comparison and need a per-arm strength.
        self.last_prior_rms = float(motion_prior.float().pow(2).mean().sqrt())

        guidance = LatentDeltaGuidance(
            motion_prior=motion_prior,
            spec=spec,
            guidance_strength=self.guidance_strength,
            guide_start=self.guide_start,
            guide_end=self.guide_end,
            total_steps=self.full_steps,
            few_step_prior_type=self.few_step_prior_type,
        )

        # Stage 2: same seed, full length, guided toward the prior.
        final_result = self.pipe(
            **shared,
            num_inference_steps=self.full_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            callback_on_step_end=guidance,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]

        del few_latents, motion_prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if return_few_result:
            return final_result, few_result
        return final_result
