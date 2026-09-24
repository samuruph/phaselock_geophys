"""The two-stage PhaseLock pipeline, driven by a backend rather than a fixed model.

Stage 1 runs a few-step generation and extracts its latent deltas as a motion prior.
Stage 2 re-runs the same seed at full length with those deltas as a guidance target.

This is the Wan2.1 support deliverable. Nothing in the experiment drivers uses it -- all
measurements run on plain baseline sampling -- but it is kept correct and layout-aware so
that Wan works alongside CogVideoX.
"""

from __future__ import annotations

import gc
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..backends.base import VideoBackend
from ..guidance import LatentDeltaGuidance, RunningMomentumGuidance, extract_prior
from ..analysis.momentum_diagnostics import MomentumTrace, StepPredictionCapture
        

class _FinalStateCapture:
    """Records the denoiser's own view at the last step of a pass.

    ``x0_hat`` and ``velocity`` are not the sampler state, so they cannot be recovered by
    re-encoding the decoded video the way ``latent`` is. They exist only alongside a
    prediction, so they are captured in the loop and the last step's values are kept --
    that being the pass's final belief about the clean video, which is the analogue of
    "the latents the few-step pass ended at".
    """

    def __init__(self, backend: Any):
        self.backend = backend
        self._previous: Optional[torch.Tensor] = None
        self.state = None

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        noise_pred = callback_kwargs.get("noise_pred")
        previous, self._previous = self._previous, latents
        # `noise_pred` belongs to the state that went *into* this step, which is what the
        # previous callback returned. Step 0 has no predecessor and is skipped.
        if previous is not None and noise_pred is not None:
            self.state = self.backend.denoiser_state(previous, noise_pred, timestep)
        return callback_kwargs

    def trajectory(self, source: str) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError(
                "no denoiser state captured. A few-step pass needs at least two steps for "
                "one of them to have a predecessor; raise phaselock.few_steps."
            )
        return self.state.x0 if source == "x0_hat" else self.state.drift


class PhaseLockPipeline:
    """Adds Latent Delta Guidance to any registered backend."""

    def __init__(
        self,
        backend: VideoBackend,
        prior_mode: str = "few_step",
        betas: Tuple[float, float] = (0.9, 0.999),
        running_momentum_mode: str = "residual",
        few_steps: int = 2,
        full_steps: int = 50,
        guidance_strength: float = 0.05,
        guide_start: int = 0,
        guide_end: Optional[int] = None,
        few_step_prior_type: str = "motion",
        source: str = "latent",
    ):
        self.backend = backend
        self.prior_mode = prior_mode
        if self.prior_mode == "running_momentum":
            self.betas = betas
            self.running_momentum_mode = running_momentum_mode
        self.few_steps = few_steps
        self.full_steps = full_steps
        self.guidance_strength = guidance_strength
        self.guide_start = guide_start
        self.guide_end = guide_end if guide_end is not None else full_steps // 2
        # Which quantity the few-step pass is mined for and the full pass is held to.
        # "motion" is PhaseLock as published; the rest are the ablation.
        self.few_step_prior_type = few_step_prior_type
        # Which tensor the operator is measured on. "latent" is the sampler state and is
        # PhaseLock's own; the others are model outputs and need `noise_pred`.
        self.source = source
        if self.prior_mode == "running_momentum" and source not in {"latent", "x0_hat", "blend"}:
            raise ValueError("running momentum source must be 'latent', 'x0_hat', or 'blend'")
        self.last_prior_rms: float = float("nan")

    @property
    def pipe(self) -> Any:
        return self.backend.pipe

    def _callback_inputs_with_noise(self) -> list:
        allowed = list(getattr(self.pipe, "_callback_tensor_inputs", []) or [])
        if "noise_pred" not in allowed:
            self.pipe._callback_tensor_inputs = list(allowed) + ["noise_pred"]
        return ["latents", "noise_pred"]

    def _callback_inputs(self) -> list:
        """Which tensors the sampler must hand the callback.

        ``noise_pred`` is not in diffusers' default allow-list, but that list is a plain
        attribute checked for membership and the value is in scope where the callback is
        invoked -- and by then it is already CFG-combined, so nothing about guidance has
        to be reimplemented. Extended on the instance, never on the class.
        """
        if self.source == "latent":
            return ["latents"]
        allowed = list(getattr(self.pipe, "_callback_tensor_inputs", []) or [])
        if "noise_pred" not in allowed:
            self.pipe._callback_tensor_inputs = list(allowed) + ["noise_pred"]
        return ["latents", "noise_pred"]

    @classmethod
    def from_backend(cls, name: str, **kwargs: Any) -> "PhaseLockPipeline":
        """Load a registered backend and wrap it."""
        from ..backends import load_backend

        phaselock_keys = {
            "prior_mode",
            "few_steps",
            "full_steps",
            "guidance_strength",
            "guide_start",
            "guide_end",
            "source",
            "betas",
            "running_momentum_mode",
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
        diagnostic_recorder: Optional[MomentumTrace] = None,
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

        # Option A: running-momentum guidance, without freezed few-step guidance
        if self.prior_mode == "running_momentum":
            prediction_capture = (
                StepPredictionCapture(self.backend, guidance_scale)
                if diagnostic_recorder is not None or self.source in {"x0_hat", "blend"}
                else None
            )
            guidance = RunningMomentumGuidance(
                spec=spec,
                guidance_strength=self.guidance_strength,
                guide_start=self.guide_start,
                guide_end=self.guide_end,
                mode=self.running_momentum_mode,
                betas=self.betas,
                recorder=diagnostic_recorder,
                backend=self.backend,
                prediction_capture=prediction_capture,
                source=self.source,
            )

            capture_context = prediction_capture if prediction_capture is not None else nullcontext()
            with capture_context:
                final_result = self.pipe(
                    **shared,
                    num_inference_steps=self.full_steps,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    callback_on_step_end=guidance,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).frames[0]

            if return_few_result:
                return final_result, None

            return final_result

        # Option B: few-step guidance, with a two-stage pass. This is the original PhaseLock.
        # Stage 1: capture the prior from a few-step pass.
        capture = _FinalStateCapture(self.backend) if self.source != "latent" else None
        few_result = self.pipe(
            **shared,
            num_inference_steps=self.few_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            # _callback_inputs, not a literal: requesting noise_pred without extending
            # the pipeline's allow-list is rejected by diffusers' own validation, and the
            # few-step pass needs it just as much as the guided one.
            **({"callback_on_step_end": capture,
                "callback_on_step_end_tensor_inputs": self._callback_inputs()}
               if capture is not None else {}),
        ).frames[0]

        if capture is None:
            # PhaseLock takes the prior from the decoded-then-re-encoded few-step video,
            # not from the in-loop latents. Kept exactly so for the reproduction.
            source_trajectory = self.backend.encode(
                self._frames_to_tensor(few_result).to(device)
            )
        else:
            source_trajectory = capture.trajectory(self.source)
        motion_prior = extract_prior(
            source_trajectory, few_step_prior_type=self.few_step_prior_type
        )
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
            source=self.source,
            backend=self.backend if self.source != "latent" else None,
        )

        # Stage 2: same seed, full length, guided toward the prior.
        final_result = self.pipe(
            **shared,
            num_inference_steps=self.full_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            callback_on_step_end=guidance,
            callback_on_step_end_tensor_inputs=self._callback_inputs(),
        ).frames[0]

        del source_trajectory, motion_prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if return_few_result:
            return final_result, few_result
        return final_result
