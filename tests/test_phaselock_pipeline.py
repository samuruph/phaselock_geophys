"""Drive `PhaseLockPipeline.__call__` end to end on CPU, with a fake sampler.

Two bugs reached a GPU run because nothing here existed: a stale variable name in the
cleanup that only the guided path touched, and `linalg.pinv` refusing the bfloat16 latents
a real sampler hands the callback. Both were a `python -m pytest` away from being caught,
and both cost a model load plus four minutes of generation to discover.

The fake pipe is deliberately thin -- it does not denoise anything. It exists to run the
*orchestration*: the few-step pass, the prior extraction, the callback contract, the
cleanup, and the return shape.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from phaselock.backends.base import DenoiserState, from_canonical, to_canonical
from phaselock.backends.cogvideox import COGVIDEOX_5B
from phaselock.pipelines.phaselock import PhaseLockPipeline

FRAMES, CHANNELS, HEIGHT, WIDTH = 13, 16, 6, 8
VIDEO_FRAMES = 4


class _FakeResult:
    def __init__(self, frames):
        self.frames = [frames]


class _FakePipe:
    """Records how it was called, and honours the callback contract diffusers offers."""

    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype
        self.calls: list[dict] = []

    def __call__(self, num_inference_steps, generator=None, callback_on_step_end=None,
                 callback_on_step_end_tensor_inputs=None, **kwargs):
        self.calls.append({
            "steps": num_inference_steps,
            "tensor_inputs": list(callback_on_step_end_tensor_inputs or []),
        })
        latents = from_canonical(
            torch.randn(FRAMES, CHANNELS, HEIGHT, WIDTH, dtype=self.dtype), COGVIDEOX_5B
        )
        for step in range(num_inference_steps):
            if callback_on_step_end is None:
                continue
            available = {
                "latents": latents,
                "prompt_embeds": torch.zeros(1),
                "noise_pred": from_canonical(
                    torch.randn(FRAMES, CHANNELS, HEIGHT, WIDTH, dtype=self.dtype),
                    COGVIDEOX_5B,
                ),
            }
            requested = {k: available[k] for k in (callback_on_step_end_tensor_inputs or [])}
            # Exactly diffusers' contract: anything not in the allow-list is a bug.
            for key in requested:
                assert key in self._callback_tensor_inputs, f"{key} was never allowed"
            out = callback_on_step_end(self, step, torch.tensor(500.0), requested)
            latents = out.get("latents", latents)
        return _FakeResult([Image.new("RGB", (WIDTH, HEIGHT))] * VIDEO_FRAMES)


class _FakeBackend:
    spec = COGVIDEOX_5B
    device = "cpu"

    def __init__(self, dtype=torch.bfloat16):
        self.pipe = _FakePipe(dtype)

    def generation_kwargs(self, **overrides):
        return {k: v for k, v in overrides.items() if v is not None}

    def encode(self, video):
        return torch.randn(FRAMES, CHANNELS, HEIGHT, WIDTH, dtype=torch.bfloat16)

    def denoiser_state(self, latents, model_output, timestep):
        z = to_canonical(latents, self.spec).float()
        v = to_canonical(model_output, self.spec).float()
        return DenoiserState(latents=z, x0=0.6 * z - 0.8 * v, eps=0.6 * v + 0.8 * z, tau=0.5)

    def renoise_scale(self, timestep):
        return torch.tensor(0.6)


def _pipeline(**kwargs) -> PhaseLockPipeline:
    return PhaseLockPipeline(_FakeBackend(), few_steps=2, full_steps=4,
                             guide_start=0, guide_end=2, **kwargs)


@pytest.mark.parametrize("prior_type", ["motion", "accel", "jerk", "perr"])
@pytest.mark.parametrize("source", ["latent", "x0_hat", "velocity"])
def test_every_setting_runs_end_to_end(prior_type, source):
    """The whole grid, in bfloat16, which is what a real sampler hands the callback."""
    pipeline = _pipeline(few_step_prior_type=prior_type, source=source)
    frames = pipeline(prompt="a ball falls", image=Image.new("RGB", (WIDTH, HEIGHT)))
    assert len(frames) == VIDEO_FRAMES
    assert pipeline.last_prior_rms == pipeline.last_prior_rms   # not NaN


def test_the_few_step_pass_runs_once_at_the_configured_length():
    pipeline = _pipeline()
    pipeline(prompt="x", image=Image.new("RGB", (WIDTH, HEIGHT)))
    assert [c["steps"] for c in pipeline.backend.pipe.calls] == [2, 4]


@pytest.mark.parametrize("source,expected", [
    ("latent", ["latents"]),
    ("x0_hat", ["latents", "noise_pred"]),
    ("velocity", ["latents", "noise_pred"]),
])
def test_noise_pred_is_requested_only_when_a_model_output_is_the_source(source, expected):
    """And requesting it must extend the pipeline's allow-list, or diffusers rejects it."""
    pipeline = _pipeline(source=source)
    pipeline(prompt="x", image=Image.new("RGB", (WIDTH, HEIGHT)))
    assert pipeline.backend.pipe.calls[-1]["tensor_inputs"] == expected
    if source != "latent":
        assert "noise_pred" in pipeline.backend.pipe._callback_tensor_inputs


def test_returning_the_few_step_result_gives_both_videos():
    pipeline = _pipeline()
    final, few = pipeline(prompt="x", image=Image.new("RGB", (WIDTH, HEIGHT)),
                          return_few_result=True)
    assert len(final) == len(few) == VIDEO_FRAMES
