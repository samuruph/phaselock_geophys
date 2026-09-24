"""CPU checks for the focused running-moment dashboard."""
from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import pytest
import torch

from phaselock.analysis.momentum_diagnostics import (
    DiagnosticStep, MomentumTrace, StepPredictionCapture, render_dashboard,
    _video_to_transition,
)


def field(seed: int, frames: int = 4, channels: int = 3) -> torch.Tensor:
    return torch.randn(frames - 1, channels, 2, 2,
                       generator=torch.Generator().manual_seed(seed))


def add(trace: MomentumTrace, step: int, seed: int, strength: float = .05) -> None:
    current = field(seed)
    trace.capture(step=step, timestep=900 - step * 100, tau=step / 10,
                  current=current, m1=current * .8, m2=current.square(),
                  mean=current * .7, variance=current.square(), correction=current * .1,
                  strength=strength, x0=torch.randn(4, 3, 2, 2))


def test_trace_summary_contains_moments_guidance_and_temporal_profiles():
    trace = MomentumTrace("motion", record_steps={0, 2})
    add(trace, 0, 1, strength=0)
    add(trace, 1, 2)
    add(trace, 2, 3)
    rows = trace.summaries()
    assert [row["step"] for row in rows] == [0, 2]
    assert rows[0]["applied_guidance_rms"] == 0
    assert len(rows[0]["m1_by_latent_transition"]) == 3
    assert len(rows[0]["m2_mean_by_latent_transition"]) == 3


def test_decoded_frames_align_to_latent_transitions():
    assert _video_to_transition(0, 4, 12) is None
    assert _video_to_transition(1, 4, 12) == 0
    assert _video_to_transition(4, 4, 12) == 0
    assert _video_to_transition(5, 4, 12) == 1
    assert _video_to_transition(48, 4, 12) == 11


def test_dashboard_and_each_selected_step_are_playable(tmp_path):
    trace = MomentumTrace("motion")
    add(trace, 0, 1)
    add(trace, 2, 2)

    def decode(latents):
        return torch.sigmoid(latents[:, :3]).mean(dim=(-2, -1), keepdim=True).expand(-1, 3, 8, 8)

    output = render_dashboard(trace, tmp_path / "dashboard.mp4", decode=decode,
                              fps=4, preview_height=32, temporal_ratio=1)
    payload = json.loads(output.with_suffix(".json").read_text())
    assert len(payload["steps"]) == 2
    assert payload["step_videos"] == ["steps/step_000.mp4", "steps/step_002.mp4"]
    assert payload["alignment"]["latent_to_video_temporal_ratio"] == 1
    combined = cv2.VideoCapture(str(output))
    assert int(combined.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    combined.release()
    for name in ("step_000.mp4", "step_002.mp4"):
        movie = cv2.VideoCapture(str(tmp_path / "steps" / name))
        assert int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        assert int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)) == 128
        movie.release()
    assert sorted(p.name for p in (tmp_path / "steps").iterdir()) == [
        "step_000.mp4", "step_002.mp4"]


def test_flow_capture_hook_captures_step_input_and_clears():
    class Transformer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2

    class Backend:
        def __init__(self):
            self.pipe = SimpleNamespace(transformer=Transformer())

        def combine_cfg(self, calls, guidance_scale):
            assert len(calls) == 1
            return calls[0]

        def denoiser_state(self, latents, output, timestep):
            return SimpleNamespace(x0=latents, drift=output, tau=.1)

    backend = Backend()
    first = torch.ones(1, 3, 2, 2)
    with StepPredictionCapture(backend, 6.0) as capture:
        backend.pipe.transformer(hidden_states=first)
        state = capture.consume(torch.tensor(900))
        assert torch.equal(state.x0, first)
        assert capture.consume(torch.tensor(800)) is None
    assert len(backend.pipe.transformer._forward_hooks) == 0


def test_m2_and_guidance_summary_stay_finite_for_zero_moments():
    zeros = torch.zeros(2, 3, 2, 2)
    step = DiagnosticStep(step=0, timestep=900, tau=0, current=zeros,
                          m1=zeros, m2=zeros, correction=zeros, strength=0)
    summary = step.summary()
    assert summary["sqrt_m2_rms"] == pytest.approx(0)
    assert summary["applied_guidance_rms"] == pytest.approx(0)
