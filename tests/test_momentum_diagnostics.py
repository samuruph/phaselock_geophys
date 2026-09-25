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
from phaselock.backends.base import from_canonical, to_canonical
from phaselock.backends.cogvideox import COGVIDEOX_5B
from phaselock.backends.wan import WAN21_T2V_1_3B
from phaselock.guidance import RunningMomentumGuidance


def field(seed: int, frames: int = 4, channels: int = 3) -> torch.Tensor:
    return torch.randn(frames - 1, channels, 2, 2,
                       generator=torch.Generator().manual_seed(seed))


def add(trace: MomentumTrace, step: int, seed: int, strength: float = .05) -> None:
    current = field(seed)
    trace.capture(step=step, timestep=900 - step * 100, tau=step / 10,
                  current=current, m1=current * .8, m2=current.square(),
                  mean=current * .7, variance=current.square() * 2, correction=current * .1,
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
    assert rows[0]["m1_rms"] == pytest.approx(float((field(1) * .7).square().mean().sqrt()))
    assert rows[0]["raw_m1_rms"] != pytest.approx(rows[0]["m1_rms"])
    assert rows[0]["m2_mean"] == pytest.approx(rows[0]["raw_m2_mean"] * 2)


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
        assert int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)) == 176
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
    assert summary["sqrt_bias_corrected_variance_rms"] == pytest.approx(0)
    assert summary["applied_guidance_rms"] == pytest.approx(0)


@pytest.mark.parametrize("spec", [COGVIDEOX_5B, WAN21_T2V_1_3B])
@pytest.mark.parametrize("source", ["latent", "x0_hat"])
def test_running_momentum_uses_post_step_latent_or_current_step_x0_hat(spec, source):
    # Deliberately make the callback's post-step latents and x0 motion different.
    pre = torch.zeros(3, spec.channels, 2, 2)
    pre[1:] = 1
    clean = torch.zeros_like(pre)
    clean[1] = 3
    clean[2] = 8
    post = torch.full_like(pre, 17)
    state = SimpleNamespace(latents=pre, x0=clean, drift=torch.zeros_like(pre), tau=.25)

    class Capture:
        def consume(self, timestep):
            assert float(timestep) == 750
            return state

    trace = MomentumTrace("motion")
    controller = RunningMomentumGuidance(
        spec, source=source, prediction_capture=Capture(), recorder=trace,
        guidance_strength=.05, guide_end=2,
    )
    out = controller(None, 0, torch.tensor(750),
                     {"latents": from_canonical(post, spec)})
    selected = post if source == "latent" else clean
    expected = selected[1:] - selected[:-1]
    assert torch.equal(trace.steps[0].current, expected)
    assert torch.allclose(controller.m1, expected * .1)
    assert torch.count_nonzero(controller.m2) == 0  # first residual is zero
    assert torch.allclose(to_canonical(out["latents"], spec)[1:],
                          post[1:] + .05 * trace.steps[0].correction)
    assert torch.equal(to_canonical(out["latents"], spec)[0], post[0])
    assert trace.steps[0].timestep == 750


def test_post_step_latent_momentum_does_not_require_prediction_capture():
    controller = RunningMomentumGuidance(COGVIDEOX_5B, guide_end=2)
    post = from_canonical(torch.zeros(3, COGVIDEOX_5B.channels, 2, 2), COGVIDEOX_5B)
    controller(None, 0, torch.tensor(750), {"latents": post})


@pytest.mark.parametrize("velocity_decay", [0.0, 0.01, 0.2])
def test_constant_motion_is_a_fixed_point_with_velocity_decay(velocity_decay):
    controller = RunningMomentumGuidance(
        COGVIDEOX_5B, velocity_decay=velocity_decay, guide_end=20,
    )
    motion = torch.full((2, 3, 2, 2), 2.0)
    for _ in range(12):
        correction = controller._update_momentum(motion)
        assert torch.allclose(controller.mean, motion, atol=1e-6)
        assert controller.variance.abs().max() < 1e-10
        assert correction.abs().max() < 1e-5


def test_second_moment_tracks_the_applied_residual_and_not_raw_first_moment():
    controller = RunningMomentumGuidance(
        COGVIDEOX_5B, betas=(0.5, 0.5), velocity_decay=0.2, guide_end=3,
    )
    first = torch.ones(2, 3, 2, 2)
    second = torch.full_like(first, 3.0)
    controller._update_momentum(first)
    controller._update_momentum(second)
    residual = controller.mean - second
    expected_variance = (0.5 * residual.square()) / (1 - 0.5**2)
    assert torch.allclose(controller.variance, expected_variance)


@pytest.mark.parametrize("change", [1e-5, 10.0])
def test_guidance_update_is_finite_and_bounded(change):
    spec = COGVIDEOX_5B
    controller = RunningMomentumGuidance(
        spec, guidance_strength=1.0, guide_end=3, max_update_ratio=0.05,
    )
    z = torch.zeros(3, spec.channels, 2, 2)
    z[1:] = 1.0
    controller(None, 0, torch.tensor(750), {"latents": from_canonical(z, spec)})
    changed = z.clone()
    changed[2] += change
    out = controller(None, 1, torch.tensor(500),
                     {"latents": from_canonical(changed, spec)})
    guided = to_canonical(out["latents"], spec)
    update = guided[1:] - changed[1:]
    assert torch.isfinite(guided).all()
    assert torch.equal(guided[0], changed[0])
    assert update.square().mean().sqrt() <= 0.05 * changed.square().mean().sqrt() + 1e-7
    if change == 10.0:
        assert update.square().mean().sqrt() == pytest.approx(
            0.05 * changed.square().mean().sqrt(), rel=1e-5,
        )


def test_x0_hat_momentum_requires_captured_prediction():
    controller = RunningMomentumGuidance(COGVIDEOX_5B, guide_end=2, source="x0_hat",
                                         prediction_capture=SimpleNamespace(consume=lambda _: None))
    post = from_canonical(torch.zeros(3, COGVIDEOX_5B.channels, 2, 2), COGVIDEOX_5B)
    with pytest.raises(RuntimeError, match="prediction captured"):
        controller(None, 0, torch.tensor(750), {"latents": post})


@pytest.mark.parametrize("spec", [COGVIDEOX_5B, WAN21_T2V_1_3B])
@pytest.mark.parametrize("step,weight", [(0, 0.0), (1, 0.15625), (2, 0.5), (4, 1.0)])
def test_blended_running_momentum_interpolates_motion_before_moments(spec, step, weight):
    clean = torch.zeros(3, spec.channels, 2, 2)
    clean[1], clean[2] = 3, 8
    post = torch.zeros_like(clean)
    post[1], post[2] = 2, 6
    state = SimpleNamespace(x0=clean, drift=torch.zeros_like(clean), tau=.25)
    capture = SimpleNamespace(consume=lambda _: state)
    trace = MomentumTrace("blend")
    controller = RunningMomentumGuidance(
        spec, source="blend", prediction_capture=capture, recorder=trace,
        guidance_strength=.05, guide_end=5,
    )
    out = controller(None, step, torch.tensor(750), {"latents": from_canonical(post, spec)})
    clean_motion = clean[1:] - clean[:-1]
    latent_motion = post[1:] - post[:-1]
    expected = (1 - weight) * clean_motion + weight * latent_motion
    assert torch.allclose(trace.steps[0].current, expected)
    assert torch.allclose(controller.m1, expected * .1)
    assert torch.count_nonzero(controller.m2) == 0  # first residual is zero
    assert trace.summaries()[0]["latent_weight"] == pytest.approx(weight)
    strength = controller.compute_schedule(step)
    assert torch.allclose(to_canonical(out["latents"], spec)[1:],
                          post[1:] + strength * trace.steps[0].correction)
    assert torch.equal(to_canonical(out["latents"], spec)[0], post[0])


def test_blended_running_momentum_one_step_window_uses_x0_hat():
    clean = torch.zeros(3, COGVIDEOX_5B.channels, 2, 2)
    clean[1:] = 3
    post = torch.zeros_like(clean)
    trace = MomentumTrace("blend")
    capture = SimpleNamespace(consume=lambda _: SimpleNamespace(
        x0=clean, drift=torch.zeros_like(clean), tau=.5,
    ))
    controller = RunningMomentumGuidance(
        COGVIDEOX_5B, source="blend", guide_end=4, guide_start=3,
        prediction_capture=capture, recorder=trace,
    )
    controller(None, 3, torch.tensor(500), {"latents": from_canonical(post, COGVIDEOX_5B)})
    assert controller.blend_latent_weight(3) == 0.0
    assert trace.summaries()[0]["latent_weight"] == 0.0
    assert torch.equal(trace.steps[0].current, clean[1:] - clean[:-1])


@pytest.mark.parametrize("source", ["x0_hat", "blend"])
def test_prediction_based_momentum_requires_capture(source):
    controller = RunningMomentumGuidance(COGVIDEOX_5B, guide_end=2, source=source,
                                         prediction_capture=SimpleNamespace(consume=lambda _: None))
    post = from_canonical(torch.zeros(3, COGVIDEOX_5B.channels, 2, 2), COGVIDEOX_5B)
    with pytest.raises(RuntimeError, match="prediction captured"):
        controller(None, 0, torch.tensor(750), {"latents": post})
