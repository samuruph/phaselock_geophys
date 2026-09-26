"""Deterministic CPU contracts for exploration and same-timestep refinement."""
from contextlib import nullcontext
from types import SimpleNamespace
from dataclasses import replace
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from phaselock.config import Config, ExplorationConfig, RefinementConfig, load
from phaselock.guidance import RunningMomentumGuidance
from phaselock.backends.cogvideox import CogVideoXBackend, COGVIDEOX_5B
from phaselock.backends.base import from_canonical, to_canonical
from phaselock.sampling.exploration import StochasticExploration, rms
from phaselock.sampling.masks import exploration_mask, motion_confidence
from phaselock.sampling.refinement import refine_step
from phaselock.analysis.momentum_diagnostics import MomentumTrace, render_dashboard


def data(frames=4):
    g = torch.Generator().manual_seed(5)
    z = torch.randn(frames, 2, 3, 4, generator=g)
    return z, z * .4, torch.ones_like(z[1:])


@pytest.mark.parametrize("mode", ["uniform", "motion", "disagreement", "motion_disagreement"])
def test_exploration_bounded_reproducible_and_anchor(mode):
    z, x, v = data()
    cfg = ExplorationConfig(enabled=True, mask_mode=mode, noise_ratio=100)
    first = StochasticExploration(cfg, 1, 3)
    second = StochasticExploration(cfg, 1, 3)
    before = torch.random.get_rng_state()
    assert first.apply(z, x, v, 0, .8)[0] is z
    assert first.generator is None
    a, stats = first.apply(z, x, v, 1, .8)
    b, _ = second.apply(z, x, v, 1, .8)
    assert torch.equal(a, b)
    assert torch.equal(a[0], z[0])
    assert rms((a-z)[1:]) <= .050001 * rms(z[1:])
    assert torch.equal(before, torch.random.get_rng_state())
    assert stats["noise_anchor_norm"] == 0
    assert not torch.equal(a, StochasticExploration(replace(cfg, seed=1), 1, 3).apply(z, x, v, 1, .8)[0])


def test_zero_strength_and_short_video():
    z, x, v = data(2)
    controller = StochasticExploration(ExplorationConfig(enabled=True, noise_ratio=0), 0, 3)
    assert controller.apply(z, x, v, 1, .8)[0] is z
    assert controller.generator is None
    controller = StochasticExploration(ExplorationConfig(enabled=True), 0, 3)
    guided, _ = controller.apply(z, x, v, 1, .8)
    assert torch.isfinite(guided).all()
    mask, motion, disagreement = exploration_mask(torch.zeros_like(x), torch.zeros_like(v))
    assert motion.count_nonzero() == disagreement.count_nonzero() == 0
    assert mask[0].count_nonzero() == 0
    assert torch.all(mask[1:] == .05)


def test_unstructured_control_has_matched_masked_rms():
    z, x, v = data()
    cfg = ExplorationConfig(enabled=True)
    _, a = StochasticExploration(cfg, 0, 3).apply(z, x, v, 1, .8)
    _, b = StochasticExploration(replace(cfg, structured=False), 0, 3).apply(z, x, v, 1, .8)
    assert a["noise_proposed_rms"] == pytest.approx(b["noise_proposed_rms"], rel=1e-5)


def test_config_extension_validation_and_overrides():
    with pytest.raises(ValueError):
        Config(exploration=ExplorationConfig(enabled=True), refinement=RefinementConfig(enabled=True))
    with pytest.raises(ValueError):
        ExplorationConfig(noise_ratio=float("nan"))
    cfg = load("configs/experiments/physics_iq_running_momentum.yaml", exploration__enabled="true")
    assert cfg.exploration.enabled and not cfg.refinement.enabled


class Progress:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self): pass


class TinyTransformer(torch.nn.Module):
    config = SimpleNamespace(sample_height=3, sample_width=4, sample_frames=4,
                             patch_size_t=None, in_channels=4,
                             use_rotary_positional_embeddings=False, ofs_embed_dim=None)

    def __init__(self):
        super().__init__()
        self.calls = []

    def cache_context(self, *args): return nullcontext()

    def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
        self.calls.append((hidden_states.detach().clone(), timestep.detach().clone()))
        return (hidden_states[:, :, :2] * .17 + encoder_hidden_states[:, :1, :1, None, None] * .01,)


class TinyPipe:
    vae_scale_factor_spatial = 1
    vae_scale_factor_temporal = 1
    _execution_device = torch.device("cpu")
    _interrupt = False
    _guidance_scale = 2.

    def __init__(self):
        from diffusers import CogVideoXDDIMScheduler
        self.scheduler = CogVideoXDDIMScheduler(num_train_timesteps=100, prediction_type="v_prediction")
        self.transformer = TinyTransformer()
        self.video_processor = SimpleNamespace(preprocess=lambda image, **kw: image)
        self.initials = []

    @property
    def interrupt(self): return self._interrupt
    @property
    def guidance_scale(self): return self._guidance_scale
    def check_inputs(self, **kwargs): pass
    def progress_bar(self, **kwargs): return Progress()
    def maybe_free_model_hooks(self): pass
    def prepare_extra_step_kwargs(self, generator, eta): return {"eta": eta, "generator": generator}
    def encode_prompt(self, **kwargs): return torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
    def prepare_latents(self, image, batch, channels, frames, height, width, dtype, device, generator, latents):
        z = torch.randn(batch, frames, channels, height, width, generator=generator)
        self.initials.append(z.clone())
        return z, torch.zeros_like(z)


def setup_controller(source="latent", recorder=None, strength=.05):
    pipe = TinyPipe()
    backend = CogVideoXBackend(pipe, device="cpu", mode="i2v")
    guide = RunningMomentumGuidance(backend.spec, backend=backend, source=source,
                                   recorder=recorder, guide_start=0, guide_end=2,
                                   guidance_strength=strength)
    return pipe, backend, guide


def run_tiny(k=1, recorder=None, source="latent", seed=0, strength=.05):
    from phaselock.sampling.cogvideox_pnp import run_refinement
    pipe, backend, guide = setup_controller(source, recorder, strength)
    original_scheduler = pipe.scheduler
    result, metrics = run_refinement(backend, guide, RefinementConfig(enabled=True, steps_per_timestep=k, seed=seed),
        image=torch.zeros(1, 3, 3, 4), prompt="ball", num_frames=4, height=3, width=4,
        num_inference_steps=4, guidance_scale=2., output_type="latent",
        generator=torch.Generator().manual_seed(42))
    assert pipe.scheduler is original_scheduler
    return result.frames, metrics, guide, pipe


def test_k_zero_matches_native_ddim_exactly():
    from diffusers import CogVideoXImageToVideoPipeline
    actual, metrics, guide, _ = run_tiny(k=0, strength=0)
    pipe = TinyPipe()
    expected = CogVideoXImageToVideoPipeline.__call__(pipe,
        image=torch.zeros(1, 3, 3, 4), prompt="ball", num_frames=4, height=3, width=4,
        num_inference_steps=4, guidance_scale=2., output_type="latent",
        generator=torch.Generator().manual_seed(42)).frames
    assert torch.equal(actual, expected)
    assert metrics["guided_predictions"] == guide.step == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pnp_renoises_full_clean_estimate_and_accepts_final_ddim_step(dtype):
    pipe, backend, _ = setup_controller()
    pipe.scheduler.set_timesteps(4)
    timestep = pipe.scheduler.timesteps[0]
    original = torch.randn(1, 4, 2, 3, 4,
                           generator=torch.Generator().manual_seed(11), dtype=dtype)
    inputs = []

    def predict(candidate, t):
        assert torch.equal(t, timestep)
        inputs.append(candidate.clone())
        return candidate.float() * .17 + .02

    first_prediction = predict(original, timestep)
    ordinary_next, clean = pipe.scheduler.step(first_prediction, timestep, original, return_dict=False)
    inputs.clear()
    seed = 19
    expected_noise = torch.randn(to_canonical(clean, backend.spec).shape,
                                 generator=torch.Generator().manual_seed(seed), dtype=dtype)
    expected_candidate = from_canonical(
        backend.renoise(to_canonical(clean, backend.spec), expected_noise, timestep), backend.spec
    ).to(dtype)
    expected_prediction = expected_candidate.float() * .17 + .02
    expected_next, _ = pipe.scheduler.step(
        expected_prediction, timestep, expected_candidate, return_dict=False
    )
    expected_state = backend.denoiser_state(expected_candidate, expected_prediction, timestep)
    expected_next = expected_next.to(dtype)
    ordinary_next = ordinary_next.to(dtype)

    accepted, state, _, extras = refine_step(
        backend, pipe.scheduler, original, timestep, predict, 1,
        torch.Generator().manual_seed(seed), record=True,
    )
    assert len(inputs) == 2
    assert torch.equal(inputs[0], original)
    assert torch.allclose(inputs[1], expected_candidate)
    assert not torch.equal(inputs[1][:, 0], original[:, 0])
    assert torch.allclose(accepted, expected_next)
    assert torch.equal(state.x0, expected_state.x0)
    assert torch.equal(extras["original_clean"],
                       backend.denoiser_state(original, first_prediction, timestep).x0)
    assert torch.allclose(extras["refinement_update"],
                          to_canonical(expected_next.float() - ordinary_next.float(), backend.spec))


@pytest.mark.parametrize("source", ["latent", "x0_hat", "blend"])
def test_pnp_predictions_timing_rng_and_diagnostics(source, tmp_path):
    trace = MomentumTrace("pnp", record_steps={0, 1})
    actual, metrics, guide, pipe = run_tiny(k=2, source=source, recorder=trace)
    without, _, _, _ = run_tiny(k=2, source=source)
    assert torch.equal(actual, without)
    assert metrics["guided_predictions"] == 8
    assert guide.step == 4
    times = [int(t[0]) for _, t in pipe.transformer.calls]
    assert times[:3] == [times[0]] * 3 and times[3:6] == [times[3]] * 3
    other, _, _, other_pipe = run_tiny(k=2, source=source, seed=1)
    assert torch.equal(pipe.initials[0], other_pipe.initials[0])
    assert not torch.equal(actual, other)
    assert trace.steps[0].extras["guided_predictions"] == 3
    assert "inner_clean_0" not in trace.steps[0].extras
    def decode(x):
        rgb = torch.cat((x, x[:, :1]), dim=1).sigmoid()
        return rgb.repeat_interleave(2, -1).repeat_interleave(2, -2)
    render_dashboard(trace, tmp_path / "dashboard.mp4", decode=decode, preview_height=32, temporal_ratio=1)
    assert (tmp_path / "extensions.mp4").stat().st_size > 0


def test_scheduler_restored_after_prediction_failure():
    from phaselock.sampling.cogvideox_pnp import run_refinement
    pipe, backend, guide = setup_controller()
    scheduler = pipe.scheduler
    def fail(**kwargs): raise RuntimeError("injected")
    pipe.transformer.forward = fail
    with pytest.raises(RuntimeError, match="injected"):
        run_refinement(backend, guide, RefinementConfig(enabled=True),
            image=torch.zeros(1, 3, 3, 4), prompt="ball", num_frames=4,
            num_inference_steps=4, output_type="latent")
    assert pipe.scheduler is scheduler


def test_gate_changes_correction_not_moments():
    z, x, _ = data()
    _, backend, first = setup_controller()
    _, _, second = setup_controller()
    latents = from_canonical(z, backend.spec)
    for guide in (first, second):
        guide.apply_step(None, 0, 99, latents)
    changed = latents * .7
    ungated = first.apply_step(None, 1, 74, changed)
    gated = second.apply_step(None, 1, 74, changed, confidence=torch.zeros_like(z[1:, 0]))
    assert torch.equal(first.mean, second.mean) and torch.equal(first.variance, second.variance)
    assert torch.equal(gated, changed) and not torch.equal(ungated, changed)
    u, c = motion_confidence(x, x)
    assert torch.all(u == 0) and torch.all(c == 1)


def test_report_pairs_samples_and_reports_auxiliary_diversity(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/report_sampling_extensions.py"
    runs = []
    for seed in (0, 1):
        run = tmp_path / f"run_{seed}"
        runs.append(run)
        videos = run / "videos/motion_on_latent"
        metadata = run / "sampling/motion_on_latent"
        videos.mkdir(parents=True)
        metadata.mkdir(parents=True)
        rows = []
        for sample in ("0001", "0002"):
            name = f"{sample}_clip.mp4"
            path = videos / name
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (16, 16))
            for frame in range(3):
                writer.write(np.full((16, 16, 3), 40 + seed * 20 + frame, np.uint8))
            writer.release()
            rows.append({"sample_id": sample, "setting": "motion_on_latent", "output_name": name,
                         "motion": ".1", "spatial_iou": ".8", "spatiotemporal_iou": ".8",
                         "weighted_spatial_iou": ".8", "mse": ".1", "raw_score": ".8",
                         "variance_spatial_iou": ".8", "variance_spatiotemporal_iou": ".8",
                         "variance_weighted_spatial_iou": ".8", "variance_mse": ".1"})
            (metadata / f"{sample}_clip.json").write_text(json.dumps({
                "exploration": {"enabled": True, "seed": seed}, "refinement": {"enabled": False},
                "generation_seed": 42, "guided_predictions": 50, "transformer_calls": 50,
                "wall_seconds": 2.0,
            }))
        with (run / "physics_iq_motion_on_latent.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    output = tmp_path / "comparison.csv"
    subprocess.run([sys.executable, str(script), *(str(run) for run in runs),
                    "--out", str(output)], check=True, capture_output=True, text=True)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(int(r["samples"]) == 2 for r in rows)
    assert all(float(r["mean_pairwise_diversity"]) > 0 for r in rows)
    assert all(float(r["mean_transformer_calls"]) == 50 for r in rows)
