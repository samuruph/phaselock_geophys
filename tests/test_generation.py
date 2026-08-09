"""Tests for CFG reassembly and the Stage 6/7 drivers. CPU only.

The riskiest piece here is :meth:`VideoBackend.combine_cfg`. A forward hook on the
transformer sees the raw calls, not the guided combination that actually drove the
trajectory, and the two backends disagree about how those calls are issued. Getting the
chunk order or the call order backwards would silently record the *unconditional*
prediction, which looks entirely plausible and would quietly poison every generation
measurement.
"""

from __future__ import annotations

import pytest
import torch

from phaselock.backends import COGVIDEOX_5B, WAN21_T2V_1_3B, from_canonical
from phaselock.backends.base import CFG_BATCHED, CFG_SEQUENTIAL
from phaselock.backends.cogvideox import CogVideoXBackend
from phaselock.backends.wan import WanBackend
from phaselock.experiments.generation import Candidate, best_of_n, frames_to_tensor
from phaselock.experiments.step_sweep import (
    DEFAULT_BLUR,
    DEFAULT_STEPS,
    SweepCell,
    blur_survival,
    ordering_by_steps,
)
from phaselock.metrics.geophys import STATISTICS
from phaselock.metrics.motion_mask import MotionMaskScores

FRAMES, CHANNELS, HEIGHT, WIDTH = 5, 16, 4, 4


def canonical(value: float) -> torch.Tensor:
    return torch.full((FRAMES, CHANNELS, HEIGHT, WIDTH), value)


# -- CFG declarations -------------------------------------------------------


def test_backends_declare_the_right_cfg_style():
    """CogVideoX doubles the batch; Wan calls the transformer twice."""
    assert CogVideoXBackend.cfg_style.fget(object.__new__(CogVideoXBackend)) == CFG_BATCHED
    assert WanBackend.cfg_style.fget(object.__new__(WanBackend)) == CFG_SEQUENTIAL


# -- batched CFG (CogVideoX) ------------------------------------------------


def _cog() -> CogVideoXBackend:
    backend = object.__new__(CogVideoXBackend)
    backend.spec = COGVIDEOX_5B
    return backend


def test_batched_cfg_uses_the_uncond_cond_chunk_order():
    """prompt_embeds is cat([negative, positive]), so chunk(2) is (uncond, cond)."""
    backend = _cog()
    uncond, cond = canonical(1.0), canonical(3.0)
    outputs = torch.cat([from_canonical(uncond, COGVIDEOX_5B), from_canonical(cond, COGVIDEOX_5B)])
    inputs = torch.cat([from_canonical(canonical(0.0), COGVIDEOX_5B)] * 2)

    _, guided = backend.combine_cfg([(inputs, outputs)], guidance_scale=6.0)
    # uncond + 6 * (cond - uncond) = 1 + 6*2 = 13
    assert torch.allclose(guided, canonical(13.0))


def test_batched_cfg_at_scale_one_returns_the_conditional():
    backend = _cog()
    outputs = torch.cat(
        [from_canonical(canonical(1.0), COGVIDEOX_5B), from_canonical(canonical(3.0), COGVIDEOX_5B)]
    )
    inputs = torch.cat([from_canonical(canonical(0.0), COGVIDEOX_5B)] * 2)
    _, guided = backend.combine_cfg([(inputs, outputs)], guidance_scale=1.0)
    assert torch.allclose(guided, canonical(3.0))


def test_batched_cfg_without_guidance_passes_a_single_prediction_through():
    backend = _cog()
    inputs = from_canonical(canonical(0.0), COGVIDEOX_5B)
    outputs = from_canonical(canonical(2.0), COGVIDEOX_5B)
    _, guided = backend.combine_cfg([(inputs, outputs)], guidance_scale=6.0)
    assert torch.allclose(guided, canonical(2.0))


def test_image_conditioning_channels_are_stripped_from_the_recorded_latents():
    """CogVideoX I2V concatenates image latents on the channel axis before the forward."""
    backend = _cog()
    video = from_canonical(canonical(0.5), COGVIDEOX_5B)
    image = from_canonical(canonical(9.0), COGVIDEOX_5B)
    inputs = torch.cat([video, image], dim=2)  # channel axis under BTCHW
    outputs = from_canonical(canonical(1.0), COGVIDEOX_5B)

    latents, _ = backend.combine_cfg([(inputs, outputs)], guidance_scale=1.0)
    assert latents.shape == (FRAMES, CHANNELS, HEIGHT, WIDTH)
    assert torch.allclose(latents, canonical(0.5))


def test_batched_backend_rejects_more_than_one_call():
    with pytest.raises(ValueError, match="one batched call"):
        _cog().combine_cfg([(torch.zeros(1), torch.zeros(1))] * 2, guidance_scale=1.0)


# -- sequential CFG (Wan) ---------------------------------------------------


def _wan() -> WanBackend:
    backend = object.__new__(WanBackend)
    backend.spec = WAN21_T2V_1_3B
    return backend


def test_sequential_cfg_treats_the_first_call_as_conditional():
    """Wan computes noise_pred (cond) then noise_uncond, in that order."""
    backend = _wan()
    cond = (from_canonical(canonical(0.0), WAN21_T2V_1_3B), from_canonical(canonical(3.0), WAN21_T2V_1_3B))
    uncond = (from_canonical(canonical(0.0), WAN21_T2V_1_3B), from_canonical(canonical(1.0), WAN21_T2V_1_3B))

    _, guided = backend.combine_cfg([cond, uncond], guidance_scale=6.0)
    assert torch.allclose(guided, canonical(13.0))


def test_sequential_cfg_with_one_call_is_ungiuded():
    backend = _wan()
    call = (from_canonical(canonical(0.0), WAN21_T2V_1_3B), from_canonical(canonical(2.0), WAN21_T2V_1_3B))
    _, guided = backend.combine_cfg([call], guidance_scale=6.0)
    assert torch.allclose(guided, canonical(2.0))


def test_sequential_backend_rejects_three_calls():
    call = (from_canonical(canonical(0.0), WAN21_T2V_1_3B), from_canonical(canonical(1.0), WAN21_T2V_1_3B))
    with pytest.raises(ValueError, match="one or two calls"):
        _wan().combine_cfg([call] * 3, guidance_scale=1.0)


def test_both_styles_agree_given_the_same_predictions():
    """Layout and call convention must not change the guided result."""
    cond, uncond, scale = canonical(3.0), canonical(1.0), 4.0

    _, batched = _cog().combine_cfg(
        [(
            torch.cat([from_canonical(canonical(0.0), COGVIDEOX_5B)] * 2),
            torch.cat([from_canonical(uncond, COGVIDEOX_5B), from_canonical(cond, COGVIDEOX_5B)]),
        )],
        guidance_scale=scale,
    )
    _, sequential = _wan().combine_cfg(
        [
            (from_canonical(canonical(0.0), WAN21_T2V_1_3B), from_canonical(cond, WAN21_T2V_1_3B)),
            (from_canonical(canonical(0.0), WAN21_T2V_1_3B), from_canonical(uncond, WAN21_T2V_1_3B)),
        ],
        guidance_scale=scale,
    )
    assert torch.allclose(batched, sequential)


def test_no_captures_raises():
    with pytest.raises(ValueError, match="no transformer calls"):
        _cog().combine_cfg([], guidance_scale=1.0)


# -- best-of-N --------------------------------------------------------------


def make_candidate(seed: int, raw: float) -> Candidate:
    return Candidate(
        sample_id="ball_drop/0/valid",
        scenario="ball_drop",
        seed=seed,
        num_steps=50,
        blur_sigma=0.0,
        scores=MotionMaskScores(raw, raw, raw, 0.0),
    )


def test_best_of_n_picks_the_lowest_verifier_score():
    """Every GeoPhys statistic is oriented so that lower means more plausible."""
    candidates = [make_candidate(index, 0.1 * index) for index in range(4)]
    selected, baseline, oracle = best_of_n(candidates, scores=[3.0, 1.0, 2.0, 4.0])

    assert selected.seed == 1
    assert baseline.seed == 0            # no-verifier control is the first draw
    assert oracle.seed == 3              # highest fidelity available


def test_best_of_n_validates_its_inputs():
    with pytest.raises(ValueError, match="no candidates"):
        best_of_n([], [])
    with pytest.raises(ValueError, match="but 1 scores"):
        best_of_n([make_candidate(0, 1.0), make_candidate(1, 2.0)], [1.0])


def test_frames_to_tensor_handles_pipeline_output_shapes():
    import numpy as np

    array = np.random.rand(4, 8, 8, 3).astype("float32")
    assert frames_to_tensor(array).shape == (4, 3, 8, 8)
    tensor = torch.rand(4, 3, 8, 8)
    assert frames_to_tensor(tensor) is tensor


# -- step sweep -------------------------------------------------------------


def make_cell(steps: int, sigma: float, curv: float, phase: float = 0.5) -> SweepCell:
    return SweepCell(
        sample_id="ball_drop/0/valid",
        scenario="ball_drop",
        seed=0,
        num_steps=steps,
        blur_sigma=sigma,
        geophys={name: curv for name in STATISTICS},
        phase_difference=phase,
        phase_coherence=0.7,
        magnitude_correlation=0.9,
        motion_mask=MotionMaskScores(0.2, 0.2, 0.1, 0.01),
    )


def test_sweep_row_schema_is_complete():
    row = make_cell(2, 0.0, 0.3).flatten()
    assert set(SweepCell.fieldnames()) == set(row)


def test_ordering_by_steps_averages_within_a_blur_level():
    cells = [make_cell(2, 0.0, 0.3), make_cell(2, 0.0, 0.5), make_cell(50, 0.0, 0.9)]
    means = ordering_by_steps(cells, "curv", blur_sigma=0.0)
    assert means == {2: pytest.approx(0.4), 50: pytest.approx(0.9)}


def test_ordering_by_steps_ignores_other_blur_levels():
    cells = [make_cell(2, 0.0, 0.3), make_cell(2, 16.0, 9.0)]
    assert ordering_by_steps(cells, "curv", blur_sigma=0.0) == {2: pytest.approx(0.3)}


def test_blur_survival_reports_a_surviving_advantage():
    """The interesting outcome: the few-step arm stays more regular at every sigma."""
    cells = []
    for sigma in DEFAULT_BLUR:
        cells.append(make_cell(2, sigma, 0.3))
        cells.append(make_cell(50, sigma, 0.9))

    survival = blur_survival(cells, "curv")
    assert set(survival) == set(DEFAULT_BLUR)
    assert all(row["few_step_favoured"] == 1.0 for row in survival.values())
    assert all(row["gap"] > 0 for row in survival.values())


def test_blur_survival_detects_an_advantage_that_is_only_sharpness():
    """The negative outcome the control exists to catch.

    If the K=2 advantage disappears once texture is removed, the ordering was measuring
    blur, not physics -- worth reporting, since PhaseLock's spectral metric survives the
    same control.
    """
    cells = [
        make_cell(2, 0.0, 0.3), make_cell(50, 0.0, 0.9),    # advantage at sigma=0
        make_cell(2, 16.0, 0.9), make_cell(50, 16.0, 0.3),  # reversed once blurred
    ]
    survival = blur_survival(cells, "curv")
    assert survival[0.0]["few_step_favoured"] == 1.0
    assert survival[16.0]["few_step_favoured"] == 0.0


def test_blur_survival_handles_a_metric_where_higher_is_better():
    """Phase-difference correlation is oriented the other way from the statistics."""
    cells = [
        make_cell(2, 0.0, 0.3, phase=0.358),
        make_cell(50, 0.0, 0.9, phase=0.100),
    ]
    survival = blur_survival(
        cells, "phase_difference_corr", lower_is_more_regular=False
    )
    assert survival[0.0]["few_step_favoured"] == 1.0
    assert survival[0.0]["gap"] == pytest.approx(0.258)


def test_blur_survival_skips_incomplete_sweeps():
    assert blur_survival([make_cell(2, 0.0, 0.3)], "curv") == {}


def test_default_sweep_matches_the_papers_settings():
    assert DEFAULT_STEPS == (2, 10, 30, 50)
    assert DEFAULT_BLUR == (0.0, 8.0, 16.0)


def test_statistics_from_record_works_without_a_pair():
    """Generation has no matched pair, and must still produce the inversion schema."""
    from phaselock.datasets import VideoSample
    from phaselock.experiments.detection import statistics_from_record
    from phaselock.probes import LATENT, ProbeRecord, TrajectoryKey

    record = ProbeRecord(
        trajectories={TrajectoryKey(LATENT, -1, 0): torch.randn(8, 6)},
        taus={0: 0.5},
        provenance={},
    )
    sample = VideoSample(sample_id="ball_drop/000/valid", path="x.mp4", label=0,
                         group="g", dataset="likephys", meta={"scenario": "ball_drop"})

    rows = statistics_from_record(record, sample)
    assert rows and rows[0].scenario == "ball_drop"
    assert rows[0].violation == "", "generated clips have no violation"
    assert set(rows[0].statistics) == {"speed", "curv", "ang", "accel", "perr"}


def test_spearman_endpoints_and_ties():
    from phaselock.experiments.generation import spearman

    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # A constant series has no ranking to correlate with.
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) != spearman([1, 1, 1, 1], [1, 2, 3, 4])


def test_score_against_fidelity_finds_the_predictive_signal():
    """A signal that tracks fidelity must rank ahead of one that does not.

    Negative rho is the useful direction: every statistic is oriented so larger means
    less regular, so a good detector gives *worse* fidelity as it grows.
    """
    from phaselock.experiments.generation import score_against_fidelity

    fidelity = {f"c{i}": 1.0 - 0.1 * i for i in range(6)}
    statistics = []
    for sample, score in fidelity.items():
        statistics.append({
            "sample_id": sample, "source": "hidden_states", "block": "3", "step": "0",
            "phi_accel": str(1.0 - score),      # tracks fidelity inversely -> rho ~ -1
            "phi_curv": str(hash(sample) % 7),  # unrelated
        })
    candidates = [{"sample_id": s, "raw_score": str(v)} for s, v in fidelity.items()]

    ranked = score_against_fidelity(statistics, candidates)
    assert ranked, "expected some scored signals"
    assert ranked[0].statistic == "accel", [r.statistic for r in ranked]
    assert ranked[0].rho < -0.9, ranked[0].rho


def test_score_against_fidelity_needs_ground_truth():
    from phaselock.experiments.generation import score_against_fidelity

    assert score_against_fidelity([{"sample_id": "c", "source": "x", "block": "0",
                                    "step": "0", "phi_accel": "1"}], []) == []
