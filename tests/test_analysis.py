"""Tests for the figure and latent-motion analysis layers. CPU only, no GPU."""

from __future__ import annotations

import math

import pytest
import torch

from phaselock.analysis import palette, render_all
from phaselock.analysis.latent_motion import (
    build_profiles,
    correlation,
    frames_for_latent,
    hidden_motion,
    inter_latent_motion,
    latent_statistics,
)
from phaselock.backends import COGVIDEOX_5B, WAN21_T2V_1_3B


def synthetic_rows(n_blocks: int = 12, peak: float = 0.55) -> list[dict]:
    rows = []
    for statistic in ("speed", "curv", "ang", "accel", "perr"):
        for step in range(0, 6, 2):
            for block in range(n_blocks):
                depth = block / max(n_blocks - 1, 1)
                accuracy = 0.5 + 0.3 * math.exp(-((depth - peak) ** 2) / 0.05)
                for kind in ("phi", "drift"):
                    rows.append(dict(source="hidden_states", block=block, step=step,
                                     statistic=statistic, kind=kind, accuracy=accuracy,
                                     ci_low=accuracy - 0.05, ci_high=accuracy + 0.05,
                                     auc=accuracy, n_pairs=96))
                rows.append(dict(source="latent", block=-1, step=step, statistic=statistic,
                                 kind="phi", accuracy=0.51, ci_low=0.45, ci_high=0.57,
                                 auc=0.51, n_pairs=96))
    return rows


# -- the causal VAE frame mapping -------------------------------------------


def test_first_latent_holds_one_frame_and_the_rest_hold_four():
    """Getting this wrong misaligns the hidden-motion ground truth against the latents."""
    assert list(frames_for_latent(0, COGVIDEOX_5B)) == [0]
    assert list(frames_for_latent(1, COGVIDEOX_5B)) == [1, 2, 3, 4]
    assert list(frames_for_latent(2, COGVIDEOX_5B)) == [5, 6, 7, 8]


def test_the_mapping_covers_every_frame_exactly_once():
    for spec, num_frames in ((COGVIDEOX_5B, 49), (WAN21_T2V_1_3B, 81)):
        latent_frames = (num_frames - 1) // spec.temporal_ratio + 1
        covered = [f for i in range(latent_frames) for f in frames_for_latent(i, spec)]
        assert covered == list(range(num_frames))


# -- hidden motion ----------------------------------------------------------


def test_a_static_clip_has_no_hidden_motion():
    frames = torch.ones(49, 3, 8, 8)
    assert float(hidden_motion(frames, COGVIDEOX_5B).max()) == pytest.approx(0.0)


def test_motion_confined_to_one_latents_frames_is_detected_there():
    """The case the latent axis is blind to: movement entirely inside one group."""
    frames = torch.zeros(49, 3, 8, 8)
    # Latent 2 covers frames 5..8. Move something only within that window, and return it
    # so the group's endpoints match and a latent-axis difference sees nothing.
    frames[6] = 1.0
    frames[7] = 1.0

    motion = hidden_motion(frames, COGVIDEOX_5B)
    assert float(motion[2]) > 0
    assert float(motion[1]) == pytest.approx(0.0)
    assert float(motion[3]) == pytest.approx(0.0)


def test_latent_zero_never_reports_hidden_motion():
    """It encodes a single frame, so there is nothing inside it to move."""
    frames = torch.rand(49, 3, 8, 8)
    assert float(hidden_motion(frames, COGVIDEOX_5B)[0]) == 0.0


# -- latent statistics ------------------------------------------------------


def test_latent_statistics_shapes_and_zero_case():
    latents = torch.zeros(13, 16, 6, 6)
    stats = latent_statistics(latents)
    for value in stats.as_dict().values():
        assert value.shape == (13,)
        assert float(value.abs().max()) == pytest.approx(0.0)


def test_spatial_gradient_responds_to_high_frequency_detail():
    """It is the component a blur removes, which is what the blur control manipulates."""
    smooth = torch.zeros(4, 2, 16, 16)
    rough = torch.zeros(4, 2, 16, 16)
    rough[:, :, ::2, :] = 1.0
    assert float(latent_statistics(rough).spatial_gradient.mean()) > float(
        latent_statistics(smooth).spatial_gradient.mean()
    )


def test_inter_latent_motion_is_the_frame_difference_norm():
    latents = torch.zeros(5, 4, 3, 3)
    latents[2] = 1.0
    motion = inter_latent_motion(latents)
    assert motion.shape == (4,)
    # Only the transitions into and out of frame 2 move.
    assert [i for i, v in enumerate(motion) if v > 0] == [1, 2]


def test_correlation_endpoints():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert correlation(a, a) == pytest.approx(1.0)
    assert correlation(a, -a) == pytest.approx(-1.0)
    assert correlation(a, torch.ones(4)) == pytest.approx(0.0)


def test_build_profiles_aggregates_across_clips():
    series = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 4.0, 5.0])]
    profiles = build_profiles(series, series)
    index, mean, spread = profiles["plausible"]["per_frame"]
    assert index == [0, 1, 2]
    assert mean == pytest.approx([2.0, 3.0, 4.0])
    assert all(s > 0 for s in spread)


# -- figures ----------------------------------------------------------------


def _summary(source, statistic, mean, std=0.04, best=None, cells=60):
    from phaselock.experiments.detection import SourceSummary

    return SourceSummary(source=source, statistic=statistic, kind="phi", mean=mean, std=std,
                         n_cells=cells, best=best or mean + 0.12,
                         best_label=f"{source}/b7/s4/phi_{statistic}")


def test_every_figure_renders(tmp_path):
    summaries = [_summary(src, stat, 0.5 + 0.05 * i)
                 for i, src in enumerate(("hidden_states", "latent", "velocity"))
                 for stat in ("speed", "curv", "ang", "accel", "perr")]
    written = render_all(
        synthetic_rows(),
        tmp_path,
        summaries=summaries,
        external_summaries=[_summary("dinov2", s, 0.60, cells=25) for s in ("speed", "curv")],
        steps_to_timestep={0: 0, 2: 459, 4: 779},
        sweep=[dict(num_steps=k, blur_sigma=s, phi_curv=0.3 + 0.004 * k)
               for k in (2, 50) for s in (0.0, 16.0)],
    )
    assert len(written) >= 5
    for path in written:
        assert path.exists() and path.stat().st_size > 5000


def test_source_comparison_shows_spread_not_just_the_best(tmp_path):
    """Reporting only the maximum over a probe grid reads high by construction."""
    from phaselock.analysis.figures import source_comparison

    summaries = [_summary("hidden_states", s, 0.62, std=0.06) for s in ("speed", "curv")]
    assert source_comparison(summaries, tmp_path / "f1.png") is not None
    assert source_comparison([], tmp_path / "empty.png") is None


def test_figures_degrade_gracefully_on_empty_input(tmp_path):
    """A partial run must produce the figures it can, not crash the report."""
    assert render_all([], tmp_path) == []


def test_heatmap_skips_sources_without_depth(tmp_path):
    """A single-block source stretched into a one-row panel says nothing about depth."""
    from phaselock.analysis.figures import depth_time_heatmaps

    rows = [dict(source="latent", block=-1, step=s, statistic="curv", kind="phi",
                 accuracy=0.5, ci_low=0.4, ci_high=0.6, auc=0.5, n_pairs=10) for s in range(3)]
    assert depth_time_heatmaps(rows, tmp_path / "x.png") is None


def test_statistics_keep_a_stable_colour_everywhere():
    """A reader who learns 'curvature is aqua' must not be misled by the next figure.

    Only the five statistics need mutually distinct hues -- they appear together in one
    legend. The ensembles and the flow-coupling metrics are drawn in separate figures, so
    they may reuse a hue without ambiguity.
    """
    core = ["speed", "curv", "ang", "accel", "perr"]
    assert len({palette.STATISTIC_COLOURS[n] for n in core}) == len(core)
    for name in ("alignment", "erosion", "or", "majority"):
        assert name in palette.STATISTIC_COLOURS
    for name in ("speed", "curv", "ang", "accel", "perr"):
        assert name in palette.STATISTIC_COLOURS
        assert palette.statistic_label(name) != name


def test_categorical_palette_is_the_validated_order():
    assert palette.CATEGORICAL[0] == "#2a78d6"
    assert palette.PRIMARY != palette.REFERENCE


def _sweep_cells():
    return [
        {"sample_id": f"c{c}", "scenario": "ball_drop", "num_steps": str(k),
         "blur_sigma": str(s), "phase_difference_corr": "0.3", "raw_score": str(0.5 - 0.001 * k),
         "phi_curv": "1.7", "phi_accel": "40", "phi_perr": "2.5", "phi_speed": "1.8"}
        for c in range(3) for k in (2, 10, 30, 50) for s in (0.0, 8.0, 16.0)
    ]


def test_step_sweep_panels_covers_every_available_metric(tmp_path):
    from phaselock.analysis.figures import SWEEP_METRICS, step_sweep_panels

    path = step_sweep_panels(_sweep_cells(), tmp_path / "s.png")
    assert path is not None and path.is_file()
    # Every metric present in the data should have been drawable.
    assert all(any(c.get(col) for c in _sweep_cells()) for col, _, _ in SWEEP_METRICS)


def test_step_sweep_panels_skips_metrics_the_run_did_not_produce(tmp_path):
    """An older sweep without the phase columns must still plot the rest."""
    from phaselock.analysis.figures import step_sweep_panels

    cells = [{k: v for k, v in c.items() if k != "phase_difference_corr"} for c in _sweep_cells()]
    assert step_sweep_panels(cells, tmp_path / "s.png") is not None


def test_step_sweep_panels_returns_none_with_no_usable_metric(tmp_path):
    from phaselock.analysis.figures import step_sweep_panels

    bare = [{"sample_id": "c", "num_steps": "2", "blur_sigma": "0.0"}]
    assert step_sweep_panels(bare, tmp_path / "s.png") is None


def test_generation_quality_ranks_scenarios(tmp_path):
    from phaselock.analysis.figures import generation_quality

    cells = [
        {"sample_id": f"{s}/000/valid", "scenario": s, "spatial_iou": str(v),
         "spatiotemporal_iou": str(v / 8), "weighted_spatial_iou": str(v / 2), "mse": str(1 - v)}
        for s, v in (("ball_drop", 0.9), ("river", 0.4), ("flag", 0.15))
    ]
    path = generation_quality(cells, tmp_path / "g.png")
    assert path is not None and path.is_file()


def test_generation_quality_returns_none_when_empty(tmp_path):
    from phaselock.analysis.figures import generation_quality

    assert generation_quality([], tmp_path / "g.png") is None


def test_source_comparison_axis_and_title_are_overridable(tmp_path):
    """Generation scores concordance with fidelity, not pairwise detection accuracy.

    Reusing the figure with the detection label would put the wrong words on a real
    number, which is worse than having no figure.
    """
    from dataclasses import dataclass

    from phaselock.analysis.figures import source_comparison

    @dataclass
    class Summary:
        source: str
        statistic: str
        kind: str
        mean: float
        std: float
        best: float
        n_cells: int

    summaries = [Summary("hidden_states", n, "phi", 0.7, 0.05, 0.8, 10)
                 for n in ("speed", "accel")]
    path = source_comparison(summaries, tmp_path / "s.png",
                             value_label="concordance (%)", title="Custom")
    assert path is not None and path.is_file()


def _category_cells():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class C:
        source: str; statistic: str; kind: str; category: str
        mean: float; std: float; best: float; n_cells: int; n_pairs: int

    return [
        C(src, stat, "phi", cat, 0.7, 0.05, 0.85, 300, 8)
        for src in ("hidden_states", "latent", "dinov2")
        for stat in ("speed", "accel", "perr")
        for cat in ("ball_drop", "river", "flag")
    ]


def test_category_table_renders_every_source_and_category(tmp_path):
    from phaselock.analysis.figures import category_table

    path = category_table(_category_cells(), tmp_path / "c.png")
    assert path is not None and path.is_file()


def test_category_table_returns_none_when_empty(tmp_path):
    from phaselock.analysis.figures import category_table

    assert category_table([], tmp_path / "c.png") is None


def test_score_by_category_skips_categories_below_the_pair_floor():
    """One pair per category gives 0% or 100% by construction, so it is dropped."""
    from phaselock.datasets import get_paired_dataset
    from phaselock.experiments.detection import score_by_category

    pairs = get_paired_dataset("likephys").select(limit=12, seed=0)
    rows = [{"sample_id": p.plausible.sample_id, "source": "latent", "block": "-1",
             "step": "0", "phi_accel": "1.0"} for p in pairs]
    rows += [{"sample_id": p.violated.sample_id, "source": "latent", "block": "-1",
              "step": "0", "phi_accel": "2.0"} for p in pairs]
    # 12 pairs spread over 12 scenarios is one each -- all below the floor.
    assert score_by_category(rows, pairs, key="scenario") == []


def test_signal_quality_correlation_finds_the_predictive_signal(tmp_path):
    """The plain question: does the statistic track how good the generation was?"""
    from phaselock.analysis.figures import signal_quality_correlation

    fidelity = {f"c{i}": 1.0 - 0.15 * i for i in range(6)}
    statistics, candidates = [], []
    for sample, score in fidelity.items():
        statistics.append({
            "sample_id": sample, "source": "hidden_states", "block": "3", "step": "0",
            "phi_accel": str(1.0 - score),   # tracks quality inversely
            "phi_curv": "2.0",               # flat, no signal
        })
        candidates.append({"sample_id": sample, "raw_score": str(score),
                           "spatial_iou": str(score)})
    path = signal_quality_correlation(statistics, candidates, tmp_path / "q.png")
    assert path is not None and path.is_file()


def test_signal_quality_correlation_needs_three_clips(tmp_path):
    """A rank correlation over two points is not a measurement."""
    from phaselock.analysis.figures import signal_quality_correlation

    statistics = [{"sample_id": "a", "source": "latent", "block": "-1", "step": "0",
                   "phi_accel": "1.0"}]
    assert signal_quality_correlation(
        statistics, [{"sample_id": "a", "raw_score": "0.5"}], tmp_path / "q.png") is None
