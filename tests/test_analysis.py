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


def test_every_figure_renders(tmp_path):
    written = render_all(
        synthetic_rows(),
        tmp_path,
        external=dict(source="dinov2", block=-1, step=0, statistic="speed", kind="phi",
                      accuracy=0.722, ci_low=0.589, ci_high=0.835, auc=0.639, n_pairs=800),
        sweep=[dict(num_steps=k, blur_sigma=s, phi_curv=0.3 + 0.004 * k)
               for k in (2, 50) for s in (0.0, 16.0)],
    )
    assert len(written) >= 5
    for path in written:
        assert path.exists() and path.stat().st_size > 5000


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
    """A reader who learns 'curvature is aqua' must not be misled by the next figure."""
    assert len(set(palette.STATISTIC_COLOURS.values())) == len(palette.STATISTIC_COLOURS)
    for name in ("speed", "curv", "ang", "accel", "perr"):
        assert name in palette.STATISTIC_COLOURS
        assert palette.statistic_label(name) != name


def test_categorical_palette_is_the_validated_order():
    assert palette.CATEGORICAL[0] == "#2a78d6"
    assert palette.PRIMARY != palette.REFERENCE
