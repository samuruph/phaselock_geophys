"""Tests for the spectral and motion-mask metrics.

Both are reproductions of published protocols -- PhaseLock's Fig. 3a blur control and the
Physics-IQ evaluation -- so they are pinned against constructed cases where the right
answer is known by hand.
"""

from __future__ import annotations

import math

import pytest
import torch

from phaselock.datasets import gaussian_blur
from phaselock.metrics.motion_mask import (
    MotionMaskScores,
    motion_mask,
    motion_mask_scores,
    physics_iq_score,
    psnr,
    spatial_iou,
    spatiotemporal_iou,
    weighted_spatial_iou,
)
from phaselock.metrics.spectral import (
    inter_frame_phase_difference,
    low_frequency_mask,
    magnitude_correlation,
    pearson,
    phase_coherence,
    phase_difference_correlation,
    to_luminance,
)


def moving_square(frames: int = 12, size: int = 32, speed: int = 2) -> torch.Tensor:
    """A bright square translating at constant velocity."""
    video = torch.zeros(frames, 3, size, size)
    for index in range(frames):
        left = (index * speed) % (size - 6)
        video[index, :, 10:16, left : left + 6] = 1.0
    return video


# -- spectral ---------------------------------------------------------------


def test_pearson_endpoints():
    a = torch.randn(64, dtype=torch.float64)
    assert pearson(a, a) == pytest.approx(1.0)
    assert pearson(a, -a) == pytest.approx(-1.0)
    assert pearson(a, torch.zeros(64, dtype=torch.float64)) == pytest.approx(0.0)


def test_luminance_collapses_channels():
    assert to_luminance(torch.rand(4, 3, 8, 8)).shape == (4, 8, 8)
    grey = torch.rand(4, 1, 8, 8)
    assert torch.allclose(to_luminance(grey), grey[:, 0])


def test_phase_difference_is_wrapped_into_the_principal_branch():
    """Subtracting two `angle` values leaks 2*pi jumps that dominate the correlation."""
    difference = inter_frame_phase_difference(moving_square())
    assert difference.shape[0] == 11
    assert float(difference.min()) >= -math.pi - 1e-6
    assert float(difference.max()) <= math.pi + 1e-6


def test_a_clip_correlates_perfectly_with_itself():
    video = moving_square()
    assert phase_difference_correlation(video, video) == pytest.approx(1.0)
    assert phase_coherence(video, video) == pytest.approx(1.0)
    assert magnitude_correlation(video, video) == pytest.approx(1.0)


def test_matching_motion_beats_mismatched_motion():
    """The metric must reward reproducing the reference's dynamics, not its appearance."""
    reference = moving_square(speed=2)
    same = moving_square(speed=2)
    different = moving_square(speed=5)
    assert phase_difference_correlation(same, reference) > phase_difference_correlation(
        different, reference
    )


def test_blur_control_preserves_the_ranking_it_is_meant_to_test():
    """PhaseLock's control: blur every arm, check the ordering survives.

    Here the "good" clip reproduces the reference's motion and the "bad" one does not.
    Blurring all three must not reverse that, or the control could never detect anything.
    """
    reference = moving_square(speed=2)
    good = moving_square(speed=2)
    bad = moving_square(speed=5)

    for sigma in (0.0, 2.0, 4.0):
        blurred = [gaussian_blur(clip, sigma) for clip in (reference, good, bad)]
        assert phase_difference_correlation(blurred[1], blurred[0]) > phase_difference_correlation(
            blurred[2], blurred[0]
        ), f"ranking inverted at sigma={sigma}"


def test_low_frequency_mask_selects_a_centred_band():
    mask = low_frequency_mask((8, 16, 16), cutoff=0.4)
    assert mask.shape == (8, 16, 16)
    assert bool(mask[0, 0, 0])  # DC is always inside
    assert not bool(mask[4, 8, 8])  # the Nyquist corner is not
    assert 0 < int(mask.sum()) < mask.numel()


def test_low_frequency_mask_normalises_each_axis_separately():
    """A 13-frame by 60x90 volume must not be dominated by its longest axis."""
    square = low_frequency_mask((16, 16, 16), cutoff=0.4).float().mean()
    oblong = low_frequency_mask((4, 64, 64), cutoff=0.4).float().mean()
    assert abs(float(square) - float(oblong)) < 0.15


def test_spectral_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        phase_difference_correlation(moving_square(), moving_square(frames=8))


# -- motion mask ------------------------------------------------------------


def test_motion_mask_marks_only_where_something_moved():
    video = moving_square()
    mask = motion_mask(video)
    assert mask.shape == (11, 32, 32)
    # The square lives in rows 10..15, so nothing outside that band should move.
    assert int(mask[:, :10].sum()) == 0
    assert int(mask[:, 16:].sum()) == 0
    assert int(mask.sum()) > 0


def test_motion_mask_needs_at_least_two_frames():
    with pytest.raises(ValueError, match="at least two frames"):
        motion_mask(torch.zeros(1, 3, 8, 8))


def test_identical_clips_score_perfectly():
    video = moving_square()
    scores = motion_mask_scores(video, video)
    assert scores.spatial_iou == pytest.approx(1.0)
    assert scores.spatiotemporal_iou == pytest.approx(1.0)
    assert scores.weighted_spatial_iou == pytest.approx(1.0)
    assert scores.mse == pytest.approx(0.0)


def test_spatiotemporal_iou_punishes_bad_timing_that_spatial_iou_forgives():
    """The distinction the two metrics exist to draw: right place, wrong time."""
    reference = motion_mask(moving_square(speed=2))
    shifted = torch.roll(reference, shifts=5, dims=0)
    assert spatial_iou(shifted, reference) > spatiotemporal_iou(shifted, reference)


def test_weighted_iou_separates_repeated_from_one_pass_motion():
    """A pixel that moved every frame must not match one that moved once."""
    always = torch.ones(10, 4, 4, dtype=torch.bool)
    once = torch.zeros(10, 4, 4, dtype=torch.bool)
    once[0] = True
    assert spatial_iou(once, always) == pytest.approx(1.0)
    assert weighted_spatial_iou(once, always) == pytest.approx(0.1)


def test_a_static_generation_scores_badly_against_a_moving_reference():
    static = torch.zeros(12, 3, 32, 32)
    static[:, :, 10:16, 4:10] = 1.0
    scores = motion_mask_scores(static, moving_square())
    assert scores.spatial_iou < 0.5
    assert scores.spatiotemporal_iou < 0.5


def test_mask_metrics_agree_when_neither_clip_moves():
    still = torch.zeros(6, 3, 16, 16)
    scores = motion_mask_scores(still, still)
    assert scores.spatial_iou == pytest.approx(1.0)


def test_scores_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        motion_mask_scores(moving_square(), moving_square(size=16))


# -- Physics-IQ normalisation ----------------------------------------------


def test_normalising_against_the_real_ceiling_puts_a_second_take_at_100_percent():
    """The benchmark's definition: 100% means indistinguishable from reality repeating."""
    ceiling = MotionMaskScores(0.30, 0.24, 0.16, 0.01)
    assert physics_iq_score(ceiling, ceiling) == pytest.approx(100.0)

    worse = MotionMaskScores(0.15, 0.12, 0.08, 0.02)
    assert 0.0 < physics_iq_score(worse, ceiling) < 100.0


def test_score_without_a_ceiling_is_the_raw_combination():
    scores = MotionMaskScores(0.30, 0.24, 0.16, 0.01)
    assert physics_iq_score(scores) == pytest.approx(100.0 * (0.30 + 0.24 + 0.16 - 0.01))


def test_a_degenerate_ceiling_raises_rather_than_dividing_by_zero():
    ceiling = MotionMaskScores(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="ceiling scored"):
        physics_iq_score(MotionMaskScores(0.1, 0.1, 0.1, 0.0), ceiling)


def test_psnr_endpoints_and_ordering():
    """Used by the inversion reconstruction check, so its direction must be right."""
    reference = torch.rand(4, 3, 16, 16)
    assert psnr(reference, reference) == float("inf")

    close = (reference + 0.01).clamp(0, 1)
    far = (reference + 0.20).clamp(0, 1)
    assert psnr(close, reference) > psnr(far, reference)


def test_psnr_matches_the_closed_form():
    reference = torch.zeros(2, 3, 8, 8)
    generated = torch.full_like(reference, 0.1)
    # MSE = 0.01 -> 10*log10(1/0.01) = 20 dB
    assert psnr(generated, reference) == pytest.approx(20.0, abs=1e-6)
