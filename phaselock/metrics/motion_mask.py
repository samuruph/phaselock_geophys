"""Motion-mask fidelity metrics, from the Physics-IQ evaluation protocol.

A binary motion mask is extracted by thresholding frame-to-frame pixel change, then
summarised four ways:

* **Spatial IoU** -- *where* action happens. Collapse the mask over time with a max, then
  IoU against the real one.
* **Spatiotemporal IoU** -- *where and when*. IoU on the full ``T x H x W`` mask. A model
  that gets the location right but the timing wrong scores well on the first and badly
  here.
* **Weighted Spatial IoU** -- *where and how much*. Collapse time by per-pixel action
  density, then sum-of-minimum over sum-of-maximum. Separates repeated motion (a
  pendulum) from one-pass motion (a rolling ball).
* **MSE** -- *how*. Plain pixel error against the real continuation. Lower is better.

:func:`physics_iq_score` combines them exactly as the official evaluator does: each IoU is
divided by its **physical variance** -- the same metric between two real takes of the scene
-- and MSE is subtracted after the same correction. The normalisation is what makes the
number interpretable: it is an empirical ceiling set by the scene's own repeatability, not
an arbitrary scale. It is computed by pooling over the whole evaluation set before
dividing, so it is a property of a *run*, not of a clip.

Used here for LikePhys continuations, which are rendered with a static camera and so
satisfy the protocol's assumptions directly, and later for Physics-IQ itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

_EPS = 1e-12


def motion_mask(frames: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Binary motion mask from frame-to-frame change, shape ``(F-1, H, W)``.

    Args:
        frames: ``(F, C, H, W)`` in [0, 1].
        threshold: Absolute per-pixel change, averaged over channels, above which a
            pixel counts as moving.
    """
    if frames.ndim != 4:
        raise ValueError(f"expected (F, C, H, W) frames, got {tuple(frames.shape)}")
    if frames.shape[0] < 2:
        raise ValueError("a motion mask needs at least two frames")
    difference = (frames[1:] - frames[:-1]).abs().mean(dim=1)
    return difference > threshold


def spatial_iou(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """IoU of the time-collapsed masks: does action happen in the right place?"""
    a = generated.any(dim=0)
    b = reference.any(dim=0)
    union = float((a | b).sum())
    if union < 1:
        return 1.0  # neither clip moved anywhere; vacuously in agreement
    return float((a & b).sum()) / union


def spatiotemporal_iou(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """IoU of the full masks: does action happen in the right place *and* time?"""
    union = float((generated | reference).sum())
    if union < 1:
        return 1.0
    return float((generated & reference).sum()) / union


def weighted_spatial_iou(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """Sum-of-min over sum-of-max on per-pixel action density.

    Distinguishes a pixel that moved once from one that moved throughout, which plain
    spatial IoU cannot.
    """
    a = generated.float().mean(dim=0)
    b = reference.float().mean(dim=0)
    denominator = float(torch.maximum(a, b).sum())
    if denominator < _EPS:
        return 1.0
    return float(torch.minimum(a, b).sum()) / denominator


def frame_mse(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean squared pixel error against the real continuation."""
    return float((generated - reference).pow(2).mean())


def psnr(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB, for inputs in [0, 1].

    Used for the inversion reconstruction check. Note what that check can and cannot
    tell you: "The Invisible Hand of Physics" reports probe accuracy collapsing from
    0.82 to 0.57 as integration steps drop from 100 to 20, while the reconstruction
    stays visually faithful. A high PSNR is necessary but nowhere near sufficient --
    it rules out a broken inversion, not a coarse trajectory.
    """
    error = frame_mse(generated, reference)
    if error < 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / error))


@dataclass(frozen=True)
class MotionMaskScores:
    """The four Physics-IQ component metrics for one generated clip."""

    spatial_iou: float
    spatiotemporal_iou: float
    weighted_spatial_iou: float
    mse: float

    @property
    def raw_score(self) -> float:
        """``sum(IoU) - MSE``: a per-clip diagnostic, **not** the benchmark's score.

        Un-normalised, so it is not comparable across scenes, and it weights the three
        IoUs equally with no physical-variance correction. Use :func:`physics_iq_score`
        for the reportable number.
        """
        return (
            self.spatial_iou
            + self.spatiotemporal_iou
            + self.weighted_spatial_iou
            - self.mse
        )


def motion_mask_scores(
    generated: torch.Tensor, reference: torch.Tensor, threshold: float = 0.05
) -> MotionMaskScores:
    """All four component metrics for one generated clip against its real continuation.

    Both arguments are ``(F, C, H, W)`` in [0, 1] and must already be temporally aligned
    and spatially matched.
    """
    if generated.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {tuple(generated.shape)} vs {tuple(reference.shape)}. "
            "Resample and letterbox both clips first."
        )
    generated_mask = motion_mask(generated, threshold)
    reference_mask = motion_mask(reference, threshold)
    return MotionMaskScores(
        spatial_iou=spatial_iou(generated_mask, reference_mask),
        spatiotemporal_iou=spatiotemporal_iou(generated_mask, reference_mask),
        weighted_spatial_iou=weighted_spatial_iou(generated_mask, reference_mask),
        mse=frame_mse(generated, reference),
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def physics_iq_score(
    scores: Sequence[MotionMaskScores], variances: Sequence[MotionMaskScores]
) -> float:
    """The benchmark's own score, matching ``physiq/calculate_iq_score.py``.

    Each IoU is divided by its **physical variance** -- the same metric computed between two
    real takes of the same scene -- and the three ratios are averaged. MSE is *subtracted*
    after the same bias correction, because it is an error rather than an agreement::

        score = mean(ST_IoU/var_ST, spatial/var_spatial, weighted/var_weighted)
                - (MSE - var_MSE)
        score = clip(100 * score, 0, 100)

    **This is a dataset-level quantity and cannot be computed per clip and averaged.** Every
    term above is a mean over the whole evaluation set *before* any division, which is what
    makes it stable: a single scene where nothing moves has a physical variance near zero,
    and per-clip ratios would divide by it and produce thousands of percent. Pooling first
    puts that scene's small numerator and small denominator into the same two sums, where it
    carries its own weight and nothing else's.

    Args:
        scores: One entry per generated clip.
        variances: The matching real-vs-real (take 2 against take 1) scores, in the same
            order. These are the empirical ceiling: the level at which a generation is
            indistinguishable, in motion-mask terms, from reality repeating itself.

    Returns:
        Percentage in [0, 100], clipped exactly as the official evaluator clips it.
    """
    if len(scores) != len(variances):
        raise ValueError(
            f"got {len(scores)} scores but {len(variances)} physical variances; "
            "every clip needs its own real-vs-real reference"
        )
    if not scores:
        raise ValueError("no clips to score")

    ratios = []
    for field in ("spatiotemporal_iou", "spatial_iou", "weighted_spatial_iou"):
        denominator = _mean([getattr(v, field) for v in variances])
        if abs(denominator) < _EPS:
            raise ValueError(
                f"the real-vs-real {field} pooled to ~0 across {len(scores)} clips, so it "
                "cannot normalise anything; check that the take-2 clips are aligned"
            )
        ratios.append(_mean([getattr(s, field) for s in scores]) / denominator)

    error = _mean([s.mse for s in scores]) - _mean([v.mse for v in variances])
    return float(min(max(100.0 * (sum(ratios) / 3.0 - error), 0.0), 100.0))
