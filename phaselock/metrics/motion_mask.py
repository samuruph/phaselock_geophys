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

They combine into a single score as ``sum(IoU) - MSE``, normalised so that a second real
take of the same scene scores 100%. That normalisation is what makes the number
interpretable: it is an empirical ceiling set by the scene's own physical variance, not
an arbitrary scale.

Used here for LikePhys continuations, which are rendered with a static camera and so
satisfy the protocol's assumptions directly, and later for Physics-IQ itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

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
        """``sum(IoU) - MSE``, before normalisation against the real-vs-real ceiling."""
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


def physics_iq_score(
    scores: MotionMaskScores, ceiling: Optional[MotionMaskScores] = None
) -> float:
    """Normalise the combined score so the real-vs-real ceiling reads 100%.

    Args:
        ceiling: Scores of a second real take of the same scene against the first. This
            is the benchmark's empirical upper bound -- the level at which a generation
            is indistinguishable, in motion-mask terms, from reality repeating itself.
            Without it the raw score is returned as a percentage, which is not comparable
            across scenes.

    Returns:
        Percentage. 0 means no overlap with reality; 100 means it matched the noise floor.
    """
    if ceiling is None:
        return 100.0 * scores.raw_score
    denominator = ceiling.raw_score
    if abs(denominator) < _EPS:
        raise ValueError(
            "the real-vs-real ceiling scored ~0, so it cannot normalise anything; "
            "check that the take-2 clip is aligned with take 1"
        )
    return 100.0 * scores.raw_score / denominator
