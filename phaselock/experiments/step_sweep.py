"""Stage 7: does trajectory geometry reproduce the few-step effect?

PhaseLock's central observation is that a 2-step generation is *more physically
consistent* than the 50-step output from the same model and seed -- Physics-IQ 34.02 at
K=2 falling to 30.82 at K=50 -- while visual quality moves the other way (LPIPS
0.23 -> 0.19). The question here is whether the five geometric statistics rank the same
way.

**The confound, and the control.** A 2-step output is simply blurrier, so any feature
trajectory could look "more regular" purely from having less texture to move around.
PhaseLock hit this and controlled for it: Gaussian blur at sigma in {0, 8, 16} applied to
*every* arm, including the real reference, then check the ordering survives. Blurring only
the sharp arm would test a different hypothesis entirely.

Three measurements per (K, sigma) cell, deliberately not interchangeable:

* **DINOv2 GeoPhys** -- the published external path, and the primary result.
* **PhaseLock's own phase metric** -- inter-frame phase-difference correlation against
  the reference. Doubles as a reproduction check (the paper reports 0.358 at K=2 against
  0.100 at K=50, under sigma=16) and puts the spectral and geometric views on one axis.
* **Motion-mask fidelity** -- ground-truth-based, so it does not depend on GeoPhys being
  right, and arbitrates if the two disagree.

Guidance is off in every arm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from ..datasets.video_io import gaussian_blur
from ..encoders import DINOv2Encoder
from ..metrics.geophys import STATISTICS, geophys_statistics
from ..metrics.motion_mask import MotionMaskScores, motion_mask_scores
from ..metrics.spectral import magnitude_correlation, phase_coherence, phase_difference_correlation

logger = logging.getLogger(__name__)

DEFAULT_STEPS = (2, 10, 30, 50)
DEFAULT_BLUR = (0.0, 8.0, 16.0)


@dataclass(frozen=True)
class SweepCell:
    """One (step count, blur sigma) measurement for one conditioning setup."""

    sample_id: str
    scenario: str
    seed: int
    num_steps: int
    blur_sigma: float
    geophys: dict[str, float]
    phase_difference: float
    phase_coherence: float
    magnitude_correlation: float
    motion_mask: MotionMaskScores

    def flatten(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "sample_id": self.sample_id,
            "scenario": self.scenario,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "blur_sigma": self.blur_sigma,
            "phase_difference_corr": self.phase_difference,
            "phase_coherence": self.phase_coherence,
            "magnitude_corr": self.magnitude_correlation,
            "spatial_iou": self.motion_mask.spatial_iou,
            "spatiotemporal_iou": self.motion_mask.spatiotemporal_iou,
            "weighted_spatial_iou": self.motion_mask.weighted_spatial_iou,
            "mse": self.motion_mask.mse,
            "raw_score": self.motion_mask.raw_score,
        }
        row.update({f"phi_{name}": value for name, value in self.geophys.items()})
        return row

    @staticmethod
    def fieldnames() -> list[str]:
        return [
            "sample_id", "scenario", "seed", "num_steps", "blur_sigma",
            "phase_difference_corr", "phase_coherence", "magnitude_corr",
            "spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou", "mse", "raw_score",
        ] + [f"phi_{name}" for name in STATISTICS]


def measure_cell(
    generated: torch.Tensor,
    reference: torch.Tensor,
    encoder: DINOv2Encoder,
    *,
    sample_id: str,
    scenario: str,
    seed: int,
    num_steps: int,
    blur_sigma: float,
    layer: Optional[int] = None,
    ar_order: int = 3,
    residual_fit: str = "span",
) -> SweepCell:
    """All three measurements for one cell, with blur applied to both arms.

    ``generated`` and ``reference`` are unblurred ``(F, 3, H, W)`` in [0, 1]; the blur is
    applied here so it cannot be forgotten on one side.
    """
    if generated.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {tuple(generated.shape)} vs {tuple(reference.shape)}"
        )

    generated = gaussian_blur(generated, blur_sigma)
    reference = gaussian_blur(reference, blur_sigma)

    trajectory = encoder.encode(generated, layer=layer)
    return SweepCell(
        sample_id=sample_id,
        scenario=scenario,
        seed=seed,
        num_steps=num_steps,
        blur_sigma=blur_sigma,
        geophys={
            name: float(value)
            for name, value in geophys_statistics(
                trajectory, order=ar_order, fit=residual_fit
            ).items()
        },
        phase_difference=phase_difference_correlation(generated, reference),
        phase_coherence=phase_coherence(generated, reference),
        magnitude_correlation=magnitude_correlation(generated, reference),
        motion_mask=motion_mask_scores(generated, reference),
    )


def ordering_by_steps(
    cells: Sequence[SweepCell], metric: str, blur_sigma: float
) -> dict[int, float]:
    """Mean value of one metric per step count, at a fixed blur level."""
    totals: dict[int, list[float]] = {}
    for cell in cells:
        if cell.blur_sigma != blur_sigma:
            continue
        value = cell.geophys.get(metric) if metric in STATISTICS else cell.flatten().get(metric)
        if value is None:
            continue
        totals.setdefault(cell.num_steps, []).append(float(value))
    return {steps: sum(values) / len(values) for steps, values in sorted(totals.items())}


def blur_survival(
    cells: Sequence[SweepCell],
    metric: str,
    low_steps: int = 2,
    high_steps: int = 50,
    lower_is_more_regular: bool = True,
) -> dict[float, dict[str, float]]:
    """Does the few-step advantage survive the blur sweep?

    For each sigma, reports the mean metric at ``low_steps`` and ``high_steps`` and
    whether the few-step arm is still favoured.

    A metric where the advantage *vanishes* under blur is measuring sharpness, not
    physics. That is a real possible outcome and worth reporting: PhaseLock's spectral
    metric does survive the same control, so a geometric statistic that does not would
    separate the two families rather than waste the experiment.
    """
    out: dict[float, dict[str, float]] = {}
    for sigma in sorted({cell.blur_sigma for cell in cells}):
        means = ordering_by_steps(cells, metric, sigma)
        if low_steps not in means or high_steps not in means:
            continue
        low, high = means[low_steps], means[high_steps]
        favoured = low < high if lower_is_more_regular else low > high
        out[sigma] = {
            f"k{low_steps}": low,
            f"k{high_steps}": high,
            "gap": high - low if lower_is_more_regular else low - high,
            "few_step_favoured": float(favoured),
        }
    return out
