"""The published GeoPhys path: five statistics on frozen external encoder features.

This exists to answer "is the implementation right?" before any internal number is
believed. GeoPhys reports 78-81% pairwise accuracy on LikePhys from a single frozen
backbone; landing near that validates the statistics, the pairing, the preprocessing and
the scoring rule all at once. Landing far from it means something upstream is wrong and
every internal result is uninterpretable.

It is also the yardstick the internal representations are measured against, and it is the
primary path for the Stage 7 step sweep, which follows the original paper in using DINOv2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..config import Config
from ..datasets import VideoPair, VideoSample, load_video
from ..encoders import DINOv2Encoder
from ..metrics.geophys import STATISTICS, geophys_statistics
from ..metrics.scoring import PairwiseResult, evaluate_pairs

TemporalPool = str
"""``"none"`` keeps one trajectory point per video frame; ``"latent"`` pools to the
backend's latent grid. See :func:`pool_to_latent_grid`."""


def pool_to_latent_grid(trajectory: "torch.Tensor", spec) -> "torch.Tensor":
    """Average per-frame features over each latent's frame group.

    Without this the external and internal paths are not comparable. A causal VAE folds
    four video frames into one latent, so Wan gives 21 trajectory points for an 81-frame
    clip while DINOv2 gives 81. Both the number of points and the spacing between them
    differ, and every GeoPhys statistic depends on both -- ``phi_speed`` is a standard
    deviation over ``T-1`` displacements, ``phi_curv`` a mean over ``T-2`` angles, and
    consecutive frames 1/4 as far apart produce systematically smaller displacements and
    different turning angles.

    Averaging the frames the VAE would have folded together is the matched operation: it
    reproduces the VAE's temporal pooling in feature space, leaving the two paths with the
    same trajectory length and the same temporal support per point.
    """
    from ..analysis.latent_motion import frames_for_latent
    from ..backends.base import num_latent_frames

    total = trajectory.shape[0]
    pooled = []
    for index in range(num_latent_frames(total, spec)):
        group = [f for f in frames_for_latent(index, spec) if f < total]
        if group:
            pooled.append(trajectory[group].mean(dim=0))
    return torch.stack(pooled)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalRow:
    """One clip's statistics from an external encoder at one readout layer."""

    sample_id: str
    label: int
    scenario: str
    violation: str
    layer: int
    statistics: dict[str, float]

    def flatten(self) -> dict[str, object]:
        row: dict[str, object] = {
            "sample_id": self.sample_id,
            "label": self.label,
            "scenario": self.scenario,
            "violation": self.violation,
            "encoder_layer": self.layer,
        }
        row.update({f"phi_{name}": value for name, value in self.statistics.items()})
        return row


def encode_sample(
    encoder: DINOv2Encoder,
    sample: VideoSample,
    pair: VideoPair,
    config: Config,
    layers: Optional[Sequence[int]] = None,
    num_frames: Optional[int] = None,
    temporal_pool: TemporalPool = "none",
) -> list[ExternalRow]:
    """Encode one clip and compute the statistics at each requested layer.

    All layers come from a single forward pass, so sweeping the readout costs nothing
    beyond the statistics themselves.

    The clip is loaded at the *backend's* native frame count rather than the encoder's
    preferred one, so the external and internal paths see the same temporal sampling.
    Comparing a 60-frame DINOv2 trajectory against a 13-latent-frame internal one would
    confound the representation with the frame rate. Spatial size comes from the encoder,
    which resizes internally anyway.

    ``num_frames`` overrides that for a standalone gate run, where matching the backend is
    an arbitrary constraint and more trajectory points means less noisy statistics.
    """
    frames = load_video(
        sample.path,
        num_frames=num_frames or _frame_budget(config),
        height=encoder.image_size,
        width=encoder.image_size,
        window=config.data.window,
        blur_sigma=config.data.blur_sigma,
    )

    trajectories = encoder.encode_all_layers(frames)
    if temporal_pool == "latent":
        from ..backends import get_spec

        spec = get_spec(config.backend.name)
        trajectories = {k: pool_to_latent_grid(v, spec) for k, v in trajectories.items()}
    elif temporal_pool != "none":
        raise ValueError(f"temporal_pool must be 'none' or 'latent', got {temporal_pool!r}")
    chosen = sorted(trajectories) if layers is None else list(layers)

    return [
        ExternalRow(
            sample_id=sample.sample_id,
            label=sample.label,
            scenario=pair.scenario,
            violation=pair.violation,
            layer=layer,
            statistics={
                name: float(value)
                for name, value in geophys_statistics(
                    trajectories[layer],
                    order=config.metrics.ar_order,
                    fit=config.metrics.residual_fit,
                ).items()
            },
        )
        for layer in chosen
    ]


def _frame_budget(config: Config) -> int:
    """The backend's native frame count, used to match temporal sampling."""
    from ..backends import get_spec

    return get_spec(config.backend.name).default_num_frames


def score_external(
    rows: Sequence[ExternalRow],
    pairs: Sequence[VideoPair],
    resamples: int = 1000,
) -> dict[tuple[int, str], PairwiseResult]:
    """Pairwise accuracy per ``(layer, statistic)``."""
    by_layer: dict[int, dict[str, ExternalRow]] = {}
    for row in rows:
        by_layer.setdefault(row.layer, {})[row.sample_id] = row

    results: dict[tuple[int, str], PairwiseResult] = {}
    for layer, by_sample in by_layer.items():
        usable = [
            pair
            for pair in pairs
            if pair.plausible.sample_id in by_sample and pair.violated.sample_id in by_sample
        ]
        if not usable:
            continue
        for name in STATISTICS:
            results[(layer, name)] = evaluate_pairs(
                [by_sample[p.plausible.sample_id].statistics[name] for p in usable],
                [by_sample[p.violated.sample_id].statistics[name] for p in usable],
                groups=[p.scenario for p in usable],
                resamples=resamples,
            )
    return results
