"""What a single latent holds, and whether that is a usable prior.

A causal VAE bundles four video frames into one latent (the first latent holds only
frame 0). That has a consequence for every frame-difference method, PhaseLock's latent
delta included: **motion inside a latent is invisible to a difference along the latent
axis**. A latent whose four frames contain a fast bounce looks identical to a static one
if its endpoints happen to match.

So there are two separable questions:

1. *How much motion is hidden inside each latent?* Measurable only by decoding, which is
   the ground truth this module correlates against.
2. *Can a statistic of the latent itself predict it?* If some cheap latent statistic
   tracks the hidden motion, it is a prior that a frame difference cannot see -- and a
   candidate signal for physical plausibility that PhaseLock leaves unused.

Everything here reduces a latent to scalars per latent frame, so plausible and violated
clips can be compared directly and along the denoising trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..backends.base import LatentSpec, num_latent_frames


def frames_for_latent(index: int, spec: LatentSpec) -> range:
    """Which video frames a given latent frame encodes.

    The causal VAE gives latent 0 the single first frame; every later latent covers
    ``temporal_ratio`` frames. Getting this mapping wrong silently misaligns the
    hidden-motion ground truth against the latent statistics.
    """
    ratio = spec.temporal_ratio
    if index == 0:
        return range(0, 1)
    start = (index - 1) * ratio + 1
    return range(start, start + ratio)


def hidden_motion(frames: torch.Tensor, spec: LatentSpec) -> torch.Tensor:
    """Pixel motion *inside* each latent's frame group, shape ``(T_latent,)``.

    Mean absolute frame-to-frame difference within the group. This is the quantity a
    latent-axis difference cannot see. Latent 0 covers one frame and so has none.
    """
    if frames.ndim != 4:
        raise ValueError(f"expected (F, C, H, W) frames, got {tuple(frames.shape)}")
    latent_frames = num_latent_frames(frames.shape[0], spec)

    out = torch.zeros(latent_frames)
    for index in range(latent_frames):
        group = [f for f in frames_for_latent(index, spec) if f < frames.shape[0]]
        if len(group) < 2:
            continue
        block = frames[group].float()
        out[index] = (block[1:] - block[:-1]).abs().mean()
    return out


@dataclass(frozen=True)
class LatentStatistics:
    """Per-latent-frame scalars, each shape ``(T_latent,)``."""

    channel_std: torch.Tensor
    """Std over channels of the spatial mean -- how much the latent's channels disagree."""

    spatial_std: torch.Tensor
    """Std over space, averaged across channels -- how much structure it carries."""

    spatial_gradient: torch.Tensor
    """Mean absolute spatial gradient -- high-frequency detail, the part a blur removes."""

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "channel_std": self.channel_std,
            "spatial_std": self.spatial_std,
            "spatial_gradient": self.spatial_gradient,
        }


def latent_statistics(latents: torch.Tensor) -> LatentStatistics:
    """Reduce a canonical ``(T, C, H, W)`` latent to per-frame scalars."""
    if latents.ndim != 4:
        raise ValueError(f"expected canonical (T, C, H, W) latents, got {tuple(latents.shape)}")
    z = latents.float()
    dy = (z[:, :, 1:, :] - z[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    dx = (z[:, :, :, 1:] - z[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    return LatentStatistics(
        channel_std=z.mean(dim=(2, 3)).std(dim=1),
        spatial_std=z.std(dim=(2, 3)).mean(dim=1),
        spatial_gradient=0.5 * (dy + dx),
    )


def inter_latent_motion(latents: torch.Tensor) -> torch.Tensor:
    """``||z_{t+1} - z_t||`` per transition, shape ``(T-1,)``.

    PhaseLock's latent delta, and GeoPhys's first-order velocity, reduced to a magnitude.
    Included so it can be compared against the motion it cannot see.
    """
    z = latents.float()
    return (z[1:] - z[:-1]).flatten(1).norm(dim=1)


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two per-frame series."""
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if float(denominator) > 1e-12 else 0.0


def aggregate(series: Sequence[torch.Tensor]) -> tuple[list[float], list[float], list[float]]:
    """Mean and standard deviation across clips, plus the shared index."""
    if not series:
        return [], [], []
    length = min(len(s) for s in series)
    stacked = torch.stack([s[:length].float() for s in series])
    return (
        list(range(length)),
        stacked.mean(dim=0).tolist(),
        stacked.std(dim=0).tolist() if len(series) > 1 else [0.0] * length,
    )


def build_profiles(
    plausible: Sequence[torch.Tensor],
    violated: Sequence[torch.Tensor],
    plausible_tau: Optional[Sequence[tuple[list[float], list[float]]]] = None,
    violated_tau: Optional[Sequence[tuple[list[float], list[float]]]] = None,
) -> dict[str, dict]:
    """Shape the per-clip series into what :func:`~phaselock.analysis.figures.latent_motion_profile` draws."""
    profiles: dict[str, dict] = {}
    for label, series, taus in (
        ("plausible", plausible, plausible_tau),
        ("violated", violated, violated_tau),
    ):
        entry: dict[str, object] = {"per_frame": aggregate(series)}
        if taus:
            length = min(len(t[0]) for t in taus)
            grid = taus[0][0][:length]
            values = torch.tensor([t[1][:length] for t in taus], dtype=torch.float32)
            entry["per_tau"] = (
                grid,
                values.mean(dim=0).tolist(),
                values.std(dim=0).tolist() if len(taus) > 1 else [0.0] * length,
            )
        profiles[label] = entry
    return profiles
