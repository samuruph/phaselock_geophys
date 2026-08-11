"""Frame operators for guidance: what a prior is extracted from, and matched against.

PhaseLock constrains one quantity during sampling, its latent delta operator::

    T(z) = z[2:F] - z[1:F-1]

which is the **first difference along frames**. Every GeoPhys quantity built from that same
difference is a candidate for the same treatment, and the detection study says PhaseLock is
using the weakest of them: on the VAE latent, `phi_speed` -- the statistic of this very
operator -- reads 46.6%, below chance, while the residual, second and third differences read
72.6%, 67.3% and 67.2%.

So this module is the axis of the ablation. Each operator maps a canonical trajectory
``(F, C, H, W)`` to a tensor of *the same kind*, one value per frame, channel and spatial
position -- no pooling, no scalars, and therefore no gradients anywhere in the guidance path.

Each also declares how many leading frames it consumes, which becomes the anchor: an order-n
operator's output at index i depends on frames i..i+n, so writing the correction to
``z[n:]`` leaves the first n frames untouched. For ``motion`` that is n=1, exactly
PhaseLock's "the first frame remains unmodified as it serves as the image condition anchor".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .metrics.geophys import residual_vectors

# The default window for the residual operator, matching `metrics.geophys`.
RESIDUAL_ORDER = 3


def _first_difference(z: torch.Tensor) -> torch.Tensor:
    """``z[t+1] - z[t]``. PhaseLock's latent delta operator, verbatim."""
    return z[1:] - z[:-1]


def _second_difference(z: torch.Tensor) -> torch.Tensor:
    """``z[t+2] - 2 z[t+1] + z[t]`` -- acceleration."""
    v = _first_difference(z)
    return v[1:] - v[:-1]


def _third_difference(z: torch.Tensor) -> torch.Tensor:
    """``a[t+1] - a[t]`` -- jerk."""
    a = _second_difference(z)
    return a[1:] - a[:-1]


def _span_residual(z: torch.Tensor) -> torch.Tensor:
    """The component of ``z[t+1]`` orthogonal to the affine span of the previous frames.

    Fitted **per spatial position**: each ``(h, w)`` is its own ``C``-dimensional
    trajectory. That matches the elementwise character of the difference operators, where
    a position's correction depends only on that position. Flattening ``(C, H, W)`` into
    one long vector instead would fit a single affine span jointly across the whole frame,
    which is a different quantity -- one span for the image rather than one per position.
    """
    frames, channels, height, width = z.shape
    # (F, C, H, W) -> (H*W, F, C): one trajectory per position, batched through a single
    # vectorised pinv rather than a Python loop over 5400 positions.
    per_position = z.permute(2, 3, 0, 1).reshape(height * width, frames, channels)
    residual = residual_vectors(per_position, order=RESIDUAL_ORDER)
    windows = residual.shape[-2]
    return (
        residual.reshape(height, width, windows, channels)
        .permute(2, 3, 0, 1)
        .contiguous()
    )


@dataclass(frozen=True)
class FrameOperator:
    """One guidance target: what to measure, and how many frames it anchors.

    ``anchor`` is both the number of frames the operator consumes before producing its
    first value and the number of leading frames guidance leaves alone. Those are the same
    number by construction, which is what makes the image condition safe for every
    operator rather than only for ``motion``.
    """

    name: str
    apply: Callable[[torch.Tensor], torch.Tensor]
    anchor: int
    minimum_frames: int
    description: str

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(
                f"expected a canonical (F, C, H, W) trajectory, got {tuple(z.shape)}"
            )
        if z.shape[0] < self.minimum_frames:
            raise ValueError(
                f"operator {self.name!r} needs at least {self.minimum_frames} frames, "
                f"got {z.shape[0]}"
            )
        return self.apply(z)


OPERATORS: dict[str, FrameOperator] = {
    "motion": FrameOperator(
        "motion", _first_difference, anchor=1, minimum_frames=2,
        description="first difference -- PhaseLock's own latent delta operator",
    ),
    "accel": FrameOperator(
        "accel", _second_difference, anchor=2, minimum_frames=3,
        description="second difference -- acceleration",
    ),
    "jerk": FrameOperator(
        "jerk", _third_difference, anchor=3, minimum_frames=4,
        description="third difference -- the change in acceleration",
    ),
    "perr": FrameOperator(
        "perr", _span_residual, anchor=RESIDUAL_ORDER,
        minimum_frames=RESIDUAL_ORDER + 1,
        description="departure from the affine span of the previous frames",
    ),
}


def get_operator(name: str) -> FrameOperator:
    if name not in OPERATORS:
        raise ValueError(
            f"unknown guidance operator {name!r}; expected one of {sorted(OPERATORS)}"
        )
    return OPERATORS[name]
