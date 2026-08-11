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

import functools
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


@functools.lru_cache(maxsize=None)
def amplification(name: str, frames: int = 13, trials: int = 16) -> float:
    """How far a correction written onto the latent overshoots what was asked for.

    A frame is shared between neighbouring windows: frame ``i`` is in both ``z[i]-z[i-1]``
    and ``z[i+1]-z[i]``, so moving it to correct one window necessarily disturbs the
    other. Writing ``d`` onto ``z`` therefore moves ``T(z)`` by ``T(d)``, not by ``d``,
    and the number of neighbours disturbed is the operator's order: 1 for ``motion``, 2
    for ``accel``, 3 for ``jerk``. Measured overshoot is 1.4x, 2.3x and 4.1x, matching
    ``sqrt(C(2n, n))`` for an n-th difference of an uncorrelated field.

    **This is not a claim that guidance should be exact.** PhaseLock is a soft nudge with
    a decaying lambda -- it biases the trajectory toward the prior's motion rather than
    projecting onto it, and disturbing neighbours is part of that diffuse bias. What is
    not intended is the *size* of the intervention changing by 4x when the operator
    changes, which at third order stops the iteration settling: over a guided schedule the
    latent inflates and the video dissolves.

    So this exists to keep a nudge a nudge. Dividing lambda by the overshoot, relative to
    ``motion``, makes one unit of lambda the same intervention for every operator -- which
    the ablation needs anyway, since otherwise the settings are not comparable and a
    ranking would partly measure step size.

    The operators are linear and shift-invariant along the frame axis, so this depends
    only on the operator and the frame count, never on the clip -- hence the cache.
    Probed rather than derived because ``perr``'s span projection has no closed form.
    """
    operator = OPERATORS[name]
    ratios = []
    for seed in range(trials):
        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(frames, 4, 2, 2, dtype=torch.float64, generator=generator)
        d = torch.randn(frames, 4, 2, 2, dtype=torch.float64, generator=generator)
        moved = z.clone()
        moved[operator.anchor:] = z[operator.anchor:] + d[operator.anchor:]
        ratios.append(
            float((operator(moved) - operator(z)).norm() / d[operator.anchor:].norm())
        )
    return sum(ratios) / len(ratios)


def strength_scale(name: str, frames: int = 13, reference: str = "motion") -> float:
    """Factor on ``lambda`` that makes ``name`` as strong an intervention as ``reference``.

    Normalised to ``motion`` rather than to 1 so the published arm is left exactly alone:
    ``strength_scale("motion") == 1.0``, and the reproduction stays byte-identical.
    """
    return amplification(reference, frames) / amplification(name, frames)


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
