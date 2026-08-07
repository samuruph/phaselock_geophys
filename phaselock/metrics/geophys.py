"""The five GeoPhys geometric statistics.

From "GEOPHYS: The Geometry of Physical Plausibility". Given a video's per-frame
feature trajectory ``Z = (z_1 ... z_T)`` in ``R^{T x D}``, obtained by spatially
pooling a frozen encoder's (or, here, a diffusion model's) per-frame features:

    v_t = z_{t+1} - z_t                        t = 1 .. T-1     displacement
    s_t = ||v_t||_2                                             speed
    theta_t = arccos(<v_t, v_{t+1}> / (||v_t|| ||v_{t+1}||))     turning angle
    a_t = v_{t+1} - v_t = z_{t+2} - 2 z_{t+1} + z_t              acceleration
    eps_t = z_{t+1} - zhat_{t+1}                                 AR residual

    phi_speed = std({s_t})           temporal standard deviation of speed
    phi_curv  = mean({theta_t})      mean turning angle
    phi_ang   = std({theta_t})       turning-angle consistency
    phi_accel = mean({||a_t||^2})    note: SQUARED norm
    phi_perr  = mean({||eps_t||})    note: UNSQUARED norm

``s_t`` and ``theta_t`` are per-frame intermediates, not statistics; the statistics
are their temporal summaries. For all five, larger means less regular, which the
paper reads as less physically plausible.

Everything here is differentiable with respect to the trajectory, because
:mod:`phaselock.metrics.flow_geometry` differentiates the statistics to obtain their
rate of change under the model's own flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

STATISTICS: tuple[str, ...] = ("speed", "curv", "ang", "accel", "perr")

ResidualFit = Literal["span", "ridge", "scalar"]

_EPS = 1e-12


def as_float(x: torch.Tensor) -> torch.Tensor:
    """Promote to float32, but never demote float64.

    Probe tensors arrive as fp16/bf16 and must be widened before differencing. Tests
    and gradient checks run in float64, where a forced float32 cast would put a 1e-3
    floor on any finite-difference comparison.
    """
    return x if x.dtype in (torch.float32, torch.float64) else x.to(torch.float32)


def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 norm with a finite gradient at the origin.

    ``Tensor.norm`` has an undefined gradient at zero, which a perfectly straight or
    perfectly stationary trajectory hits exactly. The offset is far below float32
    resolution for any realistic feature magnitude.
    """
    return torch.sqrt((x * x).sum(dim=dim) + 1e-24)


def _check(trajectory: torch.Tensor, minimum: int) -> torch.Tensor:
    if trajectory.ndim != 2:
        raise ValueError(f"expected a (T, D) trajectory, got shape {tuple(trajectory.shape)}")
    if trajectory.shape[0] < minimum:
        raise ValueError(
            f"need at least {minimum} frames for this statistic, got {trajectory.shape[0]}"
        )
    return trajectory


def displacements(trajectory: torch.Tensor) -> torch.Tensor:
    """``v_t = z_{t+1} - z_t``, shape ``(T-1, D)``.

    This is also exactly PhaseLock's latent delta operator when the trajectory is a
    sequence of VAE latents.
    """
    trajectory = _check(trajectory, 2)
    return trajectory[1:] - trajectory[:-1]


def speeds(trajectory: torch.Tensor) -> torch.Tensor:
    """``s_t = ||v_t||_2``, shape ``(T-1,)``."""
    return _safe_norm(displacements(trajectory))


def turning_angles(trajectory: torch.Tensor) -> torch.Tensor:
    """``theta_t``, the angle between consecutive displacements, shape ``(T-2,)``.

    The paper writes this as ``arccos(<v_t, v_{t+1}> / (||v_t|| ||v_{t+1}||))``. That
    form is mathematically correct but numerically poor: ``arccos`` loses roughly half
    the available precision near 0 and pi, where its derivative diverges, and a nearly
    straight trajectory sits exactly there. A perfectly straight float32 trajectory
    returns ~2e-4 rather than 0, and the gradient is unusable.

    We use the algebraically identical half-angle form on the normalised
    displacements, ``theta = 2*atan2(||a - b||, ||a + b||)``, which is well conditioned
    across the whole range and differentiable. This matters because
    :mod:`phaselock.metrics.flow_geometry` differentiates ``phi_curv`` and ``phi_ang``.
    """
    v = displacements(_check(trajectory, 3))
    a = v[:-1] / _safe_norm(v[:-1]).unsqueeze(-1)
    b = v[1:] / _safe_norm(v[1:]).unsqueeze(-1)
    return 2.0 * torch.atan2(_safe_norm(a - b), _safe_norm(a + b))


def accelerations(trajectory: torch.Tensor) -> torch.Tensor:
    """``a_t = z_{t+2} - 2 z_{t+1} + z_t``, shape ``(T-2, D)``."""
    v = displacements(_check(trajectory, 3))
    return v[1:] - v[:-1]


def _windows(trajectory: torch.Tensor, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding windows of ``order`` past frames and the frame each one predicts.

    Returns ``(past, target)`` with shapes ``(N, order, D)`` and ``(N, D)``, where
    ``N = T - order``.
    """
    n = trajectory.shape[0] - order
    past = torch.stack([trajectory[i : i + order] for i in range(n)])
    target = trajectory[order:]
    return past, target


def prediction_residuals(
    trajectory: torch.Tensor,
    order: int = 3,
    fit: ResidualFit = "span",
    ridge_lambda: float = 0.1,
) -> torch.Tensor:
    """``||eps_t||_2`` for each window, shape ``(T - order,)``.

    The paper describes this two ways that do not agree, and its code is unreleased.
    It says a linear auto-regressive predictor ``P_H : R^{H*D} -> R^D`` is fit "on past
    windows", but with ``H*D >> T`` that fit is underdetermined and the in-sample
    residual is identically zero -- for CogVideoX-5B, ``T=13`` latent frames and
    ``D=3072`` gives 10 windows against 9216 unknowns. It also says, geometrically,
    that ``eps_t`` is "the component of ``z_{t+1}`` orthogonal to the H-step linear
    span", which is well-posed and needs no fitting at all.

    ``fit`` selects between them:

    ``"span"`` (default)
        Per-window least-squares projection onto the affine span of the previous
        ``order`` frames. Well-posed, training-free, and matches the geometric
        description. This is the interpretation used everywhere unless overridden.
    ``"ridge"``
        The literal global predictor, made well-posed by ridge regularisation and
        solved in the dual (``N x N``, with ``N`` the window count) since the primal
        is ``H*D x H*D``. In-sample fit, so the residual scale depends on
        ``ridge_lambda``.
    ``"scalar"``
        ``zhat_{t+1} = sum_h c_h z_{t-h+1}`` with scalar coefficients shared across
        feature dimensions. Only ``order`` unknowns against ``N*D`` equations, so it
        is always overdetermined without regularisation.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    trajectory = _check(trajectory, order + 1)
    past, target = _windows(trajectory, order)

    if fit == "span":
        # Affine span through the window's frames, anchored at the most recent one.
        anchor = past[:, -1]                                   # (N, D)
        basis = (past[:, :-1] - anchor.unsqueeze(1))           # (N, order-1, D)
        offset = target - anchor                               # (N, D)
        if basis.shape[1] == 0:
            return _safe_norm(offset)
        design = basis.transpose(1, 2)                         # (N, D, order-1)
        # pinv rather than lstsq: differentiable, and tolerant of rank-deficient
        # windows (a stationary segment makes the basis collapse).
        coefficients = torch.linalg.pinv(design) @ offset.unsqueeze(-1)
        return _safe_norm(offset - (design @ coefficients).squeeze(-1))

    if fit == "ridge":
        x = past.flatten(1)                                    # (N, order*D)
        gram = x @ x.transpose(0, 1)                           # (N, N) dual form
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        # Scale the penalty by the Gram diagonal. An absolute lambda is meaningless
        # here: feature magnitudes differ by orders of magnitude between a VAE latent
        # and a DiT hidden state, and too small a penalty simply reproduces the
        # degenerate OLS fit this option exists to avoid.
        penalty = ridge_lambda * torch.diagonal(gram).mean()
        dual = torch.linalg.solve(gram + penalty * eye, target)
        return _safe_norm(target - gram @ dual)

    if fit == "scalar":
        design = past.permute(0, 2, 1).reshape(-1, order)      # (N*D, order)
        rhs = target.reshape(-1, 1)                            # (N*D, 1)
        gram = design.transpose(0, 1) @ design
        eye = torch.eye(order, dtype=gram.dtype, device=gram.device)
        coefficients = torch.linalg.solve(gram + _EPS * eye, design.transpose(0, 1) @ rhs)
        predicted = (design @ coefficients).reshape(target.shape)
        return _safe_norm(target - predicted)

    raise ValueError(f"unknown residual fit {fit!r}; expected one of span, ridge, scalar")


def _std(values: torch.Tensor) -> torch.Tensor:
    """Population standard deviation.

    The paper writes ``std`` over a complete finite set of per-frame values, so the
    population form is the natural reading. The choice is a constant factor for
    trajectories of equal length and so cannot change any within-pair comparison.
    """
    return values.std(unbiased=False)


@dataclass(frozen=True)
class GeoPhysSignals:
    """Per-frame signals plus the five scalar statistics.

    The per-frame arrays are kept because both source papers inspect *when* along a
    clip the signal spikes -- GeoPhys against EEG time courses, and the violation
    frame in the paired benchmarks.
    """

    speed: torch.Tensor
    turning_angle: torch.Tensor
    acceleration: torch.Tensor
    residual: torch.Tensor
    statistics: dict[str, torch.Tensor]


def geophys_signals(
    trajectory: torch.Tensor,
    order: int = 3,
    fit: ResidualFit = "span",
    ridge_lambda: float = 0.1,
) -> GeoPhysSignals:
    """Compute the per-frame signals and the five statistics for one trajectory."""
    trajectory = as_float(trajectory)
    s = speeds(trajectory)
    theta = turning_angles(trajectory)
    accel = _safe_norm(accelerations(trajectory))
    resid = prediction_residuals(trajectory, order=order, fit=fit, ridge_lambda=ridge_lambda)

    statistics = {
        "speed": _std(s),
        "curv": theta.mean(),
        "ang": _std(theta),
        "accel": accel.pow(2).mean(),
        "perr": resid.mean(),
    }
    return GeoPhysSignals(
        speed=s, turning_angle=theta, acceleration=accel, residual=resid, statistics=statistics
    )


def geophys_statistics(
    trajectory: torch.Tensor,
    order: int = 3,
    fit: ResidualFit = "span",
    ridge_lambda: float = 0.1,
) -> dict[str, torch.Tensor]:
    """The five statistics as 0-dim tensors, differentiable w.r.t. ``trajectory``."""
    return geophys_signals(
        trajectory, order=order, fit=fit, ridge_lambda=ridge_lambda
    ).statistics


def statistic(
    name: str,
    trajectory: torch.Tensor,
    order: int = 3,
    fit: ResidualFit = "span",
    ridge_lambda: float = 0.1,
) -> torch.Tensor:
    """A single named statistic, for callers that differentiate one at a time."""
    if name not in STATISTICS:
        raise ValueError(f"unknown statistic {name!r}; expected one of {STATISTICS}")
    return geophys_statistics(
        trajectory, order=order, fit=fit, ridge_lambda=ridge_lambda
    )[name]
