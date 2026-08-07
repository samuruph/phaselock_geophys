"""Metrics coupling GeoPhys trajectory geometry to the model's own flow velocity.

GeoPhys's "velocity" runs along the **frame** axis of a video: ``v_f = z_{f+1} - z_f``.
A flow-matching sampler's velocity runs along the **denoising** axis: ``u = dz/dtau``.
These are different objects, but they are linked exactly, because the frame-difference
operator ``D`` and spatial mean-pooling are both linear and therefore commute with the
sampler ODE::

    d(D z)/dtau  =  D (dz/dtau)  =  D u(z, tau)

That is: *the flow velocity of the GeoPhys motion field is the GeoPhys motion field of
the flow velocity.* Everything in this module follows from that identity.

The identity is exact whenever the trajectory is an affine function of the ODE state --
true for the VAE latent, the clean-latent estimate ``x0_hat``, and the drift itself. It
is **not** true for DiT hidden states, which are a nonlinear function of the state and
have no ``dh/dtau``; there, :func:`geometric_drift_empirical` estimates the same
quantity by finite differences across recorded steps. Results carry an
:class:`DriftEstimator` tag so the two are never averaged together.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from .geophys import STATISTICS, ResidualFit, as_float, displacements, geophys_statistics

_EPS = 1e-12


class DriftEstimator(str, Enum):
    """How a drift value was obtained. Never mix the two in one aggregate."""

    EXACT = "exact"
    """Analytic inner product with the true PF-ODE drift."""

    EMPIRICAL = "empirical"
    """Finite difference across consecutive recorded denoising steps."""


@dataclass(frozen=True)
class FlowCoupling:
    """Flow-coupling measurements at one point on the denoising trajectory."""

    tau: float
    drift: dict[str, float]
    """Signed rate of change of each GeoPhys statistic under the flow."""

    estimator: DriftEstimator
    alignment: Optional[float] = None
    weighted_alignment: Optional[float] = None
    erosion: Optional[float] = None


def geometric_drift(
    trajectory: torch.Tensor,
    flow: torch.Tensor,
    order: int = 3,
    fit: ResidualFit = "span",
) -> dict[str, float]:
    """Rate of change of each GeoPhys statistic under the model's own flow.

    ``g_sigma(tau) = <grad_z phi_sigma(z), u(z, tau)>``

    A negative value means this denoising step is making the trajectory **more**
    geometrically regular; a positive value means it is **eroding** it. That is
    PhaseLock's erosion thesis stated per step, generalised from the first-order term
    to all five statistics.

    The gradient is taken through the statistic only -- a ``(T, D)`` tensor -- never
    through the transformer, so this costs essentially nothing on top of a sampler step
    that has already been computed.

    Args:
        trajectory: Pooled per-frame trajectory ``(T, D)``.
        flow: The pooled PF-ODE drift at the same point, same shape. Pool the drift the
            same way the trajectory was pooled; because pooling is linear, that is the
            drift of the pooled trajectory.

    Returns:
        One signed float per statistic in :data:`~phaselock.metrics.geophys.STATISTICS`.
    """
    if trajectory.shape != flow.shape:
        raise ValueError(
            f"trajectory and flow must have the same shape, got "
            f"{tuple(trajectory.shape)} and {tuple(flow.shape)}"
        )
    z = as_float(trajectory.detach()).clone().requires_grad_(True)
    u = as_float(flow.detach())

    statistics = geophys_statistics(z, order=order, fit=fit)
    drift: dict[str, float] = {}
    for name in STATISTICS:
        (gradient,) = torch.autograd.grad(statistics[name], z, retain_graph=True)
        drift[name] = float((gradient * u).sum())
    return drift


def geometric_drift_empirical(
    trajectory_before: torch.Tensor,
    trajectory_after: torch.Tensor,
    delta_tau: float,
    order: int = 3,
    fit: ResidualFit = "span",
) -> dict[str, float]:
    """Finite-difference drift, for representations that are not ODE states.

    DiT hidden states have no analytic ``dh/dtau``, so their drift is estimated as
    ``[phi(h(tau_{k+1})) - phi(h(tau_k))] / (tau_{k+1} - tau_k)``. This is the same
    quantity :func:`geometric_drift` computes exactly, but it inherits the recording
    stride as its resolution: with only a handful of recorded steps it is a coarse
    secant, not a tangent.
    """
    if delta_tau == 0:
        raise ValueError("delta_tau must be non-zero")
    before = geophys_statistics(trajectory_before, order=order, fit=fit)
    after = geophys_statistics(trajectory_after, order=order, fit=fit)
    return {name: float((after[name] - before[name]) / delta_tau) for name in STATISTICS}


def transport_alignment(
    trajectory: torch.Tensor, flow: torch.Tensor
) -> tuple[float, float]:
    """Cosine between the frame-wise motion and its rate of change under the flow.

    ``rho_f = cos((D z)_f, (D u)_f)``, where ``D u`` is exactly ``d(D z)/dtau`` by the
    commutation identity. Positive means the step is *growing* the motion already
    present; negative means it is *rewriting* it.

    This is the exact first-order special case of :func:`geometric_drift`, and it is
    the quantity that maps one-to-one onto PhaseLock's latent delta.

    Returns:
        ``(mean, displacement_weighted_mean)``. The weighted form uses ``||(D z)_f||``
        as the weight so that near-static frames, whose cosine is dominated by noise,
        do not carry the same vote as frames where something is actually moving.
    """
    dz = displacements(as_float(trajectory))
    du = displacements(as_float(flow))
    cosine = torch.nn.functional.cosine_similarity(dz, du, dim=-1, eps=_EPS)

    weights = dz.norm(dim=-1)
    total = weights.sum()
    weighted = (cosine * weights).sum() / total if total > _EPS else cosine.mean()
    return float(cosine.mean()), float(weighted)


def erosion_rate(trajectory: torch.Tensor, flow: torch.Tensor) -> float:
    """``mean_f ||(D u)_f|| / ||(D z)_f||`` -- how fast motion is being restructured.

    Large values mean the denoising step is changing the motion field by a lot relative
    to how much motion there is, regardless of direction.
    """
    dz = displacements(as_float(trajectory))
    du = displacements(as_float(flow))
    return float((du.norm(dim=-1) / (dz.norm(dim=-1) + _EPS)).mean())


def flow_coupling(
    trajectory: torch.Tensor,
    flow: torch.Tensor,
    tau: float,
    order: int = 3,
    fit: ResidualFit = "span",
) -> FlowCoupling:
    """All exact flow-coupling metrics at one denoising step."""
    alignment, weighted = transport_alignment(trajectory, flow)
    return FlowCoupling(
        tau=tau,
        drift=geometric_drift(trajectory, flow, order=order, fit=fit),
        estimator=DriftEstimator.EXACT,
        alignment=alignment,
        weighted_alignment=weighted,
        erosion=erosion_rate(trajectory, flow),
    )


def guidance_alignment(guidance: torch.Tensor, flow: torch.Tensor) -> float:
    """Cosine between an externally injected guidance signal and the model's own flow.

    Not part of the experimental programme, but the machinery is already here: with
    PhaseLock's update ``z[1:] += lambda * G``, ``G = M_prior - D z``, this answers
    whether the guidance pushes with or against the flow the model would have followed.
    Pass ``G`` lifted to the full frame axis and the pooled drift.
    """
    return float(
        torch.nn.functional.cosine_similarity(
            as_float(guidance).flatten(), as_float(flow).flatten(), dim=0, eps=_EPS
        )
    )
