"""CPU tests for the flow-coupling metrics. No GPU, no model weights.

The load-bearing claim is that the frame-difference operator commutes with the sampler
ODE, so that a GeoPhys statistic's rate of change under the model's own flow is a
single inner product. These tests check that claim numerically and pin the sign
convention, because a sign error would invert every conclusion drawn from it.
"""

from __future__ import annotations

import pytest
import torch

from phaselock.metrics.flow_geometry import (
    DriftEstimator,
    erosion_rate,
    flow_coupling,
    geometric_drift,
    geometric_drift_empirical,
    guidance_alignment,
    transport_alignment,
)
from phaselock.metrics.geophys import STATISTICS, displacements, geophys_statistics

T, D = 13, 48


@pytest.fixture
def rng() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def _randn(*shape, generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


# -- the commutation identity ----------------------------------------------


@pytest.mark.parametrize("tau", [0.0, 0.3, 1.0])
def test_frame_difference_commutes_with_a_linear_flow(rng, tau):
    """For z(tau) = z0 + tau*u, d(Dz)/dtau must equal Du exactly."""
    z0, u = _randn(T, D, generator=rng), _randn(T, D, generator=rng)
    h = 1e-6
    secant = (displacements(z0 + (tau + h) * u) - displacements(z0 + tau * u)) / h
    assert torch.allclose(secant, displacements(u), atol=1e-6)


@pytest.mark.parametrize("name", STATISTICS)
def test_geometric_drift_matches_a_central_finite_difference(rng, name):
    """The whole metric is <grad phi, u>; this is the check that it really is."""
    z, u = _randn(T, D, generator=rng), _randn(T, D, generator=rng)
    h = 1e-6
    expected = float(
        (geophys_statistics(z + h * u)[name] - geophys_statistics(z - h * u)[name]) / (2 * h)
    )
    assert geometric_drift(z, u)[name] == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_drift_of_curvature_against_a_closed_form_flow():
    """Uniform circular motion has phi_curv exactly equal to its turning rate.

    Perturbing the trajectory along ``dz/drate`` must therefore change ``phi_curv`` at
    a rate of exactly 1, and along ``-dz/drate`` at exactly -1. This pins both the
    magnitude and the sign against geometry rather than against another numerical
    routine.
    """
    rate, radius = 0.4, 2.0
    k = torch.arange(T, dtype=torch.float64)
    angle = k * rate
    z = radius * torch.stack([angle.cos(), angle.sin()], dim=1)
    # d z_k / d rate
    faster = radius * torch.stack([-k * angle.sin(), k * angle.cos()], dim=1)

    assert geometric_drift(z, faster, order=2)["curv"] == pytest.approx(1.0, rel=1e-6)
    assert geometric_drift(z, -faster, order=2)["curv"] == pytest.approx(-1.0, rel=1e-6)


def test_drift_sign_convention_is_negative_for_a_regularising_flow(rng):
    """Negative drift must mean "the flow is making this statistic smaller"."""
    z = _randn(T, D, generator=rng)
    for name in STATISTICS:
        z_grad = z.clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(geophys_statistics(z_grad)[name], z_grad)
        # Descending the statistic's own gradient must register as negative drift.
        assert geometric_drift(z, -gradient)[name] < 0
        assert geometric_drift(z, gradient)[name] > 0


def test_curvature_has_a_kink_minimum_at_a_perfectly_straight_trajectory(rng):
    """A straight trajectory minimises curvature, but non-smoothly.

    Near zero the turning angle grows like ``|perturbation|``, not quadratically, so
    ``phi_curv`` has a kink rather than a smooth minimum. Its directional derivative is
    therefore one-sided and strictly positive in *every* direction, and the value
    autograd returns exactly at the kink is an artefact of the epsilon guard rather
    than a meaningful slope.

    This is worth pinning because it is the one place the drift metric is not
    trustworthy, and a straight feature trajectory is a real input -- a static scene
    produces one.
    """
    direction = _randn(D, generator=rng)
    straight = torch.arange(T, dtype=torch.float64).unsqueeze(1) * direction
    baseline = float(geophys_statistics(straight)["curv"])

    for _ in range(3):
        u = _randn(T, D, generator=rng)
        h = 1e-4
        forward = float(geophys_statistics(straight + h * u)["curv"])
        backward = float(geophys_statistics(straight - h * u)["curv"])

        # Both directions bend it: a minimum, so no signed slope exists.
        assert forward > baseline and backward > baseline
        # The central difference cancels to ~0 by symmetry, confirming the kink.
        assert abs(forward - backward) / (2 * h) < 1e-3 * (forward - baseline) / h

        # Autograd at the kink is negligible next to the genuine one-sided slope.
        assert abs(geometric_drift(straight, u)["curv"]) < 1e-3 * (forward - baseline) / h


# -- transport alignment and erosion ---------------------------------------


def test_transport_alignment_endpoints(rng):
    z = _randn(T, D, generator=rng)
    assert transport_alignment(z, z)[0] == pytest.approx(1.0, abs=1e-9)
    assert transport_alignment(z, -z)[0] == pytest.approx(-1.0, abs=1e-9)


def test_transport_alignment_weighting_favours_moving_frames():
    """A near-static frame with an adversarial cosine must not outvote a moving one."""
    z = torch.zeros(4, 2, dtype=torch.float64)
    z[1] = torch.tensor([1.0, 0.0])       # large displacement
    z[2] = torch.tensor([1.0, 0.0])       # zero displacement
    z[3] = torch.tensor([1.000001, 0.0])  # negligible displacement

    flow = torch.zeros_like(z)
    flow[1] = torch.tensor([1.0, 0.0])    # aligned with the large displacement
    flow[3] = torch.tensor([-1.0, 0.0])   # opposed, but on a negligible one

    plain, weighted = transport_alignment(z, flow)
    assert weighted > plain


def test_erosion_rate_is_the_ratio_of_motion_change_to_motion(rng):
    z = _randn(T, D, generator=rng)
    assert erosion_rate(z, 2 * z) == pytest.approx(2.0, abs=1e-9)
    assert erosion_rate(z, torch.zeros_like(z)) == pytest.approx(0.0, abs=1e-9)


def test_guidance_alignment_endpoints(rng):
    g = _randn(T, D, generator=rng)
    assert guidance_alignment(g, g) == pytest.approx(1.0, abs=1e-6)
    assert guidance_alignment(g, -g) == pytest.approx(-1.0, abs=1e-6)


# -- the empirical estimator ------------------------------------------------


def test_empirical_drift_converges_to_the_exact_one(rng):
    """The hidden-state path uses secants; they must agree in the small-step limit."""
    z, u = _randn(T, D, generator=rng), _randn(T, D, generator=rng)
    h = 1e-6
    empirical = geometric_drift_empirical(z, z + h * u, h)
    exact = geometric_drift(z, u)
    for name in STATISTICS:
        assert empirical[name] == pytest.approx(exact[name], rel=1e-4, abs=1e-8)


def test_empirical_drift_rejects_a_zero_step():
    z = torch.randn(T, D, dtype=torch.float64)
    with pytest.raises(ValueError, match="delta_tau"):
        geometric_drift_empirical(z, z, 0.0)


# -- plumbing ---------------------------------------------------------------


def test_flow_coupling_bundle_is_tagged_exact(rng):
    z, u = _randn(T, D, generator=rng), _randn(T, D, generator=rng)
    coupling = flow_coupling(z, u, tau=0.5)
    assert coupling.estimator is DriftEstimator.EXACT
    assert coupling.tau == 0.5
    assert set(coupling.drift) == set(STATISTICS)
    assert coupling.alignment is not None and coupling.erosion is not None


def test_mismatched_shapes_raise_rather_than_broadcasting(rng):
    with pytest.raises(ValueError, match="same shape"):
        geometric_drift(_randn(T, D, generator=rng), _randn(T, D + 1, generator=rng))


def test_drift_does_not_leak_gradient_state_into_its_input(rng):
    """geometric_drift takes gradients internally; the caller's tensor must be untouched."""
    z = _randn(T, D, generator=rng)
    u = _randn(T, D, generator=rng)
    geometric_drift(z, u)
    assert not z.requires_grad and z.grad is None
