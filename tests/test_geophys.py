"""CPU tests for the five GeoPhys statistics. No GPU, no model weights.

Each statistic is pinned against a trajectory whose geometry is known in closed form,
so a regression shows up as a wrong number rather than a plausible-looking one.
"""

from __future__ import annotations

import math

import pytest
import torch

from phaselock.metrics.geophys import (
    STATISTICS,
    accelerations,
    displacements,
    geophys_signals,
    geophys_statistics,
    prediction_residuals,
    speeds,
    turning_angles,
)

T, D = 13, 48


def straight_line(steps: int = T, dim: int = D, seed: int = 0) -> torch.Tensor:
    """Constant-velocity motion: every statistic is exactly zero."""
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=generator, dtype=torch.float64)
    return torch.arange(steps, dtype=torch.float64).unsqueeze(1) * direction


def circle(rate: float, steps: int = T, radius: float = 1.0) -> torch.Tensor:
    """Uniform circular motion: turning angle is exactly ``rate`` at every frame."""
    angle = torch.arange(steps, dtype=torch.float64) * rate
    return radius * torch.stack([angle.cos(), angle.sin()], dim=1)


# -- shapes -----------------------------------------------------------------


def test_intermediate_shapes_follow_the_difference_orders():
    z = torch.randn(T, D, dtype=torch.float64)
    assert displacements(z).shape == (T - 1, D)
    assert speeds(z).shape == (T - 1,)
    assert turning_angles(z).shape == (T - 2,)
    assert accelerations(z).shape == (T - 2, D)
    assert prediction_residuals(z, order=3).shape == (T - 3,)


def test_short_trajectories_raise_rather_than_returning_empty():
    for length, name in [(1, "displacement"), (2, "turning angle")]:
        with pytest.raises(ValueError, match="at least"):
            geophys_signals(torch.randn(length, D, dtype=torch.float64))
    with pytest.raises(ValueError, match="at least"):
        prediction_residuals(torch.randn(3, D, dtype=torch.float64), order=3)


# -- analytic ground truth --------------------------------------------------


def test_straight_line_gives_zero_for_every_statistic():
    stats = geophys_statistics(straight_line())
    for name in STATISTICS:
        assert abs(float(stats[name])) < 1e-9, f"{name} should vanish on a straight line"


def test_circle_has_constant_turning_angle_and_constant_speed():
    rate = 0.35
    signals = geophys_signals(circle(rate), order=2)

    # Every chord of a uniformly sampled circle subtends the same turning angle.
    assert float(signals.statistics["curv"]) == pytest.approx(rate, abs=1e-9)
    assert float(signals.statistics["ang"]) < 1e-9
    assert float(signals.statistics["speed"]) < 1e-9
    assert torch.allclose(signals.turning_angle, torch.full_like(signals.turning_angle, rate))


def test_turning_angle_is_stable_at_both_endpoints_of_its_range():
    """arccos loses half its precision near 0 and pi; the half-angle form does not."""
    # Straight: angle 0.
    assert float(turning_angles(straight_line()).abs().max()) < 1e-9

    # Doubling straight back on itself: angle exactly pi.
    direction = torch.zeros(2, dtype=torch.float64)
    direction[0] = 1.0
    out_and_back = torch.stack(
        [direction * s for s in [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]]
    )
    assert float(turning_angles(out_and_back).max()) == pytest.approx(math.pi, abs=1e-9)


def test_teleport_spikes_acceleration_and_residual_at_the_injected_frame():
    z = straight_line()
    frame = 7
    z[frame] += 25.0

    signals = geophys_signals(z, order=3)

    # a_t = z_{t+2} - 2 z_{t+1} + z_t, so the -2 coefficient makes t = frame - 1 largest.
    assert int(signals.acceleration.argmax()) == frame - 1
    # Residual index i predicts frame i + order.
    assert int(signals.residual.argmax()) == frame - 3


def test_accel_is_a_squared_norm_and_perr_is_not():
    """The two summaries use different powers; swapping them is an easy silent bug."""
    z = torch.randn(T, D, dtype=torch.float64)
    signals = geophys_signals(z)
    assert float(signals.statistics["accel"]) == pytest.approx(
        float(signals.acceleration.pow(2).mean())
    )
    assert float(signals.statistics["perr"]) == pytest.approx(float(signals.residual.mean()))


def test_statistics_are_scale_equivariant_in_the_documented_way():
    """Angles are scale free; magnitudes carry their natural power of the scale."""
    z = torch.randn(T, D, dtype=torch.float64)
    scale = 7.0
    base, scaled = geophys_statistics(z), geophys_statistics(z * scale)

    assert float(scaled["curv"]) == pytest.approx(float(base["curv"]))
    assert float(scaled["ang"]) == pytest.approx(float(base["ang"]))
    assert float(scaled["speed"]) == pytest.approx(scale * float(base["speed"]))
    assert float(scaled["perr"]) == pytest.approx(scale * float(base["perr"]))
    assert float(scaled["accel"]) == pytest.approx(scale**2 * float(base["accel"]))


# -- the residual-fit ambiguity ---------------------------------------------


def test_naive_global_ols_residual_is_degenerate():
    """Why the default is "span" rather than the paper's literal wording.

    Fitting P_H : R^{H*D} -> R^D on a single video's windows is underdetermined
    whenever H*D exceeds the window count, which it always does here, so the in-sample
    residual collapses to zero and carries no signal at all.
    """
    z = torch.randn(T, D, dtype=torch.float64)
    order = 3
    past = torch.stack([z[i : i + order].flatten() for i in range(T - order)])
    target = z[order:]
    assert past.shape[1] > past.shape[0], "the degenerate regime this test documents"

    predictor = torch.linalg.lstsq(past, target).solution
    assert float((target - past @ predictor).norm(dim=-1).mean()) < 1e-8

    # The implemented fits all stay well away from zero on the same data.
    for fit in ("span", "ridge", "scalar"):
        assert float(prediction_residuals(z, order=order, fit=fit).mean()) > 1e-3


def test_ridge_penalty_is_scale_relative():
    """An absolute lambda would collapse to the degenerate OLS at large feature scale."""
    z = torch.randn(T, D, dtype=torch.float64)
    reference = float(prediction_residuals(z, fit="ridge").mean())
    for scale in (0.01, 100.0):
        scaled = float(prediction_residuals(z * scale, fit="ridge").mean()) / scale
        assert scaled == pytest.approx(reference, rel=1e-6)


def test_span_residual_vanishes_when_the_next_frame_lies_in_the_span():
    """A trajectory confined to a plane is exactly predicted by a 3-frame affine span."""
    steps = torch.arange(T, dtype=torch.float64)
    planar = torch.zeros(T, D, dtype=torch.float64)
    planar[:, 0] = steps
    planar[:, 1] = steps**2  # quadratic: still inside the affine span of 3 points
    assert float(prediction_residuals(planar, order=3, fit="span").max()) < 1e-8


def test_unknown_residual_fit_raises():
    with pytest.raises(ValueError, match="unknown residual fit"):
        prediction_residuals(torch.randn(T, D, dtype=torch.float64), fit="nonsense")


# -- differentiability, which flow_geometry depends on ----------------------


@pytest.mark.parametrize("name", STATISTICS)
@pytest.mark.parametrize("kind", ["random", "straight", "stationary"])
def test_gradients_are_finite_including_at_degenerate_trajectories(name, kind):
    """A stationary or perfectly straight clip is a real input, not a corner case."""
    trajectories = {
        "random": torch.randn(T, D, dtype=torch.float64),
        "straight": straight_line(),
        "stationary": torch.zeros(T, D, dtype=torch.float64),
    }
    z = trajectories[kind].clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(geophys_statistics(z)[name], z)
    assert torch.isfinite(gradient).all()


def test_float64_input_is_not_silently_downcast():
    """Finite-difference checks in flow_geometry need the precision preserved."""
    z = torch.randn(T, D, dtype=torch.float64)
    assert geophys_statistics(z)["speed"].dtype == torch.float64
    assert geophys_statistics(z.to(torch.bfloat16))["speed"].dtype == torch.float32


def test_energy_momentum_and_jerk_vanish_on_a_straight_line():
    """The same analytic anchor the other five have: perfectly regular motion scores 0."""
    import torch

    from phaselock.metrics.geophys import geophys_statistics

    straight = torch.arange(12, dtype=torch.float32).unsqueeze(1) * torch.ones(1, 6)
    stats = geophys_statistics(straight)
    for name in ("energy", "momentum", "jerk"):
        assert float(stats[name]) == pytest.approx(0.0, abs=1e-5), name


def test_momentum_is_one_for_a_round_trip():
    """Out and back: every step moved, none of it went anywhere."""
    import torch

    from phaselock.metrics.geophys import geophys_statistics

    out = torch.arange(6, dtype=torch.float32).unsqueeze(1) * torch.ones(1, 4)
    trip = torch.cat([out, out.flip(0)[1:]])
    assert float(geophys_statistics(trip)["momentum"]) == pytest.approx(1.0, abs=1e-4)


def test_jerk_spikes_on_an_impulse_but_not_on_constant_acceleration():
    """Jerk is the change of force, so a steady force must not trigger it."""
    import torch

    from phaselock.metrics.geophys import geophys_statistics

    t = torch.arange(12, dtype=torch.float32).unsqueeze(1)
    parabola = (t**2) * torch.ones(1, 4)          # constant acceleration
    impulse = (t * torch.ones(1, 4)).clone()
    impulse[6:] += 20.0                            # a teleport

    assert float(geophys_statistics(parabola)["jerk"]) == pytest.approx(0.0, abs=1e-3)
    assert float(geophys_statistics(impulse)["jerk"]) > 100.0


def test_energy_is_scale_free():
    """A coefficient of variation, so a VAE latent and a hidden state are comparable
    despite feature magnitudes differing by orders of magnitude."""
    import torch

    from phaselock.metrics.geophys import geophys_statistics

    torch.manual_seed(0)
    z = torch.randn(14, 8).cumsum(0)
    assert float(geophys_statistics(z)["energy"]) == pytest.approx(
        float(geophys_statistics(z * 1000)["energy"]), rel=1e-4
    )

