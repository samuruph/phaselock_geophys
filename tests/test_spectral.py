"""PhaseLock's spectral metrics (Fig. 3a). CPU only."""

from __future__ import annotations

import pytest




def test_phase_correlation_restricts_to_the_low_frequency_band():
    """The mask its sibling metrics use must be applied here too.

    Regression: phase_difference_correlation correlated the full spectrum. Phase at high
    spatial frequency is near-uniform noise -- at 480x720 three quarters of the bins are
    out of band -- so a real sweep returned 2e-4 in every (K, sigma) cell.

    This pins that the band is applied and that the metric is calibrated at its
    endpoints. It deliberately does not assert a magnitude on real-ish input: see
    `test_phase_correlation_is_fragile_to_low_magnitude_bins` for why that is still open.
    """
    import torch

    from phaselock.metrics.spectral import phase_difference_correlation

    torch.manual_seed(0)
    base = torch.rand(9, 3, 64, 64)

    assert phase_difference_correlation(base, base) == pytest.approx(1.0, abs=1e-9)
    assert abs(phase_difference_correlation(base, torch.rand(9, 3, 64, 64))) < 0.05

    # Widening the cutoff past the Nyquist radius selects every bin, so it must change
    # the answer -- if it does not, the mask is not being applied at all.
    narrow = phase_difference_correlation(base, base * 0.5 + 0.25, cutoff=0.2)
    wide = phase_difference_correlation(base, base * 0.5 + 0.25, cutoff=10.0)
    assert narrow != wide


def test_phase_correlation_is_fragile_to_low_magnitude_bins():
    """A known limitation, pinned so it is not mistaken for a passing metric.

    Phase is meaningless where the magnitude is near zero, and an unweighted Pearson over
    phase treats those bins equally with the ones carrying the motion. A smooth blob
    perturbed by mild noise therefore scores near zero even in band, and the sweep does
    not reproduce PhaseLock's reported 0.358 vs 0.100. Masking was necessary and is not
    sufficient; magnitude weighting is the obvious next step and is not implemented.
    """
    import torch

    from phaselock.metrics.spectral import phase_difference_correlation

    torch.manual_seed(0)
    grid = torch.arange(64, dtype=torch.float64)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    blob = torch.stack([
        torch.exp(-(((x - 20 - 2 * f) ** 2 + (y - 32) ** 2) / 200.0)) for f in range(9)
    ]).unsqueeze(1).repeat(1, 3, 1, 1).float()

    perturbed = (blob + 0.05 * torch.randn_like(blob)).clamp(0, 1)
    # Same motion, mild noise -- and the metric still cannot see it.
    assert abs(phase_difference_correlation(blob, perturbed)) < 0.1
