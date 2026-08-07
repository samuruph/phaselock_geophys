"""Spectral metrics from PhaseLock, for the blur control and the erosion analysis.

Two distinct measurements, easily conflated:

**Inter-frame phase-difference correlation** (paper Fig. 3a, the blur control). Take a
frame-wise 2D FFT, form the phase difference between consecutive frames, and correlate
the generated map against the ground-truth one with Pearson's r. Under Gaussian blur of
sigma in {0, 8, 16} applied to *every* arm, a 2-step output held 0.358 against a 50-step
output's 0.100 -- evidence the 2-step advantage is structural rather than an artefact of
blurriness.

**Phase coherence and magnitude correlation** (paper section 3.2). Take a 3D
spatio-temporal FFT of the video or latent, restrict to the low-frequency band
(normalised radius < 0.4), and compare against ground truth. Phase coherence fell ~18%
between step 2 and step 50 while magnitude correlation fell only 2-3%, which is the
asymmetry the whole method rests on.
"""

from __future__ import annotations

from typing import Optional

import torch

_EPS = 1e-12

# Rec. 709 luminance weights.
_LUMA = (0.2126, 0.7152, 0.0722)


def to_luminance(frames: torch.Tensor) -> torch.Tensor:
    """``(F, C, H, W)`` -> ``(F, H, W)``. Colour channels are already greyscale-safe."""
    if frames.ndim != 4:
        raise ValueError(f"expected (F, C, H, W) frames, got {tuple(frames.shape)}")
    if frames.shape[1] == 1:
        return frames[:, 0]
    weights = torch.tensor(_LUMA, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    return (frames[:, :3] * weights).sum(dim=1)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened."""
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    if a.numel() != b.numel():
        raise ValueError(f"length mismatch: {a.numel()} vs {b.numel()}")
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    if float(denominator) < _EPS:
        return 0.0
    return float((a * b).sum() / denominator)


def inter_frame_phase_difference(frames: torch.Tensor) -> torch.Tensor:
    """Wrapped phase difference between consecutive frames, ``(F-1, H, W)``.

    Computed as ``angle(Z_{f+1} * conj(Z_f))`` rather than by subtracting two angles.
    The two are equal modulo 2*pi, but this form wraps into ``(-pi, pi]`` automatically
    and avoids the spurious 2*pi jumps that plague a naive difference of ``angle``
    values -- which would otherwise dominate the correlation.
    """
    spectrum = torch.fft.fft2(to_luminance(frames).to(torch.float64))
    return torch.angle(spectrum[1:] * spectrum[:-1].conj())


def phase_difference_correlation(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """PhaseLock Fig. 3a: Pearson r between generated and reference phase-difference maps.

    Both arguments are ``(F, C, H, W)`` in [0, 1] and must already be temporally aligned
    and, for the blur control, blurred by the same sigma.
    """
    if generated.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {tuple(generated.shape)} vs {tuple(reference.shape)}. "
            "Resample and letterbox both clips before comparing."
        )
    return pearson(inter_frame_phase_difference(generated), inter_frame_phase_difference(reference))


def low_frequency_mask(
    shape: tuple[int, ...], cutoff: float = 0.4, device=None
) -> torch.Tensor:
    """Boolean mask selecting normalised radii below ``cutoff`` in an N-D FFT grid.

    Each axis is normalised to [-1, 1] by its own Nyquist frequency before the radius is
    taken, so a non-cubic volume -- 13 latent frames against a 60x90 grid -- is not
    dominated by whichever axis happens to be longest.
    """
    axes = []
    for length in shape:
        frequency = torch.fft.fftfreq(length, device=device).to(torch.float64)
        axes.append(frequency / (0.5 if length > 1 else 1.0))
    grid = torch.meshgrid(*axes, indexing="ij")
    radius = torch.sqrt(sum(axis**2 for axis in grid) / len(shape))
    return radius < cutoff


def _spatiotemporal_spectrum(video: torch.Tensor) -> torch.Tensor:
    """3D FFT over ``(T, H, W)``, accepting ``(T, C, H, W)`` by averaging channels."""
    if video.ndim == 4:
        video = video.mean(dim=1)
    if video.ndim != 3:
        raise ValueError(f"expected (T, H, W) or (T, C, H, W), got {tuple(video.shape)}")
    return torch.fft.fftn(video.to(torch.float64), dim=(0, 1, 2))


def phase_coherence(
    generated: torch.Tensor, reference: torch.Tensor, cutoff: float = 0.4
) -> float:
    """Mean cosine similarity between generated and reference phase angles.

    Restricted to the low-frequency band, where coarse motion lives and where the 2-step
    output has enough energy to be meaningful. Returns a value in [-1, 1].
    """
    generated_spectrum = _spatiotemporal_spectrum(generated)
    reference_spectrum = _spatiotemporal_spectrum(reference)
    if generated_spectrum.shape != reference_spectrum.shape:
        raise ValueError("generated and reference must have matching shapes")

    mask = low_frequency_mask(generated_spectrum.shape, cutoff, generated_spectrum.device)
    delta = torch.angle(generated_spectrum[mask]) - torch.angle(reference_spectrum[mask])
    return float(torch.cos(delta).mean())


def magnitude_correlation(
    generated: torch.Tensor, reference: torch.Tensor, cutoff: float = 0.4
) -> float:
    """Pearson correlation of log-magnitudes over the low-frequency band.

    Logs rather than raw magnitudes: an FFT magnitude spectrum spans many orders of
    magnitude, and a linear correlation would be decided almost entirely by the DC term.
    """
    generated_spectrum = _spatiotemporal_spectrum(generated)
    reference_spectrum = _spatiotemporal_spectrum(reference)
    mask = low_frequency_mask(generated_spectrum.shape, cutoff, generated_spectrum.device)
    return pearson(
        torch.log(generated_spectrum[mask].abs() + _EPS),
        torch.log(reference_spectrum[mask].abs() + _EPS),
    )


def swap_phase_magnitude(
    phase_source: torch.Tensor, magnitude_source: torch.Tensor
) -> torch.Tensor:
    """Recombine one video's phase with another's magnitude.

    Supports the paper's causal argument that motion lives in the phase spectrum.
    """
    phase = torch.angle(_spatiotemporal_spectrum(phase_source))
    magnitude = _spatiotemporal_spectrum(magnitude_source).abs()
    return torch.fft.ifftn(magnitude * torch.exp(1j * phase), dim=(0, 1, 2)).real


def corrupt_spectrum(
    video: torch.Tensor,
    component: str = "phase",
    fraction: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Inject uniform noise into either the phase or the magnitude spectrum.

    The paper's controlled corruption experiment: 50% noise into the phase produced a
    RAFT end-point error of 9.74 pixels against 1.14 for equivalent magnitude
    corruption, an 8.5x disparity, which is its causal evidence that motion depends on
    phase.

    Args:
        component: ``"phase"`` or ``"magnitude"``.
        fraction: Interpolation weight toward the noise, in [0, 1].
    """
    if component not in ("phase", "magnitude"):
        raise ValueError(f"component must be 'phase' or 'magnitude', got {component!r}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    spectrum = _spatiotemporal_spectrum(video)
    phase = torch.angle(spectrum)
    magnitude = spectrum.abs()

    generator = torch.Generator(device=spectrum.device).manual_seed(seed)
    if component == "phase":
        import math

        noise = (
            torch.rand(phase.shape, generator=generator, device=spectrum.device, dtype=torch.float64)
            * 2
            * math.pi
            - math.pi
        )
        phase = (1 - fraction) * phase + fraction * noise
    else:
        noise = torch.rand(
            magnitude.shape, generator=generator, device=spectrum.device, dtype=torch.float64
        ) * magnitude.mean()
        magnitude = (1 - fraction) * magnitude + fraction * noise

    return torch.fft.ifftn(magnitude * torch.exp(1j * phase), dim=(0, 1, 2)).real
