"""Recovering a real video's latent trajectory by running the sampler backwards.

Diffusion models expose no latent trajectory for a video they did not generate, so the
internal states associated with a *real* clip -- the only ones with a known plausibility
label -- are never produced. "The Invisible Hand of Physics" recovers them by integrating
the learned velocity field backward from the clean latent to noise.

The exact reverse of an Euler step is implicit and would need an iterative solver at
every step. Following the paper, we use the explicit approximation: evaluate the denoiser
at the known endpoint and step from there, which costs one network evaluation per step,
the same as forward sampling.

Both supported backends reduce to the same two operations -- estimate ``(x0, eps)`` at
the current point, then place that pair at the next noise level -- so one loop serves the
VP (DDIM) and rectified-flow parameterisations alike.

The paper's warning is worth repeating: reconstruction quality and trajectory fidelity
come apart. At 20 steps the recovered noise still reconstructs the video with only mild
quality loss while probe accuracy collapses from 0.82 to 0.57. A good reconstruction is
necessary but nowhere near sufficient, which is why the step count is swept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch

from ..backends.base import VideoBackend, from_canonical, to_canonical, token_grid
from ..probes.pooling import PoolMode
from ..probes.recorder import ProbeRecorder
from ..probes.trajectory import ProbeRecord


def evenly_spaced(total: int, count: int) -> list[int]:
    """``count`` indices spread over ``range(total)``, including both ends."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count >= total:
        return list(range(total))
    if count == 1:
        return [total // 2]
    return sorted({int(round(i * (total - 1) / (count - 1))) for i in range(count)})


@dataclass
class InversionResult:
    """What an inversion produced."""

    noise: torch.Tensor
    """The recovered ``z_0``, canonical ``(T, C, h, w)``."""

    record: ProbeRecord
    timesteps: list[float]


def _schedule(backend: VideoBackend, num_steps: int, device: torch.device) -> torch.Tensor:
    """The sampler's own timestep schedule, descending from noise to data."""
    scheduler = backend.pipe.scheduler
    scheduler.set_timesteps(num_steps, device=device)
    return scheduler.timesteps


@torch.no_grad()
def invert(
    backend: VideoBackend,
    frames: Optional[torch.Tensor] = None,
    *,
    latents: Optional[torch.Tensor] = None,
    num_steps: int = 50,
    record_steps: int = 10,
    sources: Sequence[str] = ("hidden_states", "latent", "x0_hat", "velocity"),
    blocks: Optional[Iterable[int]] = None,
    block_stride: int = 1,
    pooling: PoolMode = "mean",
    prompt: str = "",
    conditioning: Optional[dict[str, Any]] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> InversionResult:
    """Invert a real video and record the internal states along the way.

    Args:
        frames: ``(F, 3, H, W)`` in [0, 1], already resampled and letterboxed to the
            backend's native geometry. Mutually exclusive with ``latents``.
        latents: Pre-encoded canonical ``(T, C, h, w)`` latents, for callers that have
            already run the VAE or want to skip it.
        num_steps: Integration steps. Fidelity depends on this far more strongly than
            reconstruction quality suggests.
        record_steps: How many of those steps to probe, spread evenly. Recording costs
            storage, not compute.
        prompt: Empty by default, with classifier-free guidance off, so the recovered
            trajectory is the unconditional velocity field and no text confound enters
            a plausible-versus-violated comparison.
    """
    if (frames is None) == (latents is None):
        raise ValueError("pass exactly one of `frames` or `latents`")

    if latents is None:
        latents = backend.encode(frames)
        num_frames, height, width = frames.shape[0], frames.shape[2], frames.shape[3]
    else:
        spec = backend.spec
        num_frames = (latents.shape[0] - 1) * spec.temporal_ratio + 1
        height = latents.shape[2] * spec.spatial_ratio
        width = latents.shape[3] * spec.spatial_ratio

    grid = token_grid(latents.shape, backend.spec)
    device = backend.device

    if conditioning is None:
        conditioning = backend.prepare_conditioning(
            prompt=prompt, num_frames=num_frames, height=height, width=width
        )

    # Sampling runs high noise -> low; inversion walks the same ladder upward.
    timesteps = _schedule(backend, num_steps, device)
    ascending = list(reversed(timesteps.tolist()))
    to_record = set(evenly_spaced(len(ascending), record_steps))

    z = latents.to(device)
    recorded: list[float] = []

    with ProbeRecorder(
        backend, sources=sources, grid=grid, blocks=blocks, block_stride=block_stride, pooling=pooling
    ) as probe:
        probe.set_provenance(
            direction="inversion",
            num_steps=num_steps,
            record_steps=sorted(to_record),
            prompt=prompt,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            **(provenance or {}),
        )

        # z starts clean, so the level it currently sits at leads the schedule by one.
        # Evaluating at level s and renoising back to s is the *identity* -- (x0, eps)
        # were derived from z at s, so recombining them at s reconstructs z exactly. The
        # step only advances if it targets the next level up.
        levels = [0.0] + ascending

        for index in range(len(ascending)):
            wanted = index in to_record
            # Hooks fire on every forward pass; arming decides whether they do any work.
            probe.arm() if wanted else probe.disarm()

            current = torch.tensor(levels[index], device=device)
            model_output = backend.transformer_forward(
                from_canonical(z, backend.spec), current, conditioning
            )
            state = backend.denoiser_state(
                from_canonical(z, backend.spec), model_output, current
            )

            if wanted:
                probe.capture(len(recorded), state)
                recorded.append(float(levels[index]))

            # Explicit step: evaluate at the known endpoint, then move up one level.
            target = torch.tensor(levels[index + 1], device=device)
            z = to_canonical(backend.renoise(state.x0, state.eps, target), backend.spec)

        record = probe.record

    return InversionResult(noise=z, record=record, timesteps=recorded)


@torch.no_grad()
def resample(
    backend: VideoBackend,
    noise: torch.Tensor,
    *,
    num_steps: int = 50,
    prompt: str = "",
    conditioning: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Run the forward ODE from noise back to a clean latent.

    Used to check inversion fidelity: invert a clip, resample the recovered noise, and
    compare against the source. Passing this check is necessary but not sufficient --
    see the module docstring.
    """
    device = backend.device
    z = noise.to(device)

    if conditioning is None:
        conditioning = backend.prepare_conditioning(prompt=prompt)

    timesteps = _schedule(backend, num_steps, device)
    schedule = timesteps.tolist()

    for index, timestep in enumerate(schedule):
        tensor_timestep = torch.tensor(timestep, device=device)
        model_output = backend.transformer_forward(
            from_canonical(z, backend.spec), tensor_timestep, conditioning
        )
        state = backend.denoiser_state(
            from_canonical(z, backend.spec), model_output, tensor_timestep
        )
        # The final step lands on the clean latent itself.
        target = schedule[index + 1] if index + 1 < len(schedule) else 0.0
        z = to_canonical(
            backend.renoise(state.x0, state.eps, torch.tensor(target, device=device)),
            backend.spec,
        )
    return z
