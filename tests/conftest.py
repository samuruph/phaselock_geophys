"""Shared fixtures.

The oracle backend below is the workhorse: it satisfies the full
:class:`~phaselock.backends.base.VideoBackend` contract on CPU with no weights, and its
denoiser is an exact oracle for a known clean latent. That makes the inversion loop
analytically checkable -- inverting and resampling must return the original latent to
machine precision, so any error in the schedule direction, the renoise formula or the
canonical-layout conversions shows up as a hard failure rather than a slightly worse
reconstruction.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
import torch

from phaselock.backends.base import (
    DenoiserState,
    LatentSpec,
    VideoBackend,
    to_canonical,
)

NUM_TRAIN_TIMESTEPS = 1000


class _Scheduler:
    """Minimal flow-matching schedule: sigma = timestep / num_train_timesteps."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"num_train_timesteps": NUM_TRAIN_TIMESTEPS})()
        self.timesteps = torch.tensor([])

    def set_timesteps(self, num_steps: int, device=None) -> None:
        # Descending from near-noise to data, as every diffusers scheduler does.
        self.timesteps = torch.linspace(
            NUM_TRAIN_TIMESTEPS - 1, 0, num_steps, device=device, dtype=torch.float32
        )


class _Block(torch.nn.Module):
    """Identity block that still produces a hookable token tensor."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(dim), requires_grad=False)
        self.attn1 = torch.nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = hidden_states * self.scale
        self.attn1(out)  # give the attention source something to hook
        return out


class _Transformer(torch.nn.Module):
    def __init__(self, num_blocks: int, dim: int, grid: tuple[int, int, int]):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block(dim) for _ in range(num_blocks))
        self.grid = grid
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class _Pipe:
    def __init__(self, transformer: _Transformer):
        self.transformer = transformer
        self.scheduler = _Scheduler()
        self.vae = None


class OracleBackend(VideoBackend):
    """A backend whose denoiser knows the true clean latent exactly.

    Given ``z`` at noise level ``sigma`` on the rectified path
    ``z = (1 - sigma) x0* + sigma eps``, it recovers ``x0*`` exactly, so a full
    invert/resample round trip is the identity up to floating point.
    """

    def __init__(self, spec: LatentSpec, target: torch.Tensor, num_blocks: int = 6, dim: int = 8):
        grid = (target.shape[0], target.shape[2] // spec.patch_size[1], target.shape[3] // spec.patch_size[2])
        super().__init__(_Pipe(_Transformer(num_blocks, dim, grid)), spec, device=torch.device("cpu"))
        self.target = target
        self.grid = grid
        self.dim = dim
        self.forward_calls = 0

    @property
    def blocks(self) -> torch.nn.ModuleList:
        return self.pipe.transformer.blocks

    @staticmethod
    def block_hidden_states(block_output: Any) -> torch.Tensor:
        return block_output

    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _sigma(self, timestep) -> float:
        value = timestep.flatten()[0].item() if torch.is_tensor(timestep) else timestep
        return float(value) / NUM_TRAIN_TIMESTEPS

    def denoiser_state(self, latents, model_output, timestep) -> DenoiserState:
        z = to_canonical(latents, self.spec).float()
        v = to_canonical(model_output, self.spec).float()
        sigma = self._sigma(timestep)
        return DenoiserState(
            latents=z, x0=z - sigma * v, eps=z + (1.0 - sigma) * v, tau=1.0 - sigma
        )

    def renoise(self, x0, eps, timestep) -> torch.Tensor:
        sigma = self._sigma(timestep)
        return (1.0 - sigma) * x0 + sigma * eps

    def prepare_conditioning(self, prompt: str = "", negative_prompt=None, **kwargs) -> dict:
        return {"prompt": prompt}

    def transformer_forward(self, latents, timestep, conditioning) -> torch.Tensor:
        """Return the velocity that transports ``latents`` exactly onto the target.

        Also runs the token stack so that block hooks fire, which is what the recorder
        tests exercise.
        """
        self.forward_calls += 1
        z = to_canonical(latents, self.spec).float()
        sigma = self._sigma(timestep)

        frames, height, width = self.grid
        tokens = torch.zeros(1, frames * height * width, self.dim)
        tokens[0, :, 0] = z.mean(dim=(1, 2, 3)).repeat_interleave(height * width)
        self.pipe.transformer(tokens)

        if sigma < 1e-9:
            velocity = torch.zeros_like(z)
        else:
            eps = (z - (1.0 - sigma) * self.target) / sigma
            velocity = eps - self.target
        from phaselock.backends.base import from_canonical

        return from_canonical(velocity, self.spec)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed the global RNG before every test.

    Several tests use bare ``torch.randn``, whose values would otherwise depend on how
    many tests ran before them -- so adding a test file elsewhere could flip an unrelated
    tolerance check from passing to failing.
    """
    torch.manual_seed(0)


@pytest.fixture
def latent_spec() -> LatentSpec:
    return LatentSpec(
        name="oracle",
        layout="BCTHW",
        temporal_ratio=4,
        spatial_ratio=8,
        channels=4,
        patch_size=(1, 2, 2),
        default_num_frames=17,
        default_fps=8,
        default_height=64,
        default_width=64,
        scaling_factor=1.0,
    )


@pytest.fixture
def target_latent(latent_spec) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(5, latent_spec.channels, 4, 4, generator=generator)


@pytest.fixture
def oracle(latent_spec, target_latent) -> OracleBackend:
    return OracleBackend(latent_spec, target_latent)
