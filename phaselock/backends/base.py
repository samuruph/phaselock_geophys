"""Backend abstraction over video diffusion pipelines.

Different video diffusion models disagree about almost everything that matters
here: CogVideoX stores latents as ``(B, T, C, H, W)`` while Wan uses
``(B, C, T, H, W)``; CogVideoX normalises latents with a scalar
``scaling_factor`` while Wan uses per-channel ``latents_mean``/``latents_std``;
CogVideoX predicts ``v`` under a VP schedule while Wan predicts a flow velocity.

Everything downstream of this module works in a single canonical form --
``(T, C, H, W)``, normalised, unbatched -- so that guidance, probing and the
geometric metrics have exactly one implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

LAYOUTS = ("BTCHW", "BCTHW")


@dataclass(frozen=True)
class LatentSpec:
    """Static description of a backend's latent space and DiT tokenisation.

    Args:
        name: Registry key for this spec.
        layout: Axis order the pipeline uses for batched latents. One of ``LAYOUTS``.
        temporal_ratio: VAE temporal downsampling factor.
        spatial_ratio: VAE spatial downsampling factor.
        channels: Number of latent channels.
        patch_size: DiT patch as ``(t, h, w)``. Both supported backends use
            ``(1, 2, 2)``, which is what makes per-latent-frame token pooling exact.
        default_num_frames: Native frame count (must satisfy ``k * temporal_ratio + 1``).
        default_fps: Native frame rate.
        default_height / default_width: Native generation resolution.
        scaling_factor: Scalar latent normalisation. Mutually exclusive with
            ``latents_mean``/``latents_std``.
        latents_mean / latents_std: Per-channel latent normalisation.
        flow_shift: Scheduler shift, where the backend exposes one.
    """

    name: str
    layout: str
    temporal_ratio: int
    spatial_ratio: int
    channels: int
    patch_size: tuple[int, int, int]
    default_num_frames: int
    default_fps: int
    default_height: int
    default_width: int
    scaling_factor: Optional[float] = None
    latents_mean: Optional[tuple[float, ...]] = None
    latents_std: Optional[tuple[float, ...]] = None
    flow_shift: Optional[float] = None

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must be (t, h, w), got {self.patch_size!r}")

        scalar = self.scaling_factor is not None
        per_channel = self.latents_mean is not None or self.latents_std is not None
        if scalar == per_channel:
            raise ValueError(
                f"{self.name}: specify exactly one normalisation convention, either "
                "scaling_factor or latents_mean+latents_std"
            )
        if per_channel:
            if self.latents_mean is None or self.latents_std is None:
                raise ValueError(f"{self.name}: latents_mean and latents_std must both be set")
            if len(self.latents_mean) != self.channels or len(self.latents_std) != self.channels:
                raise ValueError(
                    f"{self.name}: latents_mean/std must have {self.channels} entries, got "
                    f"{len(self.latents_mean)}/{len(self.latents_std)}"
                )

        if (self.default_num_frames - 1) % self.temporal_ratio != 0:
            raise ValueError(
                f"{self.name}: default_num_frames={self.default_num_frames} is not of the form "
                f"k*{self.temporal_ratio}+1, which the causal VAE requires"
            )


def num_latent_frames(num_frames: int, spec: LatentSpec) -> int:
    """Frame count after the causal VAE's temporal downsampling."""
    return (num_frames - 1) // spec.temporal_ratio + 1


def token_grid(latent_shape: Sequence[int], spec: LatentSpec) -> tuple[int, int, int]:
    """DiT token grid ``(T_tok, H_tok, W_tok)`` for a canonical ``(T, C, H, W)`` latent."""
    t, _, h, w = latent_shape
    p_t, p_h, p_w = spec.patch_size
    return t // p_t, h // p_h, w // p_w


def to_canonical(latents: torch.Tensor, spec: LatentSpec) -> torch.Tensor:
    """Convert a batched latent in the spec's layout to canonical ``(T, C, H, W)``.

    Unbatched 4-D input is assumed to be canonical already and passed through.
    """
    if latents.ndim == 4:
        return latents
    if latents.ndim != 5:
        raise ValueError(f"expected a 4-D or 5-D latent, got shape {tuple(latents.shape)}")
    if latents.shape[0] != 1:
        raise ValueError(
            f"canonical form is unbatched; got batch size {latents.shape[0]}. Split the batch first."
        )
    latents = latents[0]
    if spec.layout == "BCTHW":
        latents = latents.permute(1, 0, 2, 3)
    return latents


def from_canonical(latents: torch.Tensor, spec: LatentSpec) -> torch.Tensor:
    """Convert a canonical ``(T, C, H, W)`` latent to a batched tensor in the spec's layout."""
    if latents.ndim != 4:
        raise ValueError(f"expected a canonical 4-D latent, got shape {tuple(latents.shape)}")
    if spec.layout == "BCTHW":
        latents = latents.permute(1, 0, 2, 3)
    return latents.unsqueeze(0)


def _channel_stats(spec: LatentSpec, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std broadcastable over canonical ``(T, C, H, W)``."""
    kwargs = {"device": latents.device, "dtype": latents.dtype}
    mean = torch.tensor(spec.latents_mean, **kwargs).view(1, -1, 1, 1)
    std = torch.tensor(spec.latents_std, **kwargs).view(1, -1, 1, 1)
    return mean, std


def normalize(latents: torch.Tensor, spec: LatentSpec) -> torch.Tensor:
    """Raw VAE latents -> the normalised space the transformer operates in.

    Expects canonical ``(T, C, H, W)``.
    """
    if spec.scaling_factor is not None:
        return latents * spec.scaling_factor
    mean, std = _channel_stats(spec, latents)
    return (latents - mean) / std


def denormalize(latents: torch.Tensor, spec: LatentSpec) -> torch.Tensor:
    """Normalised latents -> raw VAE latents. Inverse of :func:`normalize`."""
    if spec.scaling_factor is not None:
        return latents / spec.scaling_factor
    mean, std = _channel_stats(spec, latents)
    return latents * std + mean


@dataclass(frozen=True)
class DenoiserState:
    """The denoiser's view of one point on the sampling trajectory.

    All tensors are canonical ``(T, C, H, W)`` and live in normalised latent space.

    ``tau`` is a common diffusion-time coordinate shared by every backend:
    ``tau = 1 - timestep / num_train_timesteps``, so ``tau=0`` is pure noise and
    ``tau=1`` is clean data. This matches the convention in "The Invisible Hand of
    Physics" (``Z_0`` noise, ``Z_1`` data).
    """

    latents: torch.Tensor
    x0: torch.Tensor
    eps: torch.Tensor
    tau: float

    @property
    def drift(self) -> torch.Tensor:
        """Probability-flow drift ``dz/dtau``, oriented noise -> data.

        Defined as ``x0 - eps`` for every backend. For a flow-matching model this is
        exactly the negated network output and therefore exactly the PF-ODE drift.
        For a VP model (CogVideoX) it is the drift of the equivalent rectified path
        through the same ``(x0, eps)`` endpoints -- the standard VP/flow-matching
        correspondence -- which is what makes the two backends comparable at all.

        It is an affine function of ``latents`` and the network output, so the frame
        difference operator commutes with it exactly. See :mod:`phaselock.metrics.flow_geometry`.
        """
        return self.x0 - self.eps


class VideoBackend(ABC):
    """A loaded video diffusion pipeline plus everything needed to probe it."""

    def __init__(self, pipe: Any, spec: LatentSpec, device: Optional[torch.device] = None):
        self.pipe = pipe
        self.spec = spec
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- model structure ---------------------------------------------------

    @property
    @abstractmethod
    def blocks(self) -> torch.nn.ModuleList:
        """The DiT blocks, in depth order, for forward-hook registration."""

    @staticmethod
    @abstractmethod
    def block_hidden_states(block_output: Any) -> torch.Tensor:
        """Extract the video hidden states from one block's return value.

        Both supported backends emit ``(B, T_tok*H_tok*W_tok, D)`` with row-major
        ``(t, h, w)`` ordering and no text tokens, but they package it differently.
        """

    @property
    def num_train_timesteps(self) -> int:
        return int(self.pipe.scheduler.config.num_train_timesteps)

    # -- VAE ---------------------------------------------------------------

    def _to_vae(self, tensor: torch.Tensor, vae: Any) -> torch.Tensor:
        """Move a tensor to the VAE's dtype on the pipeline's execution device.

        Two traps. ``Tensor.to(dtype, device=...)`` is not a valid overload -- device must
        come first or both must be keywords. And under ``enable_model_cpu_offload`` the
        module's own ``.device`` reads ``cpu`` while accelerate's hooks execute it on the
        GPU, so following ``vae.device`` would hand it a CPU tensor and fail.
        """
        return tensor.to(device=self.device, dtype=vae.dtype)

    @abstractmethod
    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a ``(1, C, F, H, W)`` video in [-1, 1] to raw latents in the spec's layout."""

    @abstractmethod
    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode raw latents in the spec's layout to a ``(1, C, F, H, W)`` video in [-1, 1]."""

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to normalised canonical latents.

        Args:
            frames: ``(F, 3, H, W)`` in [0, 1].

        Returns:
            ``(T, C, h, w)`` normalised latents.

        Uses the posterior mode rather than a sample: a stochastic encode is not
        reproducible, and every downstream comparison depends on it being so.
        """
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected frames of shape (F, 3, H, W), got {tuple(frames.shape)}")
        video = frames.to(self.device) * 2.0 - 1.0
        video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, F, H, W)
        latents = self._vae_encode(video)
        return normalize(to_canonical(latents.float(), self.spec), self.spec)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalised canonical latents to ``(F, 3, H, W)`` frames in [0, 1]."""
        raw = denormalize(to_canonical(latents, self.spec), self.spec)
        video = self._vae_decode(from_canonical(raw, self.spec))
        frames = video[0].permute(1, 0, 2, 3)  # (F, 3, H, W)
        return ((frames.float() + 1.0) / 2.0).clamp(0.0, 1.0)

    # -- denoiser ----------------------------------------------------------

    @abstractmethod
    def denoiser_state(
        self,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> DenoiserState:
        """Convert the network's native prediction into a backend-independent state.

        Args:
            latents: Current latents, batched in the spec's layout or canonical.
            model_output: Raw transformer output, same layout as ``latents``.
            timestep: The timestep the output was produced at.
        """

    @abstractmethod
    def renoise(
        self, x0: torch.Tensor, eps: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        """Place a ``(x0, eps)`` pair at the noise level of ``timestep``.

        This single operation drives both directions of the ODE. Sampling evaluates the
        denoiser and renoises to a *lower* level; inversion evaluates and renoises to a
        *higher* one. Writing it once per backend is what keeps
        :mod:`phaselock.pipelines.inversion` backend-agnostic.

        Args and return are canonical ``(T, C, H, W)``.
        """

    @abstractmethod
    def transformer_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> torch.Tensor:
        """One raw transformer evaluation, batched in the spec's layout.

        ``conditioning`` carries backend-specific tensors (text embeddings, image
        embeddings, rotary embeddings) as produced by :meth:`prepare_conditioning`.
        """

    @abstractmethod
    def prepare_conditioning(
        self,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the conditioning payload consumed by :meth:`transformer_forward`."""

    # -- generation --------------------------------------------------------

    def generation_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Pipeline call kwargs with the spec's native defaults filled in.

        ``None`` values are dropped, so callers can pass optional arguments through
        unconditionally.
        """
        kwargs: dict[str, Any] = {
            "num_frames": self.spec.default_num_frames,
            "height": self.spec.default_height,
            "width": self.spec.default_width,
        }
        kwargs.update(overrides)
        return {k: v for k, v in kwargs.items() if v is not None}
