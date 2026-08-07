"""Reducing internal tensors to per-latent-frame trajectories.

GeoPhys operates on one vector per frame, obtained by spatially pooling a frozen
encoder's per-frame features. The same reduction applied to a diffusion model's internals
is what lets the geometry transfer.

Pooling happens as early as possible, inside the forward hook. A raw CogVideoX-5B hidden
state is ``(1, 13*30*45, 3072)``, about 100 MB in fp16; the pooled trajectory is
``(13, 3072)``, about 80 KB. Recording every block at every step is only affordable
because of that reduction.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch

PoolMode = Literal["mean", "flatten"]


def pool_tokens(
    hidden_states: torch.Tensor,
    grid: tuple[int, int, int],
    mode: PoolMode = "mean",
) -> torch.Tensor:
    """DiT token sequence -> per-latent-frame trajectory ``(T, D)``.

    Both supported backends patchify with ``patch_size=(1, 2, 2)`` and flatten in
    row-major ``(t, h, w)`` order, so each latent frame owns a contiguous, equal-length
    run of tokens and the reshape below is exact rather than approximate. A backend with
    temporal patching would break this, which is why
    :func:`~phaselock.backends.base.token_grid` reports the temporal patch and the
    backend tests assert it is 1.

    Args:
        hidden_states: ``(B, T*H*W, D)`` with ``B == 1``, or the unbatched ``(T*H*W, D)``.
        grid: ``(T_tok, H_tok, W_tok)`` from the backend's token grid.
        mode: ``"mean"`` reproduces GeoPhys's spatial average pooling. ``"flatten"``
            keeps every spatial position, giving ``D * H * W`` features per frame.
    """
    if hidden_states.ndim == 3:
        if hidden_states.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {hidden_states.shape[0]}")
        hidden_states = hidden_states[0]
    if hidden_states.ndim != 2:
        raise ValueError(
            f"expected (B, N, D) or (N, D) token states, got {tuple(hidden_states.shape)}"
        )

    frames, height, width = grid
    expected = frames * height * width
    if hidden_states.shape[0] != expected:
        raise ValueError(
            f"token count {hidden_states.shape[0]} does not match grid {grid} "
            f"(expected {expected}). The text tokens may not have been stripped."
        )

    per_frame = hidden_states.reshape(frames, height * width, hidden_states.shape[-1])
    if mode == "mean":
        return per_frame.mean(dim=1)
    if mode == "flatten":
        return per_frame.reshape(frames, -1)
    raise ValueError(f"unknown pool mode {mode!r}; expected 'mean' or 'flatten'")


def pool_latents(latents: torch.Tensor, mode: PoolMode = "mean") -> torch.Tensor:
    """Canonical ``(T, C, H, W)`` latent -> per-frame trajectory ``(T, D)``.

    Mean pooling is linear, which is what lets the frame-difference operator commute
    with the sampler ODE and makes the flow-coupling metrics exact for latent-space
    sources. ``"flatten"`` is also linear and preserves that property.
    """
    if latents.ndim != 4:
        raise ValueError(f"expected canonical (T, C, H, W) latents, got {tuple(latents.shape)}")
    if mode == "mean":
        return latents.mean(dim=(2, 3))
    if mode == "flatten":
        return latents.reshape(latents.shape[0], -1)
    raise ValueError(f"unknown pool mode {mode!r}; expected 'mean' or 'flatten'")


def resolve_blocks(total: int, blocks: Sequence[int] | None, stride: int = 1) -> list[int]:
    """Which block indices to record.

    ``blocks=None`` selects every ``stride``-th block. The last block is always included
    even when the stride would skip it, since it is the one the model actually emits.
    """
    if blocks is not None:
        out_of_range = [b for b in blocks if not 0 <= b < total]
        if out_of_range:
            raise ValueError(f"block indices out of range for a {total}-block model: {out_of_range}")
        return sorted(set(blocks))
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    chosen = set(range(0, total, stride))
    chosen.add(total - 1)
    return sorted(chosen)
