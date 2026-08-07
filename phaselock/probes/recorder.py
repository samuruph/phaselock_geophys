"""Capturing a diffusion model's internal state during sampling or inversion.

Usage::

    with ProbeRecorder(backend, sources=["hidden_states", "latent"], grid=grid) as probe:
        for step, timestep in enumerate(timesteps):
            model_output = backend.transformer_forward(z, timestep, conditioning)
            state = backend.denoiser_state(z, model_output, timestep)
            probe.capture(step, state)          # flushes the hooked hidden states too
            z = advance(z, state)
    record = probe.record

Hooks fire during ``transformer_forward`` and stage their pooled output; ``capture``
attaches that staging buffer to a step index together with the latent-space sources
derived from the :class:`~phaselock.backends.base.DenoiserState`. Any step you do not
call ``capture`` for is discarded, which is how a 50-step run records only 10 steps.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import torch

from ..backends.base import DenoiserState, VideoBackend
from .pooling import PoolMode, pool_latents, pool_tokens, resolve_blocks
from .sources import (
    ATTENTION,
    HIDDEN_STATES,
    LATENT,
    PER_BLOCK_SOURCES,
    SOURCES,
    VELOCITY,
    X0_HAT,
    validate_sources,
)
from .trajectory import NO_BLOCK, ProbeRecord, TrajectoryKey


class ProbeRecorder:
    """Records pooled internal trajectories over a sampling or inversion trajectory.

    Args:
        backend: The loaded backend. Supplies the block list and the layout conventions.
        sources: Which internal signals to record. See :data:`~phaselock.probes.sources.SOURCES`.
        grid: DiT token grid ``(T_tok, H_tok, W_tok)``, from
            :func:`~phaselock.backends.base.token_grid`.
        blocks: Explicit block indices, or ``None`` to take every ``block_stride``-th.
        block_stride: Used when ``blocks`` is ``None``.
        pooling: ``"mean"`` matches GeoPhys's spatial average pooling.
    """

    def __init__(
        self,
        backend: VideoBackend,
        sources: Sequence[str],
        grid: tuple[int, int, int],
        blocks: Optional[Iterable[int]] = None,
        block_stride: int = 1,
        pooling: PoolMode = "mean",
    ):
        self.backend = backend
        self.sources = validate_sources(sources)
        self.grid = grid
        self.pooling = pooling
        self.record = ProbeRecord()

        needs_blocks = bool(set(self.sources) & PER_BLOCK_SOURCES)
        self.blocks = (
            resolve_blocks(len(backend.blocks), list(blocks) if blocks is not None else None, block_stride)
            if needs_blocks
            else []
        )

        self._handles: list[Any] = []
        self._staged: dict[tuple[str, int], torch.Tensor] = {}
        self._armed = True

    # -- arming ------------------------------------------------------------

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        """Let hooks stage output on the next forward pass."""
        self._armed = True

    def disarm(self) -> None:
        """Make hooks no-op and drop anything already staged.

        A typical run probes 10 of 50 steps. Pooling is cheap next to a transformer
        forward, but it still touches a ~100 MB tensor per block, so skipping it on the
        40 unrecorded steps is worth the flag.
        """
        self._armed = False
        self._staged.clear()

    # -- context management ------------------------------------------------

    def __enter__(self) -> "ProbeRecorder":
        extract = self.backend.block_hidden_states

        def make_hook(source: str, index: int):
            def hook(_module, _inputs, output):
                if not self._armed:
                    return
                key = (source, index)
                # Under sequential CFG (Wan) the block runs twice per step, conditional
                # first. Keep the first and ignore the unconditional pass.
                if key in self._staged:
                    return

                hidden = extract(output).detach()
                if hidden.shape[0] > 1:
                    # Under batched CFG (CogVideoX) the batch is [uncond, cond]. There is
                    # no principled way to "guide" a hidden state -- guidance combines
                    # *predictions*, not activations -- so the conditional pass is
                    # recorded, matching the sequential case above.
                    hidden = hidden[hidden.shape[0] // 2 :]

                # Pool immediately: the raw state is ~100 MB, the pooled one ~80 KB.
                self._staged[key] = pool_tokens(hidden.float(), self.grid, self.pooling).cpu()

            return hook

        for index in self.blocks:
            block = self.backend.blocks[index]
            if HIDDEN_STATES in self.sources:
                self._handles.append(block.register_forward_hook(make_hook(HIDDEN_STATES, index)))
            if ATTENTION in self.sources:
                attention = getattr(block, "attn1", None)
                if attention is None:
                    raise AttributeError(
                        f"block {index} has no 'attn1' submodule; the {ATTENTION!r} source "
                        f"is not available for this backend"
                    )
                self._handles.append(attention.register_forward_hook(make_hook(ATTENTION, index)))
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._staged.clear()

    # -- recording ---------------------------------------------------------

    def capture(self, step: int, state: DenoiserState) -> None:
        """Attach the staged hidden states and the latent-space sources to ``step``.

        Call once per denoising step you want recorded, after the transformer forward
        that produced ``state``.
        """
        if step in self.record.taus:
            raise KeyError(f"step {step} already captured")
        self.record.taus[step] = state.tau

        for (source, block), pooled in self._staged.items():
            self.record.add(TrajectoryKey(source=source, block=block, step=step), pooled)
        self._staged.clear()

        latent_sources = {
            LATENT: state.latents,
            X0_HAT: state.x0,
            VELOCITY: state.drift,
        }
        for source, tensor in latent_sources.items():
            if source in self.sources:
                self.record.add(
                    TrajectoryKey(source=source, block=NO_BLOCK, step=step),
                    pool_latents(tensor.detach().float(), self.pooling),
                )

    def set_provenance(self, **fields: Any) -> None:
        """Record how this trajectory was produced, so two runs can be told apart."""
        self.record.provenance.update(
            {
                "backend": self.backend.spec.name,
                "layout": self.backend.spec.layout,
                "sources": list(self.sources),
                "blocks": self.blocks,
                "pooling": self.pooling,
                "grid": list(self.grid),
                **fields,
            }
        )
