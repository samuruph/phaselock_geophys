"""Baseline generation with internal-state recording.

Unlike inversion, which runs its own integration loop, this drives the **shipped
diffusers pipeline** and observes it. That matters: a hand-rolled sampler would drift
from the pipeline's scheduler in ways that are hard to notice and would make generated
clips incomparable to anything produced normally.

Observation is a forward hook on the transformer, which sees both the latents going in
and the prediction coming out. Two wrinkles:

* Classifier-free guidance means the raw calls are not what drove the trajectory.
  CogVideoX issues one call on a doubled batch; Wan issues two sequential calls. The
  backend declares which, and :meth:`~phaselock.backends.base.VideoBackend.combine_cfg`
  reassembles the guided prediction.
* An image-to-video pipeline concatenates conditioning channels onto the latents, so the
  hook's input has more channels than the latent space. Those are sliced back off.

Everything is baseline sampling. PhaseLock guidance is never applied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch

from ..backends.base import VideoBackend, num_latent_frames, token_grid
from ..probes.pooling import PoolMode
from ..probes.recorder import ProbeRecorder
from ..probes.trajectory import ProbeRecord
from .inversion import evenly_spaced


@dataclass
class GenerationResult:
    """A generated clip and the internal states recorded while producing it."""

    frames: list
    record: ProbeRecord
    seed: int
    num_steps: int


@dataclass
class _StepCapture:
    """Transformer calls observed within the current denoising step."""

    calls: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def clear(self) -> None:
        self.calls.clear()


@torch.no_grad()
def generate_with_probes(
    backend: VideoBackend,
    *,
    prompt: str,
    image: Any = None,
    num_steps: int = 50,
    record_steps: int = 10,
    sources: Sequence[str] = ("hidden_states", "latent", "x0_hat", "velocity"),
    blocks: Optional[Iterable[int]] = None,
    block_stride: int = 1,
    pooling: PoolMode = "mean",
    guidance_scale: float = 6.0,
    negative_prompt: Optional[str] = None,
    seed: int = 42,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> GenerationResult:
    """Generate one clip with the pipeline, recording internal states as it goes.

    Recording costs storage, not compute: the probed steps reuse the transformer
    evaluations the sampler was already making.
    """
    spec = backend.spec
    num_frames = num_frames or spec.default_num_frames
    height = height or spec.default_height
    width = width or spec.default_width

    grid = (
        num_latent_frames(num_frames, spec),
        height // spec.spatial_ratio // spec.patch_size[1],
        width // spec.spatial_ratio // spec.patch_size[2],
    )
    to_record = set(evenly_spaced(num_steps, record_steps))
    capture = _StepCapture()

    with ProbeRecorder(
        backend, sources=sources, grid=grid, blocks=blocks, block_stride=block_stride, pooling=pooling
    ) as probe:
        probe.set_provenance(
            direction="generation",
            num_steps=num_steps,
            record_steps=sorted(to_record),
            prompt=prompt,
            guidance_scale=guidance_scale,
            seed=seed,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            **(provenance or {}),
        )

        def transformer_hook(_module, args, kwargs, output):
            if not probe.armed:
                return
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            prediction = output[0] if isinstance(output, (tuple, list)) else output
            capture.calls.append((hidden_states.detach(), prediction.detach()))

        handle = backend.pipe.transformer.register_forward_hook(
            transformer_hook, with_kwargs=True
        )
        recorded: list[int] = []

        def on_step_end(_pipe, step_index, timestep, callback_kwargs):
            """Fires after the scheduler step, once all of this step's calls are in."""
            if step_index in to_record and capture.calls:
                latents, model_output = backend.combine_cfg(capture.calls, guidance_scale)
                probe.capture(
                    len(recorded), backend.denoiser_state(latents, model_output, timestep)
                )
                recorded.append(step_index)
            capture.clear()
            # Arm for the next step only if it is one we want.
            probe.arm() if (step_index + 1) in to_record else probe.disarm()
            return callback_kwargs

        probe.arm() if 0 in to_record else probe.disarm()
        try:
            call = backend.generation_kwargs(
                prompt=prompt,
                image=image,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            )
            result = backend.pipe(
                **call,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=backend.device).manual_seed(seed),
                callback_on_step_end=on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        finally:
            handle.remove()

        record = probe.record

    return GenerationResult(
        frames=result.frames[0], record=record, seed=seed, num_steps=num_steps
    )
