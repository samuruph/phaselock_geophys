"""Same-timestep P&P kernel; momentum updates only after this kernel returns."""
import torch

from ..backends.base import from_canonical, to_canonical
from .masks import motion_confidence


def refine_step(backend, scheduler, latents, timestep, predict, iterations, generator,
                record=False, save_raw=False):
    """Return accepted continuation, final denoiser state, confidence and diagnostics.

    `predict` returns one CFG-combined prediction for its input. DDIM is stateless,
    so candidate continuations do not advance history or consume generation RNG.
    Like the Wan I2V reference, P&P re-noises the complete clean estimate; image
    conditioning lives in separate model-input channels.
    """
    original = latents
    candidate = latents
    previous_clean = None
    base_next = None
    base_clean = None
    scalars = []
    extras = {}
    for inner in range(iterations + 1):
        if inner:
            noise = torch.randn(
                clean.shape, device=clean.device, dtype=original.dtype, generator=generator
            )
            candidate = from_canonical(
                backend.renoise(clean, noise, timestep), backend.spec
            ).to(original.dtype)
        prediction = predict(candidate, timestep)
        state = backend.denoiser_state(candidate, prediction, timestep)
        next_latents, scheduler_clean = scheduler.step(
            prediction, timestep, candidate, return_dict=False
        )
        next_latents = next_latents.to(original.dtype)
        # Match the clean estimate used by the sampler. In BF16 this can differ
        # slightly from the backend's FP32 reconstruction of the same prediction.
        clean = to_canonical(scheduler_clean, backend.spec)
        if inner == 0 and record:
            base_next, base_clean = next_latents, state.x0
        if record:
            motion = state.x0[1:] - state.x0[:-1]
            scalars.append({
                "iteration": inner,
                "motion_rms": float(motion.square().mean().sqrt()),
                "clean_rms": float(state.x0.square().mean().sqrt()),
            })
        if record and save_raw:
            extras[f"inner_clean_{inner}"] = state.x0
        if inner < iterations:
            previous_clean = state.x0
    confidence = None
    if iterations:
        disagreement, confidence = motion_confidence(previous_clean, state.x0)
        if record:
            extras.update(
                pnp_disagreement=disagreement,
                confidence=confidence,
                original_clean=base_clean,
                refined_clean=state.x0,
                refinement_update=to_canonical(
                    next_latents.float() - base_next.float(), backend.spec
                ),
            )
    if record:
        extras.update(inner_iterations=scalars, guided_predictions=iterations + 1)
    return next_latents, state, confidence, extras
