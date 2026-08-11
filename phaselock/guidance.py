"""Latent Delta Guidance, the PhaseLock mechanism.

Transfers the motion prior captured by a few-step generation into a full-length one, by
constraining frame-to-frame latent differences::

    M_current = T(z) = z[2:F] - z[1:F-1]
    G         = M_prior - M_current
    z[2:F]   += lambda(k) * G

with a linearly decaying schedule over ``[k_start, k_end)``. Frame 1 is the conditioning
anchor and is never modified.

``T`` is the *first difference along frames*, and it is the axis this repo ablates: see
:mod:`phaselock.operators` for the second and third differences and the span residual,
which the detection study says carry 20+ points more signal in this very space. The
mechanism below is unchanged for all of them -- only what ``T`` measures differs.

Layout correctness is load-bearing: the original implementation hardcoded CogVideoX's
``(B, T, C, H, W)`` and Wan's latents are ``(B, C, T, H, W)`` -- unfixed, the guidance
would difference the *channel* axis and still return a tensor of an entirely plausible
shape.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .backends.base import LatentSpec, from_canonical, to_canonical
from .operators import get_operator, strength_scale

SOURCES = ("latent", "x0_hat", "velocity")
"""Which tensor the frame operator is measured on.

``latent`` is the sampler state and is what PhaseLock uses. The other two are model
*outputs* -- what the network currently believes the clean video is, and the flow field
carrying the state there -- so they exist only alongside a prediction, and the correction
they imply is written back through ``x0``. See :meth:`LatentDeltaGuidance.apply_to_state`.
"""


def extract_prior(
    few_latents: torch.Tensor,
    spec: Optional[LatentSpec] = None,
    few_step_prior_type: str = "motion",
) -> torch.Tensor:
    """Apply a frame operator to the few-step pass, giving the target to match.

    ``operator="motion"`` is PhaseLock's Latent Delta Operator ``T(z) = z[2:F] - z[1:F-1]``
    and is the default, so existing callers are unaffected. The other operators in
    :mod:`phaselock.operators` are the ablation: same mechanism, same shapes, a different
    quantity held fixed.

    Args:
        few_latents: Latents from the few-step pass, canonical ``(T, C, H, W)`` or
            batched in ``spec``'s layout.
        spec: Required only if ``few_latents`` is batched.
        few_step_prior_type: Name from :data:`phaselock.operators.OPERATORS`.

    Returns:
        Canonical ``(T - anchor, C, H, W)`` prior.
    """
    if few_latents.ndim == 5:
        if spec is None:
            raise ValueError("a LatentSpec is required to interpret batched latents")
        few_latents = to_canonical(few_latents, spec)
    if few_latents.ndim != 4:
        raise ValueError(f"expected (T, C, H, W) latents, got {tuple(few_latents.shape)}")
    return get_operator(few_step_prior_type)(few_latents)


def extract_motion_prior(
    few_latents: torch.Tensor, spec: Optional[LatentSpec] = None
) -> torch.Tensor:
    """PhaseLock's first-difference prior. Kept as the name the paper uses."""
    return extract_prior(few_latents, spec, few_step_prior_type="motion")


class LatentDeltaGuidance:
    """A ``callback_on_step_end`` callable that applies Latent Delta Guidance.

    Args:
        motion_prior: Canonical ``(T-1, C, H, W)`` deltas from the few-step pass.
        spec: Latent spec of the backend being guided. This is what makes the callback
            layout-correct for both CogVideoX and Wan.
        guidance_strength: Initial strength ``lambda_0``.
        guide_start: First step at which guidance applies.
        guide_end: First step at which it stops. Defaults to ``total_steps // 2``.
        total_steps: Number of denoising steps in the guided pass.
        normalise_strength: Scale ``lambda`` so one unit is the same intervention for
            every operator, relative to ``motion``. On by default; without it a ranking
            partly measures step size, and ``jerk`` diverges. See
            :func:`phaselock.operators.amplification`.
        few_step_prior_type: Which quantity the few-step pass was mined for and this pass
            is held to -- ``motion`` is PhaseLock's own first difference. See
            :mod:`phaselock.operators`.
    """

    def __init__(
        self,
        motion_prior: torch.Tensor,
        spec: LatentSpec,
        guidance_strength: float = 0.05,
        guide_start: int = 0,
        guide_end: Optional[int] = None,
        total_steps: int = 50,
        few_step_prior_type: str = "motion",
        source: str = "latent",
        backend: Any = None,
        normalise_strength: bool = True,
    ):
        if motion_prior.ndim != 4:
            raise ValueError(
                f"motion_prior must be canonical (T-1, C, H, W), got {tuple(motion_prior.shape)}"
            )
        if guidance_strength < 0:
            raise ValueError(f"guidance_strength must be non-negative, got {guidance_strength}")
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
        if source != "latent" and backend is None:
            raise ValueError(
                f"source {source!r} is a model output, not the sampler state, so it needs "
                "the backend to invert the denoiser's parameterisation"
            )

        self.motion_prior = motion_prior
        self.few_step_prior_type = get_operator(few_step_prior_type)
        self.source = source
        self.backend = backend
        self.spec = spec
        self.guidance_strength = guidance_strength
        self.guide_start = guide_start
        self.guide_end = guide_end if guide_end is not None else total_steps // 2
        self.total_steps = total_steps
        self._previous_latents: Optional[torch.Tensor] = None

        # A frame is shared between neighbouring windows, so correcting one disturbs its
        # neighbours and the write overshoots by the operator's order -- 1.4x for motion,
        # 4.1x for jerk. PhaseLock is a soft nudge, so that diffusion is not itself wrong;
        # a 4x-larger nudge is, and at third order it stops the schedule settling at all.
        # Scaling relative to `motion` keeps the published arm exactly as it was.
        frames = motion_prior.shape[0] + self.few_step_prior_type.anchor
        self.strength_scale = (
            strength_scale(self.few_step_prior_type.name, frames)
            if normalise_strength else 1.0
        )

        if self.guide_end <= self.guide_start:
            raise ValueError(
                f"guide_end ({self.guide_end}) must exceed guide_start ({self.guide_start})"
            )

    def compute_schedule(self, step_index: int) -> float:
        """``lambda(k) = lambda_0 * (1 - (k - k_start) / (k_end - k_start))``.

        Strong while the global layout is forming, relaxing to leave later steps free
        for texture refinement.
        """
        if step_index < self.guide_start or step_index >= self.guide_end:
            return 0.0
        progress = (step_index - self.guide_start) / (self.guide_end - self.guide_start)
        return self.guidance_strength * (1.0 - progress) * self.strength_scale

    def __call__(
        self,
        pipe: Any,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        latents = callback_kwargs.get("latents")
        strength = self.compute_schedule(step_index)

        # The callback fires *after* scheduler.step(), so `latents` is already the next
        # state. Remember it: it is what the pipeline feeds the following step, which is
        # the state `noise_pred` will then correspond to. Kept regardless of strength, so
        # the chain is unbroken when guidance switches on mid-schedule.
        previous, self._previous_latents = self._previous_latents, latents

        if strength == 0.0 or latents is None:
            return callback_kwargs

        if self.source == "latent":
            callback_kwargs["latents"] = self.apply(latents, strength)
            return callback_kwargs

        # x0_hat and velocity are the model's *output*, not the sampler state, so they
        # only exist alongside a prediction. `noise_pred` is requestable once the pipeline
        # instance's _callback_tensor_inputs is extended, and it arrives already
        # CFG-combined -- diffusers does that before calling us.
        noise_pred = callback_kwargs.get("noise_pred")
        if noise_pred is None or previous is None:
            return callback_kwargs

        # `timestep` is the step just taken, which is the one `noise_pred` and
        # `previous` belong to. The latents being written are the state *after* it, so the
        # affine scale is read at the next timestep on the schedule.
        schedule = getattr(pipe, "scheduler", None)
        timesteps = getattr(schedule, "timesteps", None)
        next_timestep = (
            timesteps[step_index + 1]
            if timesteps is not None and step_index + 1 < len(timesteps)
            else 0.0
        )
        callback_kwargs["latents"] = self.apply_to_state(
            latents, previous, noise_pred, timestep, next_timestep, strength
        )
        return callback_kwargs

    def apply(self, latents: torch.Tensor, strength: float) -> torch.Tensor:
        """Nudge the operator's value toward the prior, PhaseLock's equation (2).

        ``G = M_prior - T(z)``, written onto the frames after the anchor::

            z[anchor:] <- z[anchor:] + lambda * G

        For ``motion`` the anchor is 1, which is the paper verbatim: the first frame is
        the image condition and is never modified. A higher-order operator consumes more
        leading frames before producing its first value, so it anchors that many -- the
        conditioning is preserved for every operator, not just the first-order one.

        Converts to canonical form first, so one implementation covers every layout.
        """
        canonical = to_canonical(latents, self.spec)
        anchor = self.few_step_prior_type.anchor
        expected = canonical.shape[0] - anchor
        if expected != self.motion_prior.shape[0]:
            raise ValueError(
                f"{self.few_step_prior_type.name} prior covers {self.motion_prior.shape[0]} windows but "
                f"the latents give {expected}"
            )

        prior = self.motion_prior.to(canonical.device, canonical.dtype)
        current = self.few_step_prior_type(canonical)

        guided = canonical.clone()
        guided[anchor:] = canonical[anchor:] + strength * (prior - current)
        return from_canonical(guided, self.spec).to(latents.dtype)

    def apply_to_state(
        self,
        latents: torch.Tensor,
        previous_latents: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """The same update, measured on a model output rather than the sampler state.

        ``x0_hat`` and ``velocity`` are not states we can write to -- they are what the
        network says about the state. But the two are related by an affine map: for every
        backend ``renoise(x0, eps, t)`` is a fixed linear combination, so a change to
        ``x0`` induces an exactly proportional change in the latent and nothing has to be
        reimplemented::

            renoise(x0 + d, eps, t) - renoise(x0, eps, t) = sqrt(alpha_bar_t) * d

        So the *source* chooses what is measured and the write always lands on ``x0``.
        For ``velocity = x0 - eps`` with ``eps`` held fixed, a change of ``d`` in ``x0``
        is a change of ``d`` in the velocity, so the correction carries across unchanged.
        """
        state = self.backend.denoiser_state(previous_latents, model_output, timestep)
        measured = state.x0 if self.source == "x0_hat" else state.drift

        anchor = self.few_step_prior_type.anchor
        expected = measured.shape[0] - anchor
        if expected != self.motion_prior.shape[0]:
            raise ValueError(
                f"{self.few_step_prior_type.name} prior covers {self.motion_prior.shape[0]} "
                f"windows but the {self.source} trajectory gives {expected}"
            )

        prior = self.motion_prior.to(measured.device, measured.dtype)
        correction = strength * (prior - self.few_step_prior_type(measured))

        # Only the frames after the anchor move, exactly as in `apply`.
        delta = torch.zeros_like(measured)
        delta[anchor:] = correction

        canonical = to_canonical(latents, self.spec)
        scale = self.backend.renoise_scale(next_timestep).to(canonical.device)
        guided = canonical + (scale * delta).to(canonical.dtype)
        return from_canonical(guided, self.spec).to(latents.dtype)
