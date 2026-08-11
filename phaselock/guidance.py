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
from .operators import get_operator


def extract_prior(
    few_latents: torch.Tensor,
    spec: Optional[LatentSpec] = None,
    operator: str = "motion",
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
        operator: Name from :data:`phaselock.operators.OPERATORS`.

    Returns:
        Canonical ``(T - anchor, C, H, W)`` prior.
    """
    if few_latents.ndim == 5:
        if spec is None:
            raise ValueError("a LatentSpec is required to interpret batched latents")
        few_latents = to_canonical(few_latents, spec)
    if few_latents.ndim != 4:
        raise ValueError(f"expected (T, C, H, W) latents, got {tuple(few_latents.shape)}")
    return get_operator(operator)(few_latents)


def extract_motion_prior(
    few_latents: torch.Tensor, spec: Optional[LatentSpec] = None
) -> torch.Tensor:
    """PhaseLock's first-difference prior. Kept as the name the paper uses."""
    return extract_prior(few_latents, spec, operator="motion")


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
    """

    def __init__(
        self,
        motion_prior: torch.Tensor,
        spec: LatentSpec,
        guidance_strength: float = 0.05,
        guide_start: int = 0,
        guide_end: Optional[int] = None,
        total_steps: int = 50,
        operator: str = "motion",
    ):
        if motion_prior.ndim != 4:
            raise ValueError(
                f"motion_prior must be canonical (T-1, C, H, W), got {tuple(motion_prior.shape)}"
            )
        if guidance_strength < 0:
            raise ValueError(f"guidance_strength must be non-negative, got {guidance_strength}")

        self.motion_prior = motion_prior
        self.operator = get_operator(operator)
        self.spec = spec
        self.guidance_strength = guidance_strength
        self.guide_start = guide_start
        self.guide_end = guide_end if guide_end is not None else total_steps // 2
        self.total_steps = total_steps

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
        return self.guidance_strength * (1.0 - progress)

    def __call__(
        self,
        pipe: Any,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        strength = self.compute_schedule(step_index)
        if strength == 0.0:
            return callback_kwargs

        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        callback_kwargs["latents"] = self.apply(latents, strength)
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
        anchor = self.operator.anchor
        expected = canonical.shape[0] - anchor
        if expected != self.motion_prior.shape[0]:
            raise ValueError(
                f"{self.operator.name} prior covers {self.motion_prior.shape[0]} windows but "
                f"the latents give {expected}"
            )

        prior = self.motion_prior.to(canonical.device, canonical.dtype)
        current = self.operator(canonical)

        guided = canonical.clone()
        guided[anchor:] = canonical[anchor:] + strength * (prior - current)
        return from_canonical(guided, self.spec).to(latents.dtype)
