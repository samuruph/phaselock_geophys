"""Tests for Latent Delta Guidance, focused on the layout bug that motivated the rewrite.

The original implementation unpacked ``B, T, C, H, W`` from the latent shape. CogVideoX
latents really are ``BTCHW``, but Wan's are ``BCTHW``, so on Wan the guidance differenced
the *channel* axis while still returning a tensor of an entirely plausible shape and
dtype. Nothing would have raised; the method would simply have been guiding on nonsense.
"""

from __future__ import annotations

import pytest
import torch

from phaselock.backends import COGVIDEOX_5B, WAN21_T2V_1_3B, from_canonical, to_canonical
from phaselock.guidance import LatentDeltaGuidance, extract_motion_prior

SPECS = [COGVIDEOX_5B, WAN21_T2V_1_3B]
SPEC_IDS = [spec.name for spec in SPECS]

FRAMES, CHANNELS, HEIGHT, WIDTH = 13, 16, 6, 8


def canonical_latent(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(FRAMES, CHANNELS, HEIGHT, WIDTH, generator=generator)


# -- the motion prior -------------------------------------------------------


def test_motion_prior_is_the_frame_difference():
    z = canonical_latent()
    prior = extract_motion_prior(z)
    assert prior.shape == (FRAMES - 1, CHANNELS, HEIGHT, WIDTH)
    assert torch.equal(prior, z[1:] - z[:-1])


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_motion_prior_accepts_batched_latents_in_either_layout(spec):
    z = canonical_latent()
    assert torch.equal(extract_motion_prior(from_canonical(z, spec), spec), extract_motion_prior(z))


def test_motion_prior_requires_a_spec_for_batched_input():
    with pytest.raises(ValueError, match="LatentSpec is required"):
        extract_motion_prior(from_canonical(canonical_latent(), COGVIDEOX_5B))


def test_motion_prior_needs_two_frames():
    with pytest.raises(ValueError, match="at least two latent frames"):
        extract_motion_prior(torch.randn(1, CHANNELS, HEIGHT, WIDTH))


# -- the schedule -----------------------------------------------------------


def test_schedule_decays_linearly_and_is_zero_outside_the_window():
    guidance = LatentDeltaGuidance(
        torch.zeros(FRAMES - 1, CHANNELS, HEIGHT, WIDTH),
        spec=COGVIDEOX_5B,
        guidance_strength=0.05,
        guide_start=0,
        guide_end=25,
        total_steps=50,
    )
    assert guidance.compute_schedule(0) == pytest.approx(0.05)
    assert guidance.compute_schedule(12) == pytest.approx(0.05 * (1 - 12 / 25))
    assert guidance.compute_schedule(25) == 0.0
    assert guidance.compute_schedule(49) == 0.0


def test_guide_end_defaults_to_half_the_steps():
    guidance = LatentDeltaGuidance(
        torch.zeros(FRAMES - 1, CHANNELS, HEIGHT, WIDTH), spec=COGVIDEOX_5B, total_steps=50
    )
    assert guidance.guide_end == 25


def test_invalid_guidance_settings_raise():
    prior = torch.zeros(FRAMES - 1, CHANNELS, HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="must exceed"):
        LatentDeltaGuidance(prior, spec=COGVIDEOX_5B, guide_start=30, guide_end=10)
    with pytest.raises(ValueError, match="non-negative"):
        LatentDeltaGuidance(prior, spec=COGVIDEOX_5B, guidance_strength=-0.1)
    with pytest.raises(ValueError, match="canonical"):
        LatentDeltaGuidance(torch.zeros(2, 3), spec=COGVIDEOX_5B)


# -- layout correctness -----------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_guidance_moves_the_latent_towards_the_prior(spec):
    z = canonical_latent(0)
    target = canonical_latent(1)
    prior = extract_motion_prior(target)

    guidance = LatentDeltaGuidance(prior, spec=spec, guidance_strength=1.0)
    guided = to_canonical(guidance.apply(from_canonical(z, spec), strength=1.0), spec)

    before = (extract_motion_prior(z) - prior).norm()
    after = (extract_motion_prior(guided) - prior).norm()
    assert after < before


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_frame_zero_is_never_modified(spec):
    """Frame 0 is the conditioning anchor; touching it would break I2V entirely."""
    z = canonical_latent(0)
    prior = extract_motion_prior(canonical_latent(1))
    guidance = LatentDeltaGuidance(prior, spec=spec, guidance_strength=1.0)
    guided = to_canonical(guidance.apply(from_canonical(z, spec), strength=0.5), spec)
    assert torch.allclose(guided[0], z[0])


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_full_strength_guidance_pins_the_first_transition(spec):
    """The update is simultaneous, so it is not a fixed point beyond the first frame.

    Every frame is shifted using the deltas measured *before* the update, so once frame
    1 moves it changes the delta into frame 2 as well. Only the transition out of the
    unmodified anchor lands exactly on the prior; the rest merely move toward it, which
    is what ``test_guidance_moves_the_latent_towards_the_prior`` checks.
    """
    z = canonical_latent(0)
    prior = extract_motion_prior(canonical_latent(1))
    guidance = LatentDeltaGuidance(prior, spec=spec, guidance_strength=1.0)
    guided = to_canonical(guidance.apply(from_canonical(z, spec), strength=1.0), spec)

    assert torch.allclose(guided[1] - guided[0], prior[0], atol=1e-5)


def test_both_layouts_give_the_same_canonical_result():
    """The whole point of canonicalising: layout must not change the physics."""
    z = canonical_latent(0)
    prior = extract_motion_prior(canonical_latent(1))

    results = [
        to_canonical(
            LatentDeltaGuidance(prior, spec=spec, guidance_strength=1.0).apply(
                from_canonical(z, spec), strength=0.3
            ),
            spec,
        )
        for spec in SPECS
    ]
    assert torch.allclose(results[0], results[1], atol=1e-6)


def test_the_old_layout_assumption_would_have_corrupted_wan():
    """Reproduces the original bug to show it was real, not hypothetical.

    Wan latents are BCTHW. Unpacking them as B,T,C,H,W and differencing axis 1 walks the
    channel axis instead of time, and produces a differently-shaped delta -- which the
    original code then broadcast against a prior built on the frame axis.
    """
    z = canonical_latent(0)
    wan_batched = from_canonical(z, WAN21_T2V_1_3B)

    correct = to_canonical(wan_batched, WAN21_T2V_1_3B)
    correct_delta = correct[1:] - correct[:-1]
    naive_delta = wan_batched[:, 1:] - wan_batched[:, :-1]

    assert correct_delta.shape == (FRAMES - 1, CHANNELS, HEIGHT, WIDTH)
    assert naive_delta.shape[1] == CHANNELS - 1  # differenced channels, not frames
    assert correct_delta.shape != naive_delta.shape[1:]


# -- callback plumbing ------------------------------------------------------


def test_callback_returns_kwargs_untouched_outside_the_guidance_window():
    z = from_canonical(canonical_latent(), COGVIDEOX_5B)
    prior = extract_motion_prior(canonical_latent(1))
    guidance = LatentDeltaGuidance(prior, spec=COGVIDEOX_5B, guide_start=0, guide_end=10)

    kwargs = {"latents": z}
    assert guidance(None, 50, torch.tensor(0.0), kwargs)["latents"] is z


def test_callback_modifies_latents_inside_the_window():
    z = from_canonical(canonical_latent(), COGVIDEOX_5B)
    prior = extract_motion_prior(canonical_latent(1))
    guidance = LatentDeltaGuidance(prior, spec=COGVIDEOX_5B, guidance_strength=0.5, guide_end=10)

    out = guidance(None, 0, torch.tensor(0.0), {"latents": z})["latents"]
    assert out.shape == z.shape
    assert not torch.allclose(out, z)


def test_callback_tolerates_a_missing_latents_key():
    guidance = LatentDeltaGuidance(
        extract_motion_prior(canonical_latent()), spec=COGVIDEOX_5B, guide_end=10
    )
    assert guidance(None, 0, torch.tensor(0.0), {}) == {}


def test_a_prior_of_the_wrong_length_raises_instead_of_broadcasting():
    """Silent broadcasting here would guide toward the wrong frames."""
    z = from_canonical(canonical_latent(), COGVIDEOX_5B)
    short = extract_motion_prior(canonical_latent()[:5])
    guidance = LatentDeltaGuidance(short, spec=COGVIDEOX_5B, guidance_strength=0.5)
    with pytest.raises(ValueError, match="motion prior covers"):
        guidance.apply(z, strength=0.5)


def test_guidance_preserves_dtype():
    z = from_canonical(canonical_latent(), COGVIDEOX_5B).to(torch.bfloat16)
    prior = extract_motion_prior(canonical_latent(1))
    guidance = LatentDeltaGuidance(prior, spec=COGVIDEOX_5B, guidance_strength=0.5)
    assert guidance.apply(z, strength=0.5).dtype == torch.bfloat16
