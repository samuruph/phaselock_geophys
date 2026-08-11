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
    with pytest.raises(ValueError, match="needs at least 2 frames"):
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


# -- the operator ablation --------------------------------------------------
#
# PhaseLock's T is the first difference along frames. These pin the properties that let
# the second/third difference and the span residual ride the identical mechanism, and --
# most importantly -- that swapping the operator in has not moved the `motion` arm, which
# is the reproduction of the published method.


class TestFrameOperators:
    def test_motion_arm_is_byte_for_byte_the_published_mechanism(self):
        """The regression that keeps the reproduction arm honest.

        `motion` must apply exactly `z[1:] += lambda * (prior - (z[1:] - z[:-1]))`. If
        generalising to operators moved this by even a rounding step, every comparison
        against the paper's number would be against something else.
        """
        z = canonical_latent()
        prior = extract_motion_prior(canonical_latent(seed=1))
        guidance = LatentDeltaGuidance(prior, COGVIDEOX_5B, total_steps=50)

        expected = z.clone()
        expected[1:] = z[1:] + 0.05 * (prior - (z[1:] - z[:-1]))

        got = to_canonical(
            guidance.apply(from_canonical(z, COGVIDEOX_5B), 0.05), COGVIDEOX_5B
        )
        assert torch.equal(got, expected)

    @pytest.mark.parametrize("name,anchor", [("motion", 1), ("accel", 2), ("jerk", 3),
                                             ("perr", 3)])
    def test_the_leading_frames_are_never_modified(self, name, anchor):
        """The image condition depends on this, and not only for `motion`.

        An order-n operator's first value depends on frames 0..n, so guidance writes to
        z[n:] and the first n frames are the anchor. Losing this on a higher-order arm
        would silently corrupt the conditioning frame the whole benchmark is built on.
        """
        from phaselock.guidance import extract_prior

        z = canonical_latent()
        prior = extract_prior(canonical_latent(seed=1), few_step_prior_type=name)
        guidance = LatentDeltaGuidance(prior, COGVIDEOX_5B, few_step_prior_type=name, total_steps=50)

        guided = to_canonical(
            guidance.apply(from_canonical(z, COGVIDEOX_5B), 0.5), COGVIDEOX_5B
        )
        assert torch.equal(guided[:anchor], z[:anchor])
        assert not torch.equal(guided[anchor:], z[anchor:])

    @pytest.mark.parametrize("name", ["motion", "accel", "jerk", "perr"])
    def test_matching_a_trajectory_against_its_own_prior_is_a_no_op(self, name):
        """G = T(z) - T(z) = 0, so a clip already at the target must not move.

        Guards the sign of the residual: with it flipped this test still passes at
        lambda=0 but fails here, because the correction would double rather than vanish.
        """
        from phaselock.guidance import extract_prior

        z = canonical_latent()
        guidance = LatentDeltaGuidance(
            extract_prior(z, few_step_prior_type=name), COGVIDEOX_5B, few_step_prior_type=name, total_steps=50
        )
        got = to_canonical(
            guidance.apply(from_canonical(z, COGVIDEOX_5B), 0.9), COGVIDEOX_5B
        )
        assert torch.allclose(got, z, atol=1e-6)

    @pytest.mark.parametrize("name", ["motion", "accel", "jerk", "perr"])
    def test_zero_strength_is_exactly_the_unguided_latent(self, name):
        """Keeps "guidance does nothing" distinguishable from "guidance is broken"."""
        from phaselock.guidance import extract_prior

        z = canonical_latent()
        guidance = LatentDeltaGuidance(
            extract_prior(canonical_latent(seed=2), few_step_prior_type=name),
            COGVIDEOX_5B, few_step_prior_type=name, total_steps=50,
        )
        batched = from_canonical(z, COGVIDEOX_5B)
        assert torch.equal(guidance.apply(batched, 0.0), batched)

    @pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
    @pytest.mark.parametrize("name", ["motion", "accel", "jerk", "perr"])
    def test_every_operator_is_layout_correct(self, spec, name):
        """The bug this whole module exists for, extended to the new operators."""
        from phaselock.guidance import extract_prior

        z = canonical_latent()
        prior = extract_prior(from_canonical(z, spec), spec, few_step_prior_type=name)
        assert torch.equal(prior, extract_prior(z, few_step_prior_type=name))

    def test_the_residual_operator_is_fitted_per_position_not_across_the_frame(self):
        """Settles the design choice, on a case where the two provably differ.

        Per position, each (h, w) is its own C-dim trajectory. Flattening (C, H, W) into
        one long vector instead fits a single affine span for the whole frame, which is a
        different quantity. Build a latent whose positions have unrelated dynamics: the
        per-position residual sees each one exactly, the flattened one cannot.
        """
        from phaselock.metrics.geophys import residual_vectors
        from phaselock.operators import get_operator

        generator = torch.Generator().manual_seed(7)
        z = torch.randn(FRAMES, CHANNELS, HEIGHT, WIDTH, generator=generator,
                        dtype=torch.float64)

        got = get_operator("perr")(z)

        # ... equals scoring every spatial position independently.
        expected = torch.stack([
            residual_vectors(z[:, :, h, w], order=3)
            for h in range(HEIGHT) for w in range(WIDTH)
        ])
        expected = expected.reshape(HEIGHT, WIDTH, -1, CHANNELS).permute(2, 3, 0, 1)
        assert torch.allclose(got, expected, atol=1e-10)

        # ... and is NOT the flattened fit, so the choice is a real one.
        flattened = residual_vectors(z.reshape(FRAMES, -1), order=3)
        assert not torch.allclose(got.reshape(flattened.shape), flattened, atol=1e-6)


# -- the prior source -------------------------------------------------------
#
# `latent` is the sampler state. `x0_hat` and `velocity` are model *outputs*, so they
# cannot be written to directly -- but renoise is affine in (x0, eps), so a change to x0
# induces an exactly proportional change in the latent. These pin that identity, which is
# what lets the source vary without reimplementing the scheduler or CFG.


class _AffineBackend:
    """A backend whose renoise is the VP mix, enough to exercise the state path on CPU."""

    def __init__(self, spec, alpha_bar=0.36):
        self.spec = spec
        self.alpha_bar = alpha_bar

    def renoise_scale(self, timestep):
        return torch.tensor(self.alpha_bar).sqrt()

    def renoise(self, x0, eps, timestep):
        a = torch.tensor(self.alpha_bar)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def denoiser_state(self, latents, model_output, timestep):
        from phaselock.backends.base import DenoiserState

        z = to_canonical(latents, self.spec).float()
        v = to_canonical(model_output, self.spec).float()
        a = torch.tensor(self.alpha_bar)
        x0 = a.sqrt() * z - (1 - a).sqrt() * v
        eps = a.sqrt() * v + (1 - a).sqrt() * z
        return DenoiserState(latents=z, x0=x0, eps=eps, tau=0.5)


class TestPriorSource:
    def test_renoise_is_affine_so_the_induced_latent_delta_is_exact(self):
        """The identity the whole x0_hat/velocity path rests on.

        renoise(x0 + d, eps, t) - renoise(x0, eps, t) == renoise_scale(t) * d
        exactly, for any d. If this ever stops holding, guidance measured on a model
        output would be written back with the wrong magnitude and nothing would raise.
        """
        backend = _AffineBackend(COGVIDEOX_5B)
        x0 = canonical_latent(seed=3)
        eps = canonical_latent(seed=4)
        d = canonical_latent(seed=5)

        induced = backend.renoise(x0 + d, eps, 0) - backend.renoise(x0, eps, 0)
        assert torch.allclose(induced, backend.renoise_scale(0) * d, atol=1e-6)

    @pytest.mark.parametrize("source", ["x0_hat", "velocity"])
    def test_a_state_source_needs_the_backend(self, source):
        """Constructing it without one would fail later, mid-generation, per clip."""
        with pytest.raises(ValueError, match="needs"):
            LatentDeltaGuidance(
                extract_motion_prior(canonical_latent()), COGVIDEOX_5B,
                source=source, total_steps=50,
            )

    def test_an_unknown_source_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="unknown source"):
            LatentDeltaGuidance(
                extract_motion_prior(canonical_latent()), COGVIDEOX_5B,
                source="hidden_states", total_steps=50,
            )

    @pytest.mark.parametrize("source", ["x0_hat", "velocity"])
    def test_matching_a_state_against_its_own_prior_is_a_no_op(self, source):
        """The same guard as for the latent path, on the state path.

        Build a prediction, read the source off it, use that as the prior, and the
        correction must vanish -- which catches a sign error the zero-strength test
        cannot see.
        """
        backend = _AffineBackend(COGVIDEOX_5B)
        previous = from_canonical(canonical_latent(seed=6), COGVIDEOX_5B)
        model_output = from_canonical(canonical_latent(seed=7), COGVIDEOX_5B)
        state = backend.denoiser_state(previous, model_output, 0)
        measured = state.x0 if source == "x0_hat" else state.drift

        guidance = LatentDeltaGuidance(
            extract_motion_prior(measured), COGVIDEOX_5B,
            source=source, backend=backend, total_steps=50,
        )
        latents = from_canonical(canonical_latent(seed=8), COGVIDEOX_5B)
        got = guidance.apply_to_state(latents, previous, model_output, 0, 0, 0.7)
        assert torch.allclose(got, latents, atol=1e-5)

    @pytest.mark.parametrize("source", ["x0_hat", "velocity"])
    def test_the_anchor_frames_are_untouched_on_the_state_path_too(self, source):
        backend = _AffineBackend(COGVIDEOX_5B)
        previous = from_canonical(canonical_latent(seed=6), COGVIDEOX_5B)
        model_output = from_canonical(canonical_latent(seed=7), COGVIDEOX_5B)
        latents = from_canonical(canonical_latent(seed=8), COGVIDEOX_5B)

        guidance = LatentDeltaGuidance(
            extract_motion_prior(canonical_latent(seed=9)), COGVIDEOX_5B,
            source=source, backend=backend, total_steps=50,
        )
        got = to_canonical(
            guidance.apply_to_state(latents, previous, model_output, 0, 0, 0.5),
            COGVIDEOX_5B,
        )
        before = to_canonical(latents, COGVIDEOX_5B)
        assert torch.equal(got[:1], before[:1])
        assert not torch.allclose(got[1:], before[1:])

    def test_the_first_step_is_skipped_because_it_has_no_predecessor(self):
        """`noise_pred` belongs to the state that went INTO the step, which the callback
        only knows from the step before. Skipping is correct; guessing would misalign
        every subsequent correction."""
        backend = _AffineBackend(COGVIDEOX_5B)
        guidance = LatentDeltaGuidance(
            extract_motion_prior(canonical_latent()), COGVIDEOX_5B,
            source="x0_hat", backend=backend, total_steps=50,
        )
        latents = from_canonical(canonical_latent(seed=8), COGVIDEOX_5B)
        kwargs = {"latents": latents,
                  "noise_pred": from_canonical(canonical_latent(seed=7), COGVIDEOX_5B)}

        first = guidance(None, 0, torch.tensor(0.0), dict(kwargs))
        assert torch.equal(first["latents"], latents)   # no predecessor yet
        assert guidance._previous_latents is not None   # ... but it is remembered
