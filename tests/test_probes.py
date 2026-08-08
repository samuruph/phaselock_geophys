"""Probe and inversion tests. CPU only, no model weights.

Uses the oracle backend from ``conftest``, whose denoiser recovers the true clean latent
exactly. That turns the inversion loop into something with an analytic answer: invert,
resample, and you must land back on the original latent.
"""

from __future__ import annotations

import pytest
import torch

from phaselock.backends.base import from_canonical, token_grid
from phaselock.pipelines.inversion import evenly_spaced, invert, resample
from phaselock.probes import (
    HIDDEN_STATES,
    LATENT,
    NO_BLOCK,
    SOURCES,
    VELOCITY,
    X0_HAT,
    ProbeRecord,
    ProbeRecorder,
    TrajectoryKey,
    pool_latents,
    pool_tokens,
    resolve_blocks,
    supports_exact_drift,
    validate_sources,
)


# -- pooling ----------------------------------------------------------------


def test_pool_tokens_groups_each_latent_frame_separately():
    """The reshape is only exact if each frame owns a contiguous run of tokens."""
    frames, height, width, dim = 5, 3, 4, 8
    tokens = torch.zeros(1, frames * height * width, dim)
    for frame in range(frames):
        start = frame * height * width
        tokens[0, start : start + height * width] = float(frame)

    pooled = pool_tokens(tokens, (frames, height, width))
    assert pooled.shape == (frames, dim)
    assert torch.allclose(pooled[:, 0], torch.arange(frames, dtype=torch.float32))


def test_pool_tokens_rejects_a_token_count_that_does_not_match_the_grid():
    """Catches the case where text tokens were not stripped from the sequence."""
    with pytest.raises(ValueError, match="does not match grid"):
        pool_tokens(torch.zeros(1, 226 + 60, 8), (5, 3, 4))


def test_pool_tokens_flatten_keeps_every_spatial_position():
    pooled = pool_tokens(torch.randn(1, 5 * 12, 8), (5, 3, 4), mode="flatten")
    assert pooled.shape == (5, 12 * 8)


def test_pool_latents_averages_over_space_only():
    latents = torch.randn(5, 4, 6, 6)
    pooled = pool_latents(latents)
    assert pooled.shape == (5, 4)
    assert torch.allclose(pooled, latents.mean(dim=(2, 3)))


def test_pooling_is_linear_which_the_flow_metrics_depend_on():
    """Exact geometric drift needs pooling to commute with the ODE, i.e. be linear.

    Run in float64: linearity is an algebraic property, and checking it in float32 would
    be testing the associativity of floating-point summation instead.
    """
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    a, b = randn(5, 4, 6, 6), randn(5, 4, 6, 6)
    assert torch.allclose(pool_latents(a + 2.5 * b), pool_latents(a) + 2.5 * pool_latents(b))

    grid = (5, 3, 4)
    ta, tb = randn(1, 60, 8), randn(1, 60, 8)
    assert torch.allclose(
        pool_tokens(ta + 2.5 * tb, grid), pool_tokens(ta, grid) + 2.5 * pool_tokens(tb, grid)
    )


def test_unknown_pool_mode_raises():
    with pytest.raises(ValueError, match="unknown pool mode"):
        pool_latents(torch.randn(5, 4, 6, 6), mode="max")


def test_resolve_blocks_always_includes_the_last_block():
    assert resolve_blocks(42, None, stride=8) == [0, 8, 16, 24, 32, 40, 41]
    assert resolve_blocks(6, [0, 3]) == [0, 3]
    with pytest.raises(ValueError, match="out of range"):
        resolve_blocks(6, [0, 9])


# -- source registry --------------------------------------------------------


def test_only_the_latent_supports_exact_drift():
    """Exact drift needs dr/dtau = u, which holds only when r is the ODE state itself.

    ``x0_hat`` and ``velocity`` look like they should qualify, but both are functions of
    the network's output as well as the state, so differentiating them in tau drags in a
    Jacobian of the transformer that is never formed. Hidden states are further removed
    still. Everything but the latent falls back to the finite-difference estimator.
    """
    assert supports_exact_drift(LATENT)
    assert not supports_exact_drift(X0_HAT)
    assert not supports_exact_drift(VELOCITY)
    assert not supports_exact_drift(HIDDEN_STATES)
    assert not supports_exact_drift("attention")


def test_validate_sources_deduplicates_and_rejects_unknown():
    assert validate_sources([LATENT, LATENT, HIDDEN_STATES]) == [LATENT, HIDDEN_STATES]
    with pytest.raises(ValueError, match="unknown probe sources"):
        validate_sources(["latents"])
    with pytest.raises(ValueError, match="at least one source"):
        validate_sources([])


def test_every_source_documents_itself():
    for name, source in SOURCES.items():
        assert source.name == name
        assert source.description.strip()


# -- record storage ---------------------------------------------------------


def test_probe_record_roundtrips_through_disk(tmp_path):
    record = ProbeRecord()
    record.add(TrajectoryKey(HIDDEN_STATES, 3, 0), torch.randn(5, 16))
    record.add(TrajectoryKey(LATENT, NO_BLOCK, 0), torch.randn(5, 4))
    record.taus[0] = 0.5
    record.provenance["backend"] = "oracle"

    record.save(tmp_path / "probe")
    loaded = ProbeRecord.load(tmp_path / "probe")

    assert loaded.sources == record.sources
    assert loaded.taus == {0: 0.5}
    assert loaded.provenance["backend"] == "oracle"
    assert torch.allclose(loaded.get(LATENT, 0), record.get(LATENT, 0))


def test_probe_record_refuses_to_overwrite_a_trajectory():
    record = ProbeRecord()
    key = TrajectoryKey(LATENT, NO_BLOCK, 0)
    record.add(key, torch.randn(5, 4))
    with pytest.raises(KeyError, match="recorded twice"):
        record.add(key, torch.randn(5, 4))


def test_probe_record_reports_a_helpful_error_for_a_missing_trajectory():
    record = ProbeRecord()
    record.add(TrajectoryKey(LATENT, NO_BLOCK, 0), torch.randn(5, 4))
    with pytest.raises(KeyError, match="recorded sources are"):
        record.get(HIDDEN_STATES, 0, block=3)


def test_trajectory_key_string_roundtrip():
    key = TrajectoryKey(HIDDEN_STATES, 17, 4)
    assert TrajectoryKey.from_string(key.to_string()) == key


# -- recorder ---------------------------------------------------------------


def test_recorder_captures_every_requested_source(oracle, target_latent):
    grid = token_grid(target_latent.shape, oracle.spec)
    sources = [HIDDEN_STATES, LATENT, X0_HAT, VELOCITY]

    with ProbeRecorder(oracle, sources=sources, grid=grid, block_stride=2) as probe:
        timestep = torch.tensor(500.0)
        output = oracle.transformer_forward(
            from_canonical(target_latent, oracle.spec), timestep, {}
        )
        state = oracle.denoiser_state(
            from_canonical(target_latent, oracle.spec), output, timestep
        )
        probe.capture(0, state)
        record = probe.record

    assert set(record.sources) == set(sources)
    assert record.blocks(HIDDEN_STATES) == [0, 2, 4, 5]
    assert record.blocks(LATENT) == [NO_BLOCK]
    assert record.get(LATENT, 0).shape == (target_latent.shape[0], oracle.spec.channels)
    assert record.get(HIDDEN_STATES, 0, block=0).shape[0] == target_latent.shape[0]


def test_recorder_hooks_are_removed_on_exit(oracle, target_latent):
    """A leaked hook would keep pooling on every later forward pass in the process."""
    grid = token_grid(target_latent.shape, oracle.spec)
    with ProbeRecorder(oracle, sources=[HIDDEN_STATES], grid=grid) as probe:
        assert probe._handles
    assert not probe._handles
    for block in oracle.blocks:
        assert not block._forward_hooks


def test_disarming_stops_hooks_doing_work(oracle, target_latent):
    grid = token_grid(target_latent.shape, oracle.spec)
    with ProbeRecorder(oracle, sources=[HIDDEN_STATES], grid=grid) as probe:
        probe.disarm()
        oracle.transformer_forward(from_canonical(target_latent, oracle.spec), torch.tensor(500.0), {})
        assert not probe._staged

        probe.arm()
        oracle.transformer_forward(from_canonical(target_latent, oracle.spec), torch.tensor(500.0), {})
        assert probe._staged


def test_recorder_rejects_capturing_the_same_step_twice(oracle, target_latent):
    grid = token_grid(target_latent.shape, oracle.spec)
    with ProbeRecorder(oracle, sources=[LATENT], grid=grid) as probe:
        timestep = torch.tensor(500.0)
        output = oracle.transformer_forward(from_canonical(target_latent, oracle.spec), timestep, {})
        state = oracle.denoiser_state(from_canonical(target_latent, oracle.spec), output, timestep)
        probe.capture(0, state)
        with pytest.raises(KeyError, match="already captured"):
            probe.capture(0, state)


def test_recorder_skips_block_bookkeeping_when_no_source_needs_it(oracle, target_latent):
    grid = token_grid(target_latent.shape, oracle.spec)
    probe = ProbeRecorder(oracle, sources=[LATENT], grid=grid)
    assert probe.blocks == []


# -- inversion --------------------------------------------------------------


def test_evenly_spaced_covers_both_ends():
    assert evenly_spaced(50, 10)[0] == 0
    assert evenly_spaced(50, 10)[-1] == 49
    assert len(evenly_spaced(50, 10)) == 10
    assert evenly_spaced(5, 10) == list(range(5))


def test_invert_then_resample_recovers_the_original_latent(oracle, target_latent):
    """The round trip is exact for an oracle denoiser, so any loop error is fatal here.

    This is what catches a reversed schedule, a wrong renoise formula, or a canonical
    layout conversion applied in the wrong direction -- all of which would otherwise
    just degrade a real reconstruction slightly and look like normal approximation error.
    """
    result = invert(
        oracle,
        frames=None,
        num_steps=20,
        record_steps=5,
        sources=[LATENT, X0_HAT, VELOCITY],
        latents=target_latent,
    )
    recovered = resample(oracle, result.noise, num_steps=20)
    assert torch.allclose(recovered, target_latent, atol=1e-4)


def test_inversion_walks_from_data_towards_noise(oracle, target_latent):
    """tau must decrease across recorded steps: inversion moves away from clean data."""
    result = invert(
        oracle, frames=None, num_steps=20, record_steps=5, sources=[LATENT], latents=target_latent
    )
    taus = [result.record.taus[step] for step in sorted(result.record.taus)]
    assert taus == sorted(taus, reverse=True)
    assert taus[0] > 0.9 and taus[-1] < 0.1


def test_inversion_records_exactly_the_requested_number_of_steps(oracle, target_latent):
    result = invert(
        oracle, frames=None, num_steps=40, record_steps=8, sources=[LATENT], latents=target_latent
    )
    assert len(result.record.steps) == 8
    assert len(result.timesteps) == 8


def test_inversion_evaluates_the_model_once_per_step(oracle, target_latent):
    """Recording must cost storage, not extra network evaluations."""
    oracle.forward_calls = 0
    invert(oracle, frames=None, num_steps=25, record_steps=5, sources=[LATENT], latents=target_latent)
    assert oracle.forward_calls == 25


def test_inversion_writes_provenance(oracle, target_latent):
    result = invert(
        oracle,
        frames=None,
        num_steps=10,
        record_steps=3,
        sources=[LATENT],
        latents=target_latent,
        provenance={"sample_id": "abc"},
    )
    provenance = result.record.provenance
    assert provenance["direction"] == "inversion"
    assert provenance["backend"] == "oracle"
    assert provenance["num_steps"] == 10
    assert provenance["sample_id"] == "abc"


# -- the integration loop actually integrates -------------------------------


def test_inversion_follows_the_analytic_path_and_does_not_stand_still(constant_backend):
    """Catches the identity bug: renoising to the level just evaluated at is a no-op.

    ``(x0, eps)`` are derived from ``z`` at level ``s``, so recombining them at ``s``
    reconstructs ``z`` exactly. A loop targeting the current level instead of the next
    never advances -- and with an oracle denoiser it still round-trips, because resampling
    a clean latent lands back on the target. Only a fixed ``(x0, eps)`` makes the targeted
    level observable.
    """
    num_steps = 10
    result = invert(
        constant_backend,
        frames=None,
        latents=constant_backend.fixed_x0,
        num_steps=num_steps,
        record_steps=3,
        sources=[LATENT],
    )

    # The recovered noise must sit at the top of the ladder, on the analytic path.
    top = max(constant_backend.levels_seen)
    scheduler = constant_backend.pipe.scheduler
    scheduler.set_timesteps(num_steps)
    highest = float(scheduler.timesteps.max()) / 1000.0
    expected = (1 - highest) * constant_backend.fixed_x0 + highest * constant_backend.fixed_eps

    assert torch.allclose(result.noise, expected, atol=1e-5), "did not land on the top level"
    assert not torch.allclose(result.noise, constant_backend.fixed_x0, atol=1e-2), (
        "inversion left the latent where it started"
    )
    # It climbed: the first level evaluated is the clean end, the last the noisy one.
    assert constant_backend.levels_seen[0] < constant_backend.levels_seen[-1]
    # Under the explicit scheme evaluation lags the target by one rung, so the highest
    # level is only ever *targeted*, never evaluated at.
    assert top < highest


def test_inversion_evaluates_the_clean_end_first(constant_backend):
    """The explicit scheme evaluates at the known endpoint, which starts at sigma = 0."""
    invert(
        constant_backend, frames=None, latents=constant_backend.fixed_x0,
        num_steps=8, record_steps=2, sources=[LATENT],
    )
    assert constant_backend.levels_seen[0] == pytest.approx(0.0)
    assert constant_backend.levels_seen == sorted(constant_backend.levels_seen)
