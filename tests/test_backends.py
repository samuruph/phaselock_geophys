"""Backend tests. CPU only, no weights loaded.

The layout and normalisation conventions differ between CogVideoX and Wan in ways that
fail silently rather than loudly: differentiating a BCTHW tensor along the CogVideoX
frame axis differentiates channels instead, and it still returns a tensor of a plausible
shape. These tests pin the conventions so that class of bug cannot survive.
"""

from __future__ import annotations

import pytest
import torch

from phaselock.backends import (
    BACKENDS,
    COGVIDEOX_5B,
    WAN21_I2V_14B_720P,
    WAN21_T2V_1_3B,
    LatentSpec,
    denormalize,
    from_canonical,
    get_entry,
    get_spec,
    normalize,
    num_latent_frames,
    to_canonical,
    token_grid,
)

SPECS = [COGVIDEOX_5B, WAN21_T2V_1_3B]
SPEC_IDS = [spec.name for spec in SPECS]


# -- registry ---------------------------------------------------------------


def test_registry_contains_the_expected_backends():
    assert set(BACKENDS) == {
        "cogvideox_5b_t2v",
        "cogvideox_5b_i2v",
        "wan21_t2v_1_3b",
        "wan21_t2v_14b",
        "wan21_i2v_14b_480p",
        "wan21_i2v_14b_720p",
        "wan22_i2v_a14b",
        "wan22_ti2v_5b",
    }


def test_oversized_checkpoints_are_flagged_unvalidated():
    """The 14B entries are wired but cannot be exercised on a 46GB card."""
    assert get_entry("cogvideox_5b_t2v").validated
    assert get_entry("wan21_t2v_1_3b").validated
    assert not get_entry("wan21_t2v_14b").validated
    assert not get_entry("wan21_i2v_14b_480p").validated
    assert not get_entry("wan22_i2v_a14b").validated
    assert not get_entry("wan22_ti2v_5b").validated


def test_unknown_backend_names_raise_with_the_available_list():
    with pytest.raises(KeyError, match="unknown backend"):
        get_spec("cogvideox_7b")


# -- layout -----------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_canonical_layout_roundtrips(spec):
    z = torch.randn(13, spec.channels, 12, 18)
    assert torch.equal(to_canonical(from_canonical(z, spec), spec), z)


def test_the_two_backends_really_do_disagree_about_layout():
    """If this ever passes trivially, the layout handling has stopped doing anything."""
    z = torch.randn(13, 16, 12, 18)
    cog = from_canonical(z, COGVIDEOX_5B)
    wan = from_canonical(z, WAN21_T2V_1_3B)
    assert cog.shape == (1, 13, 16, 12, 18)
    assert wan.shape == (1, 16, 13, 12, 18)
    assert cog.shape != wan.shape


def test_frame_differencing_picks_the_right_axis_under_both_layouts():
    """The bug this guards: unpacking B,T,C,H,W from a BCTHW tensor differences channels."""
    frames = 13
    z = torch.zeros(frames, 16, 4, 4)
    z[5] = 1.0  # a single frame differs

    for spec in SPECS:
        canonical = to_canonical(from_canonical(z, spec), spec)
        delta = canonical[1:] - canonical[:-1]
        # Exactly two frame transitions change: into frame 5 and out of it.
        nonzero = [int(i) for i in range(delta.shape[0]) if float(delta[i].abs().sum()) > 0]
        assert nonzero == [4, 5], f"{spec.name} differenced the wrong axis"


def test_to_canonical_rejects_a_real_batch():
    """Canonical form is unbatched; silently taking element 0 would corrupt results."""
    with pytest.raises(ValueError, match="unbatched"):
        to_canonical(torch.randn(4, 13, 16, 12, 18), COGVIDEOX_5B)


def test_to_canonical_passes_through_already_canonical_input():
    z = torch.randn(13, 16, 12, 18)
    assert torch.equal(to_canonical(z, COGVIDEOX_5B), z)


# -- normalisation ----------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_normalisation_roundtrips(spec):
    z = torch.randn(13, spec.channels, 12, 18)
    assert torch.allclose(denormalize(normalize(z, spec), spec), z, atol=1e-5)


def test_wan_normalisation_is_per_channel_not_scalar():
    """Wan's VAE has no meaningful scaling_factor; using one would skew every channel."""
    assert WAN21_T2V_1_3B.scaling_factor is None
    assert WAN21_T2V_1_3B.latents_mean is not None

    z = torch.zeros(2, 16, 3, 3)
    normalised = normalize(z, WAN21_T2V_1_3B)
    per_channel = normalised[0, :, 0, 0]
    # A constant input maps to -mean/std, which differs across channels.
    assert float(per_channel.std()) > 0.1


def test_cogvideox_normalisation_is_the_scalar_scaling_factor():
    assert COGVIDEOX_5B.scaling_factor == pytest.approx(0.7)
    z = torch.randn(2, 16, 3, 3)
    assert torch.allclose(normalize(z, COGVIDEOX_5B), z * 0.7)


def test_wan_constants_match_the_shipped_checkpoint():
    """Guards against silently drifting from the VAE config the weights were trained with."""
    import json
    from pathlib import Path

    config_path = Path("/data/weights/Wan2.1-T2V-1.3B-Diffusers/vae/config.json")
    if not config_path.is_file():
        pytest.skip("Wan checkpoint not present")

    config = json.loads(config_path.read_text())
    assert list(WAN21_T2V_1_3B.latents_mean) == pytest.approx(config["latents_mean"])
    assert list(WAN21_T2V_1_3B.latents_std) == pytest.approx(config["latents_std"])
    assert config.get("scaling_factor") is None


# -- frame and token geometry ----------------------------------------------


def test_causal_vae_frame_mapping():
    """Native frame counts must land on whole latent frames."""
    assert num_latent_frames(49, COGVIDEOX_5B) == 13
    assert num_latent_frames(81, WAN21_T2V_1_3B) == 21
    assert num_latent_frames(1, COGVIDEOX_5B) == 1


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_native_frame_count_is_representable(spec):
    assert (spec.default_num_frames - 1) % spec.temporal_ratio == 0


def test_token_grid_matches_the_dit_patch():
    """Per-latent-frame pooling depends on patch_t == 1 giving one token row per frame."""
    grid = token_grid((13, 16, 60, 90), COGVIDEOX_5B)
    assert grid == (13, 30, 45)
    for spec in SPECS:
        assert spec.patch_size[0] == 1, "temporal patching would break per-frame pooling"


# -- spec validation --------------------------------------------------------


def _spec(**overrides):
    base = dict(
        name="test",
        layout="BTCHW",
        temporal_ratio=4,
        spatial_ratio=8,
        channels=16,
        patch_size=(1, 2, 2),
        default_num_frames=49,
        default_fps=8,
        default_height=480,
        default_width=720,
        scaling_factor=0.7,
    )
    base.update(overrides)
    return LatentSpec(**base)


def test_spec_requires_exactly_one_normalisation_convention():
    with pytest.raises(ValueError, match="exactly one normalisation"):
        _spec(latents_mean=(0.0,) * 16, latents_std=(1.0,) * 16)
    with pytest.raises(ValueError, match="exactly one normalisation"):
        _spec(scaling_factor=None)


def test_spec_rejects_a_mismatched_channel_count():
    with pytest.raises(ValueError, match="must have 16 entries"):
        _spec(scaling_factor=None, latents_mean=(0.0,) * 8, latents_std=(1.0,) * 8)


def test_spec_rejects_an_unknown_layout():
    with pytest.raises(ValueError, match="layout must be"):
        _spec(layout="TCHW")


def test_spec_rejects_a_frame_count_the_causal_vae_cannot_represent():
    with pytest.raises(ValueError, match="k\\*4\\+1"):
        _spec(default_num_frames=50)


def test_720p_wan_carries_its_own_flow_shift():
    """The 720P checkpoint is trained at a different shift; sharing one would degrade it."""
    assert WAN21_I2V_14B_720P.flow_shift != WAN21_T2V_1_3B.flow_shift
    assert (WAN21_I2V_14B_720P.default_height, WAN21_I2V_14B_720P.default_width) == (720, 1280)
