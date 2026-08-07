"""Dataset tests, run against the real manifests on disk.

No video is decoded, so these stay fast. They exist to catch the failure mode that
matters most here: a loader that silently mis-pairs or silently drops samples still
produces plausible-looking numbers, and nothing downstream would notice.

Every test skips cleanly when the dataset is not present, so the suite runs anywhere.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest
import torch

from phaselock.datasets import (
    PAIRED_DATASETS,
    IntPhys2,
    LikePhys,
    PhysicsIQ,
    gaussian_blur,
    get_dataset,
    get_paired_dataset,
    letterbox,
    resample_frames,
    temporal_window,
)
from phaselock.datasets.likephys import SCENARIO_PROMPTS

LIKEPHYS_ROOT = Path("/data/datasets/LikePhys-Benchmark/data")
INTPHYS2_ROOT = Path("/data/datasets/IntPhys2")
PHYSICS_IQ_ROOT = Path("/data/datasets/physics-IQ-benchmark-verified")

needs_likephys = pytest.mark.skipif(not LIKEPHYS_ROOT.is_dir(), reason="LikePhys not present")
needs_intphys2 = pytest.mark.skipif(not INTPHYS2_ROOT.is_dir(), reason="IntPhys2 not present")
needs_physics_iq = pytest.mark.skipif(not PHYSICS_IQ_ROOT.is_dir(), reason="Physics-IQ not present")


@pytest.fixture(scope="module")
def likephys() -> LikePhys:
    return LikePhys()


@pytest.fixture(scope="module")
def intphys2() -> IntPhys2:
    return IntPhys2()


@pytest.fixture(scope="module")
def physics_iq() -> PhysicsIQ:
    return PhysicsIQ()


# -- LikePhys ---------------------------------------------------------------


@needs_likephys
def test_likephys_pair_count(likephys):
    """12 scenarios x 10 subgroups x (5-8 violations) = 800 pairs in this release.

    The paper cites 650, so this is a different release and absolute accuracies will
    not match published ones.
    """
    assert len(likephys.pairs()) == 800
    assert len(likephys.scenarios) == 12
    assert len(likephys.valid_clips()) == 120


@needs_likephys
def test_likephys_pairs_are_well_formed(likephys):
    for pair in likephys.pairs():
        assert pair.plausible.label == 0 and pair.violated.label == 1
        assert pair.plausible.group == pair.violated.group
        assert pair.violated.meta["kind"] == pair.violation
        assert Path(pair.plausible.path).exists()
        assert Path(pair.violated.path).exists()


@needs_likephys
def test_likephys_violated_clips_are_never_reused_across_pairs(likephys):
    """Each violation belongs to exactly one pair; only the valid clip is shared."""
    ids = [pair.violated.sample_id for pair in likephys.pairs()]
    assert len(ids) == len(set(ids))


@needs_likephys
def test_likephys_pairs_share_a_subgroup(likephys):
    """A violation must be paired with its own subgroup's valid clip, not another's."""
    for pair in likephys.pairs():
        assert pair.plausible.meta["subgroup"] == pair.violated.meta["subgroup"]
        assert pair.plausible.meta["scenario"] == pair.violated.meta["scenario"]


@needs_likephys
def test_every_likephys_scenario_has_a_prompt(likephys):
    """Stage 6 conditions on these; a missing one would silently become an empty prompt."""
    for scenario in likephys.scenarios:
        assert likephys.prompt_for(scenario).strip()
    assert set(SCENARIO_PROMPTS) == set(likephys.scenarios)


@needs_likephys
def test_likephys_unknown_prompt_raises(likephys):
    with pytest.raises(KeyError, match="no prompt registered"):
        likephys.prompt_for("not_a_scenario")


# -- IntPhys2 ---------------------------------------------------------------


@needs_intphys2
def test_intphys2_pair_count_matches_the_published_506(intphys2):
    assert len(intphys2.pairs()) == 506
    assert len(intphys2.samples()) == 1012
    assert set(intphys2.scenarios) == {"continuity", "immutability", "permanence", "solidity"}


@needs_intphys2
def test_intphys2_pairs_are_well_formed(intphys2):
    for pair in intphys2.pairs():
        assert pair.plausible.label == 0 and pair.violated.label == 1
        assert pair.plausible.meta["scene_index"] == pair.violated.meta["scene_index"]
        assert pair.plausible.meta["run"] == pair.violated.meta["run"]
        assert Path(pair.plausible.path).exists()
        assert Path(pair.violated.path).exists()


@needs_intphys2
def test_intphys2_every_video_belongs_to_exactly_one_pair(intphys2):
    ids = [s.sample_id for pair in intphys2.pairs() for s in (pair.plausible, pair.violated)]
    assert len(ids) == len(set(ids))


@needs_intphys2
def test_intphys2_difficulty_and_camera_filters(intphys2):
    moving = intphys2.select_by(camera="Moving")
    fixed = intphys2.select_by(camera="Fixed")
    assert len(moving) + len(fixed) == len(intphys2.pairs())
    assert all(p.plausible.meta["camera"] == "Moving" for p in moving)

    with pytest.raises(ValueError, match="unknown camera"):
        intphys2.select_by(camera="Handheld")
    with pytest.raises(ValueError, match="unknown difficulty"):
        intphys2.select_by(difficulty="Trivial")


@needs_intphys2
def test_intphys2_rejects_an_unlabelled_split():
    with pytest.raises(FileNotFoundError, match="metadata"):
        IntPhys2(split="HeldOut")

    with pytest.raises(ValueError, match="split must be"):
        IntPhys2(split="Train")


# -- Physics-IQ -------------------------------------------------------------


@needs_physics_iq
def test_physics_iq_keeps_only_the_198_benchmark_scenarios(physics_iq):
    """The CSV holds 396 rows; take 2 is the noise floor, not an evaluation scenario."""
    samples = physics_iq.generation_samples()
    assert len(samples) == 198
    assert len({s.sample_id for s in samples}) == 198


@needs_physics_iq
def test_physics_iq_resolves_every_auxiliary_asset(physics_iq):
    """Each asset type has its own filename infix; a wrong one yields silent Nones."""
    for sample in physics_iq.generation_samples():
        assert sample.image_path and Path(sample.image_path).exists()
        assert sample.reference_path and Path(sample.reference_path).exists()
        assert sample.mask_path and Path(sample.mask_path).exists()
        assert sample.meta["conditioning_path"] and Path(sample.meta["conditioning_path"]).exists()


@needs_physics_iq
def test_physics_iq_take_two_pair_has_a_different_index(physics_iq):
    """Take 2 lives at index+198, so a string substitution on the take field is wrong."""
    for sample in physics_iq.generation_samples():
        pair_path = sample.meta["pair_path"]
        assert pair_path and Path(pair_path).exists()
        assert "take-2" in Path(pair_path).name
        assert Path(pair_path).name[:4] != sample.sample_id


@needs_physics_iq
def test_physics_iq_output_name_is_the_evaluator_filename(physics_iq):
    """Outputs must be named for the official evaluator, not for the switch frame."""
    for sample in physics_iq.generation_samples():
        name = sample.meta["output_name"]
        assert name.endswith(".mp4") and "take-" not in name
        assert name.startswith(sample.sample_id)


@needs_physics_iq
def test_physics_iq_rejects_an_unavailable_frame_rate():
    with pytest.raises(ValueError, match="mask_fps must be one of"):
        PhysicsIQ(mask_fps=12)


# -- selection --------------------------------------------------------------


@needs_likephys
def test_balanced_selection_spreads_across_scenarios(likephys):
    """The first N pairs are all one scenario; a pilot run must not measure only that."""
    chosen = likephys.select(limit=24)
    counts = collections.Counter(pair.scenario for pair in chosen)
    assert len(chosen) == 24
    assert set(counts.values()) == {2}


@needs_likephys
def test_selection_is_deterministic_given_a_seed(likephys):
    first = [p.violated.sample_id for p in likephys.select(limit=24, seed=7)]
    second = [p.violated.sample_id for p in likephys.select(limit=24, seed=7)]
    assert first == second
    assert first != [p.violated.sample_id for p in likephys.select(limit=24, seed=8)]


@needs_likephys
def test_unknown_filter_values_raise_instead_of_returning_nothing(likephys):
    """Silently returning zero pairs reads as "no signal", which is far worse."""
    with pytest.raises(ValueError, match="unknown scenarios"):
        likephys.select(scenarios=["ball_drop", "typo"])
    with pytest.raises(ValueError, match="unknown violations"):
        likephys.select(violations=["not_a_violation"])


@needs_likephys
def test_limit_above_the_pair_count_returns_everything(likephys):
    assert len(likephys.select(scenarios=["ball_drop"], limit=10_000)) == 60


# -- registry ---------------------------------------------------------------


def test_registry_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("nope")
    with pytest.raises(KeyError, match="not a paired dataset"):
        get_paired_dataset("physics_iq")
    assert set(PAIRED_DATASETS) == {"likephys", "intphys2"}


# -- preprocessing ----------------------------------------------------------


def test_resample_hits_the_requested_length_in_both_directions():
    frames = torch.arange(60, dtype=torch.float32).view(60, 1, 1, 1).expand(60, 3, 4, 4)
    assert resample_frames(frames, 49).shape[0] == 49
    assert resample_frames(frames, 81).shape[0] == 81
    assert torch.equal(resample_frames(frames, 60), frames)


def test_resample_keeps_the_endpoints():
    """Dropping the first or last frame would shift every velocity in the clip."""
    frames = torch.arange(60, dtype=torch.float32).view(60, 1, 1, 1)
    out = resample_frames(frames, 49)
    assert float(out[0]) == 0.0 and float(out[-1]) == 59.0


def test_resample_does_not_interpolate_between_frames():
    """Blending neighbours would manufacture motion blur the statistics would read."""
    frames = torch.arange(60, dtype=torch.float32).view(60, 1, 1, 1)
    values = resample_frames(frames, 49).flatten().tolist()
    assert all(float(v).is_integer() for v in values)


def test_letterbox_preserves_aspect_ratio_and_pads():
    frames = torch.ones(4, 3, 512, 512)
    out = letterbox(frames, 480, 720)
    assert out.shape == (4, 3, 480, 720)
    # A square source in a wide frame leaves black bars at the left and right edges.
    assert float(out[:, :, :, 0].max()) == 0.0
    assert float(out[:, :, :, -1].max()) == 0.0
    assert float(out[0, 0, 240, 360]) == pytest.approx(1.0)


def test_temporal_window_centres_and_validates():
    frames = torch.arange(100, dtype=torch.float32).view(100, 1, 1, 1)
    out = temporal_window(frames, 0.5)
    assert out.shape[0] == 50 and float(out[0]) == 25.0
    assert torch.equal(temporal_window(frames, None), frames)
    with pytest.raises(ValueError, match="window must be"):
        temporal_window(frames, -0.5)


def test_gaussian_blur_is_a_no_op_at_zero_and_conserves_mean():
    frames = torch.rand(3, 3, 64, 64)
    assert torch.equal(gaussian_blur(frames, 0.0), frames)

    blurred = gaussian_blur(frames, 8.0)
    assert blurred.shape == frames.shape
    # Reflect padding keeps the mean; heavy blur must reduce variance.
    assert float(blurred.mean()) == pytest.approx(float(frames.mean()), abs=1e-2)
    assert float(blurred.var()) < float(frames.var())


def test_stronger_blur_removes_more_detail():
    """Ordering matters: the blur control sweeps sigma and reads off the trend."""
    frames = torch.rand(2, 3, 96, 96)
    variances = [float(gaussian_blur(frames, s).var()) for s in (0.0, 8.0, 16.0)]
    assert variances[0] > variances[1] > variances[2]
