"""Config tests.

The point of this module is that a typo raises. A config key that silently falls back to
its default produces a plausible result under settings nobody chose, and after a ten-hour
run there is no way to tell.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phaselock.config import Config, load, parse_overrides

CONFIGS = sorted(Path(__file__).parent.parent.glob("configs/experiments/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_shipped_config_loads(path):
    """A config that no longer parses is a broken experiment nobody notices until launch."""
    config = load(path)
    assert config.backend.name
    assert config.data.name


def test_defaults_load_without_a_file():
    config = load()
    assert config.backend.name == "cogvideox_5b_t2v"
    assert config.metrics.residual_fit == "span"
    assert config.inversion.prompt == ""


def test_nested_sections_become_dataclasses_not_dicts():
    """`from __future__ import annotations` makes field.type a string; hints must resolve."""
    config = load(CONFIGS[0])
    assert not isinstance(config.backend, dict)
    assert isinstance(config, Config)


# -- overrides --------------------------------------------------------------


def test_string_overrides_are_coerced_to_the_declared_type():
    config = load(**parse_overrides(
        ["data__limit=48", "inversion__num_steps=100", "data__window=0.25", "backend__offload=false"]
    ))
    assert config.data.limit == 48 and isinstance(config.data.limit, int)
    assert config.inversion.num_steps == 100
    assert config.data.window == 0.25 and isinstance(config.data.window, float)
    assert config.backend.offload is False


def test_comma_separated_overrides_become_lists():
    config = load(**parse_overrides(["probe__sources=latent,velocity"]))
    assert config.probe.sources == ["latent", "velocity"]


def test_none_overrides_are_ignored_so_argparse_defaults_pass_through():
    """Lets a CLI forward every flag unconditionally without clobbering the file."""
    config = load(CONFIGS[0], data__limit=None)
    assert config.data.limit == load(CONFIGS[0]).data.limit


def test_null_override_clears_a_value():
    assert load(**parse_overrides(["data__window=none"])).data.window is None


# -- rejection --------------------------------------------------------------


def test_unknown_override_key_raises():
    with pytest.raises(ValueError, match="unknown key 'limitt'"):
        load(**parse_overrides(["data__limitt=5"]))


def test_unknown_override_section_raises():
    with pytest.raises(ValueError, match="unknown config section"):
        load(**parse_overrides(["nosuch__x=1"]))


def test_override_without_the_section_separator_raises():
    with pytest.raises(ValueError, match="section__key"):
        load(**parse_overrides(["data.limit=5"]))


def test_unknown_yaml_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  limit: 5\n  typo_key: 3\n")
    with pytest.raises(ValueError, match="unknown configuration key"):
        load(path)


def test_unknown_yaml_section_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("nosection:\n  a: 1\n")
    with pytest.raises(ValueError, match="unknown configuration key"):
        load(path)


def test_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping at the top level"):
        load(path)


def test_empty_yaml_yields_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load(path).backend.name == Config().backend.name


# -- conveniences -----------------------------------------------------------


def test_a_scalar_stands_in_for_a_single_element_list(tmp_path):
    path = tmp_path / "one.yaml"
    path.write_text("data:\n  scenarios: ball_drop\n")
    assert load(path).data.scenarios == ["ball_drop"]


def test_output_dir_is_created_on_demand(tmp_path):
    config = load(**parse_overrides([f"output__root={tmp_path}", "output__name=run1"]))
    assert config.output.dir("trajectories").is_dir()
    # Under the defaults, backend and dataset segments come from the config itself.
    expected = tmp_path / config.backend.name / config.data.name / "run1" / "trajectories"
    assert expected.is_dir()


def test_config_serialises_for_the_run_record(tmp_path):
    config = load(CONFIGS[0])
    path = tmp_path / "config.json"
    config.save(path)
    assert '"residual_fit"' in path.read_text()


def test_parse_overrides_requires_an_equals_sign():
    with pytest.raises(ValueError, match="section__key=value"):
        parse_overrides(["data__limit"])


def test_results_are_filed_under_backend_and_dataset(tmp_path):
    """A run directory carries its own provenance.

    Every number in this project is only meaningful against the model and dataset that
    produced it, so the path encodes both rather than relying on someone naming the run
    carefully.
    """
    from phaselock.config import load

    config = load(None, backend__name="wan21_t2v_1_3b", data__name="likephys",
                  output__name="detection", output__root=str(tmp_path))
    assert config.output.base() == tmp_path / "wan21_t2v_1_3b" / "likephys" / "detection"


def test_output_segments_are_not_overridden_when_set_explicitly(tmp_path):
    from phaselock.config import load

    config = load(None, backend__name="wan21_t2v_1_3b", data__name="likephys",
                  output__backend="custom", output__dataset="other",
                  output__name="run", output__root=str(tmp_path))
    assert config.output.base() == tmp_path / "custom" / "other" / "run"


def test_frame_geometry_defaults_to_the_backend_spec():
    from phaselock.backends import get_spec
    from phaselock.experiments.detection import frame_geometry

    spec = get_spec("wan21_t2v_1_3b")
    assert frame_geometry(spec, load(None)) == (
        spec.default_num_frames, spec.default_height, spec.default_width
    )


def test_frame_geometry_honours_a_resolution_override():
    """A square source wastes 42% of the token grid on black under Wan's 480x832.

    Overriding to the source's own 512x512 removes the padding and, measured, a 40% of
    the per-clip cost with it.
    """
    from phaselock.backends import get_spec
    from phaselock.experiments.detection import frame_geometry

    spec = get_spec("wan21_t2v_1_3b")
    config = load(None, data__height="512", data__width="512")
    frames, height, width = frame_geometry(spec, config)
    assert (height, width) == (512, 512)
    assert frames == spec.default_num_frames, "frame count is a hard 4k+1 constraint"


def test_resolution_override_must_stay_legal_for_the_patch_grid():
    """Both backends patchify 2x2 over an 8x-downsampling VAE, so any override has to be
    a multiple of 16 or the reshape in pooling stops being exact."""
    from phaselock.backends import get_spec

    spec = get_spec("wan21_t2v_1_3b")
    multiple = spec.spatial_ratio * spec.patch_size[1]
    assert multiple == 16
    for size in (512, 480, 832):
        assert size % multiple == 0, f"{size} is not a legal frame size"


def test_default_run_id_is_sortable_and_self_describing():
    """Date first so a listing is chronological, then the two settings that most often
    distinguish two runs of the same code."""
    import datetime

    from phaselock.config import default_run_id

    when = datetime.datetime(2026, 8, 10, 10, 30)
    assert default_run_id(100, "native", now=when) == "20260810_1030_n100_native"
    assert default_run_id(5, "letterbox", label="smoke", now=when) == \
        "20260810_1030_n5_letterbox_smoke"
    assert default_run_id(now=when) == "20260810_1030"


def test_run_ids_sort_chronologically():
    import datetime

    from phaselock.config import default_run_id

    early = default_run_id(100, "native", now=datetime.datetime(2026, 8, 9, 23, 59))
    late = default_run_id(5, "native", now=datetime.datetime(2026, 8, 10, 0, 1))
    assert early < late, "a listing must read in run order regardless of n"


def test_run_id_becomes_the_leading_path_segment(tmp_path):
    from phaselock.config import load

    config = load(None, backend__name="wan21_t2v_1_3b", data__name="likephys",
                  output__run_id="20260810_1030_n100", output__name="inversion",
                  output__root=str(tmp_path))
    assert config.output.base() == (
        tmp_path / "20260810_1030_n100" / "wan21_t2v_1_3b" / "likephys" / "inversion"
    )


def test_running_momentum_blend_config_loads_and_default_stays_latent():
    assert load().phaselock.running_momentum_source == "latent"
    path = Path(__file__).parent.parent / "configs/experiments/physics_iq_running_momentum.yaml"
    assert load(path).phaselock.running_momentum_source == "blend"
    assert load(**parse_overrides(["phaselock__running_momentum_source=blend"])).phaselock.running_momentum_source == "blend"
