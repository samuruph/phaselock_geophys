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
    assert (tmp_path / "run1" / "trajectories").is_dir()


def test_config_serialises_for_the_run_record(tmp_path):
    config = load(CONFIGS[0])
    path = tmp_path / "config.json"
    config.save(path)
    assert '"residual_fit"' in path.read_text()


def test_parse_overrides_requires_an_equals_sign():
    with pytest.raises(ValueError, match="section__key=value"):
        parse_overrides(["data__limit"])
