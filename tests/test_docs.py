"""Guards against documentation drifting from the code.

Docs go stale silently, and a run guide that names a flag which no longer exists costs
more time than no guide at all.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from pathlib import Path

import pytest

from phaselock.config import Config

ROOT = Path(__file__).parent.parent
DOCS = ["README.md", "docs/RUNNING.md", "docs/METHOD.md", "docs/VALIDATION.md"]


@pytest.mark.parametrize("doc", DOCS)
def test_every_referenced_path_exists(doc):
    """A doc that points at a moved or deleted file sends the reader in circles."""
    text = (ROOT / doc).read_text()
    referenced = set(re.findall(r"(?:configs|scripts|phaselock|docs|tests)/[\w/.-]+\.(?:yaml|py|md)", text))
    missing = sorted(ref for ref in referenced if not (ROOT / ref).exists())
    assert not missing, f"{doc} references missing paths: {missing}"


def test_every_config_key_is_documented():
    """A key nobody documents is a key nobody uses correctly."""
    text = (ROOT / "docs/RUNNING.md").read_text()
    hints = typing.get_type_hints(Config)
    undocumented = [
        f"{section.name}.{field.name}"
        for section in dataclasses.fields(Config)
        for field in dataclasses.fields(hints[section.name])
        if f"`{field.name}`" not in text
    ]
    assert not undocumented, f"undocumented config keys: {undocumented}"


def test_no_config_key_is_dead():
    """A key declared but never read would silently do nothing when set."""
    sources = "\n".join(
        path.read_text()
        for path in list((ROOT / "phaselock").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
        if path.name != "config.py"
    )
    hints = typing.get_type_hints(Config)
    dead = [
        f"{section.name}.{field.name}"
        for section in dataclasses.fields(Config)
        for field in dataclasses.fields(hints[section.name])
        if not re.search(rf"\b{field.name}\b", sources)
    ]
    assert not dead, f"config keys declared but never read: {dead}"


@pytest.mark.parametrize("doc", DOCS)
def test_documented_test_count_is_current(doc):
    """Stale counts are a cheap signal that a doc was not revisited."""
    import subprocess

    text = (ROOT / doc).read_text()
    claimed = set(re.findall(r"(\d{3})(?= (?:CPU )?tests)", text))
    if not claimed:
        pytest.skip(f"{doc} does not claim a test count")

    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if match is None:
        pytest.skip("could not determine the collected test count")

    actual = int(match.group(1))
    for value in claimed:
        assert abs(int(value) - actual) <= 5, (
            f"{doc} claims {value} tests but {actual} are collected"
        )


def test_signal_inventory_matches_what_the_code_emits():
    """METHOD.md's signal count table has to stay true.

    It is the reference for what every number in a run means, so a source or statistic
    added without updating it would leave the doc quietly wrong.
    """
    from phaselock.metrics.geophys import STATISTICS
    from phaselock.probes import SOURCES

    text = (ROOT / "docs/METHOD.md").read_text()
    inventory = text[text.index("## 4b. The full signal inventory") : text.index("## 5. Scoring")]

    for name in STATISTICS:
        assert f"`{name}`" in inventory, f"statistic {name} missing from the inventory"
    for source in SOURCES:
        if source == "attention":  # registered but not recorded by any experiment
            continue
        assert f"`{source}`" in inventory, f"source {source} missing from the inventory"
    for extra in ("or", "majority", "alignment", "erosion"):
        assert f"`{extra}`" in inventory, f"{extra} missing from the inventory"


def test_signal_inventory_arithmetic_is_right():
    """The worked example must actually add up, or it teaches the wrong thing."""
    blocks, steps, phi_quantities, drift_quantities = 30, 10, 7, 5
    total = (
        blocks * steps * phi_quantities              # hidden_states phi
        + blocks * (steps - 1) * drift_quantities    # hidden_states drift, empirical
        + steps * phi_quantities                     # latent phi
        + steps * drift_quantities                   # latent drift, exact -> no lost step
        + steps * 2                                  # latent coupling
        + 2 * (steps * phi_quantities                # x0_hat, velocity phi
               + (steps - 1) * drift_quantities)     # ... and their empirical drift
    )
    assert total == 3820
    assert "**3820**" in (ROOT / "docs/METHOD.md").read_text()
