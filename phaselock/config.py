"""YAML configuration into frozen dataclasses.

Deliberately small: no Hydra, no interpolation, no plugin system. What it does provide is
the one thing that matters when a run takes ten hours -- **unknown keys raise**. A typo in
a config that silently falls back to a default produces a plausible-looking result under
settings nobody chose, and there is no way to tell after the fact.

Command-line overrides use ``section__key=value``::

    python scripts/run_detection.py --config configs/experiments/detection_likephys.yaml \\
        inversion__num_steps=100 data__limit=24
"""

from __future__ import annotations

import dataclasses
import functools
import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, get_args, get_origin

import yaml


@functools.lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    """Resolved type hints for a dataclass.

    ``from __future__ import annotations`` turns every annotation into a string, so
    ``dataclasses.fields(cls)[i].type`` is ``"BackendConfig"`` rather than the class.
    Resolving them is what lets nested sections be built and string overrides be coerced.
    """
    return typing.get_type_hints(cls)


@dataclass(frozen=True)
class BackendConfig:
    """Which model to load."""

    name: str = "cogvideox_5b_t2v"
    model_id: Optional[str] = None
    """Overrides the registry default, for a local checkout or a pinned revision."""

    dtype: str = "bfloat16"
    offload: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Which clips to run, and how to prepare them."""

    name: str = "likephys"
    root: Optional[str] = None
    split: Optional[str] = None
    scenarios: Optional[list[str]] = None
    violations: Optional[list[str]] = None
    categories: Optional[list[str]] = None
    limit: Optional[int] = None
    seed: int = 0
    window: Optional[float] = None
    """Centred fraction of each clip to keep. Mainly for IntPhys2's 10.6 s clips."""

    blur_sigma: float = 0.0
    """Gaussian blur applied to every arm, including the reference. PhaseLock's control."""


@dataclass(frozen=True)
class ProbeConfig:
    """What to record from inside the model."""

    sources: list[str] = field(
        default_factory=lambda: ["hidden_states", "latent", "x0_hat", "velocity"]
    )
    blocks: Optional[list[int]] = None
    block_stride: int = 1
    pooling: str = "mean"
    record_steps: int = 10
    save_trajectories: bool = False
    """Full trajectories are tens of MB per video; the statistics table is ~40 KB."""


@dataclass(frozen=True)
class InversionConfig:
    """How to recover a real video's latent trajectory."""

    num_steps: int = 50
    prompt: str = ""
    """Empty by default so no text confound enters a plausible-vs-violated comparison."""

    reconstruction_check: bool = False


@dataclass(frozen=True)
class MetricConfig:
    """Options for the geometric statistics."""

    ar_order: int = 3
    residual_fit: str = "span"
    ridge_lambda: float = 0.1
    bootstrap_resamples: int = 1000


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling settings for the generation experiments."""

    num_steps: int = 50
    step_sweep: Optional[list[int]] = None
    blur_sweep: Optional[list[float]] = None
    guidance_scale: float = 6.0
    num_candidates: int = 1
    seed: int = 42
    negative_prompt: Optional[str] = None


@dataclass(frozen=True)
class OutputConfig:
    """Where artefacts land."""

    root: str = "/data/experiments/phaselock_geophys"
    name: str = "run"

    def dir(self, *parts: str) -> Path:
        path = Path(self.root) / self.name
        for part in parts:
            path = path / part
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class Config:
    """A complete experiment configuration."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    inversion: InversionConfig = field(default_factory=InversionConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the resolved config next to the results it produced."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def _coerce(value: Any, annotation: Any) -> Any:
    """Convert a string override to the field's declared type."""
    if not isinstance(value, str):
        return value

    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value.lower() in ("none", "null"):
            return None
        if origin in (list, Sequence):
            inner = args[0] if args else str
            return [_coerce(item.strip(), inner) for item in value.split(",") if item.strip()]
        return _coerce(value, args[0]) if args else value

    if annotation is bool:
        lowered = value.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot read {value!r} as a boolean")
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    return value


def _as_list(value: Any) -> Any:
    """Let a single scalar stand in for a one-element list in YAML."""
    return value if value is None or isinstance(value, list) else [value]


def _build(cls: type, values: dict[str, Any], path: str = "") -> Any:
    """Instantiate a dataclass from a mapping, rejecting keys it does not declare."""
    hints = _hints(cls)
    declared = {f.name for f in fields(cls)}
    unknown = set(values) - declared
    if unknown:
        where = f" in section {path!r}" if path else ""
        raise ValueError(
            f"unknown configuration key(s) {sorted(unknown)}{where}. "
            f"Valid keys: {sorted(declared)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        annotation = hints[name]
        if is_dataclass(annotation) and isinstance(value, dict):
            kwargs[name] = _build(annotation, value, path=name)
            continue
        if _is_list_type(annotation):
            value = _as_list(value)
        kwargs[name] = value
    return cls(**kwargs)


def _is_list_type(annotation: Any) -> bool:
    """True for ``list[X]`` and ``Optional[list[X]]``."""
    if get_origin(annotation) in (list, Sequence):
        return True
    return any(get_origin(arg) in (list, Sequence) for arg in get_args(annotation))


def load(path: Optional[str | Path] = None, **overrides: Any) -> Config:
    """Load a config from YAML and apply ``section__key`` overrides.

    ``None`` overrides are ignored, so an argparse default of ``None`` can be forwarded
    unconditionally without clobbering the file's value.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text())
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}: expected a mapping at the top level")
            raw = loaded

    sections = _hints(Config)
    for key, value in overrides.items():
        if value is None:
            continue
        section, separator, leaf = key.partition("__")
        if not separator:
            raise ValueError(
                f"override {key!r} must be of the form section__key, e.g. data__limit=24"
            )
        if section not in sections:
            raise ValueError(f"unknown config section {section!r}; valid: {sorted(sections)}")

        declared = _hints(sections[section])
        if leaf not in declared:
            raise ValueError(
                f"unknown key {leaf!r} in section {section!r}; valid: {sorted(declared)}"
            )
        raw.setdefault(section, {})[leaf] = _coerce(value, declared[leaf])

    return _build(Config, raw)


def parse_overrides(items: Sequence[str]) -> dict[str, str]:
    """Turn ``["data__limit=24", "probe__pooling=flatten"]`` into a dict."""
    parsed: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"override {item!r} must be of the form section__key=value")
        parsed[key.strip()] = value.strip()
    return parsed
