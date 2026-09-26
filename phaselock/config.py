"""YAML configuration into frozen dataclasses.

Deliberately small: no Hydra, no interpolation, no plugin system. What it does provide is
the one thing that matters when a run takes ten hours -- **unknown keys raise**. A typo in
a config that silently falls back to a default produces a plausible-looking result under
settings nobody chose, and there is no way to tell after the fact.

Command-line overrides use ``section__key=value``::

    python scripts/run_inversion.py --config configs/experiments/inversion_likephys_cog_t2v.yaml \\
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

    height: Optional[int] = None
    width: Optional[int] = None
    """Override the backend's native frame size. ``None`` uses the spec.

    Worth setting for a square source. LikePhys is 512x512 and Wan's native 480x832
    letterboxes it to 42% black padding, which costs real compute: measured on one clip,
    480x832 took 248 s against 147 s at 512x512, a 40% saving, because the padding is a
    third of the token grid. Both are legal -- the only constraint is a multiple of
    ``spatial_ratio * patch_size``, 16 for both backends.

    Not free of consequences: 512x512 is off Wan's trained aspect ratio and inverted to
    41.3 dB against a 52.8 dB VAE ceiling, where letterboxed 480x832 reached 42.6 dB
    against 49.7 dB. Better absolute number, worse gap. Sweep it rather than assume."""


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
class DiagnosticConfig:
    """Opt-in running-momentum denoising diagnostics."""

    enabled: bool = False
    record_steps: Optional[list[int]] = None
    save_raw_tensors: bool = False
    fps: int = 8
    preview_height: int = 180
    output_subdir: str = "diagnostics"

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.preview_height <= 0:
            raise ValueError("diagnostics.fps and diagnostics.preview_height must be positive")
        if self.record_steps is not None and any(step < 0 for step in self.record_steps):
            raise ValueError("diagnostics.record_steps must contain non-negative steps")


@dataclass(frozen=True)
class ExplorationConfig:
    """Opt-in post-momentum exploration; variance is a disagreement heuristic."""

    enabled: bool = False
    mask_mode: str = "motion_disagreement"
    noise_ratio: float = 0.01
    floor: float = 0.05
    seed: int = 0
    structured: bool = True

    def __post_init__(self) -> None:
        import math
        if self.mask_mode not in {"uniform", "motion", "disagreement", "motion_disagreement"}:
            raise ValueError("unknown exploration.mask_mode")
        if not math.isfinite(self.noise_ratio) or self.noise_ratio < 0:
            raise ValueError("exploration.noise_ratio must be finite and non-negative")
        if not 0 <= self.floor <= 1:
            raise ValueError("exploration.floor must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("exploration.seed must be non-negative")


@dataclass(frozen=True)
class RefinementConfig:
    """CogVideoX DDIM adaptation of same-timestep Predict-and-Perturb."""

    enabled: bool = False
    steps_per_timestep: int = 1
    confidence_gate: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps_per_timestep < 0 or self.seed < 0:
            raise ValueError("refinement steps and seed must be non-negative")


@dataclass(frozen=True)
class PhaseLockConfig:
    """Latent Delta Guidance settings.

    Its own section rather than more keys on ``generation``: these describe a *method*
    applied during sampling, not the sampling itself, and the signal the guidance is built
    from is the axis this study varies.

    Defaults are the paper's. ``guide_end=None`` resolves to half of
    ``generation.num_steps``, which is the paper's schedule stated relative to step count
    rather than pinned at 25.
    """

    few_steps: int = 2
    guidance_strength: float = 0.05
    guide_start: int = 0
    guide_end: Optional[int] = None
    few_step_prior_type: str = "motion"
    """Which quantity the few-step prior is built from, and the full pass is held to.

    ``motion`` is PhaseLock's own first-order latent delta, i.e. the published method.
    ``accel``, ``jerk`` and ``perr`` are the ablation: the identical mechanism holding a
    different quantity fixed. See :mod:`phaselock.operators`, which is also the list this
    is validated against."""

    few_step_prior_source: str = "latent"
    """Which tensor the prior is measured on.

    ``latent`` is the sampler state, and is what PhaseLock uses. ``x0_hat`` and
    ``velocity`` are model outputs -- the network's current belief about the clean video,
    and the flow field carrying the state there. See :data:`phaselock.guidance.SOURCES`."""

    prior_mode: str = "running_momentum"  # running_momentum or few_step (standard PhaaseLock) 
    beta1: float = 0.9
    beta2: float = 0.999
    velocity_decay: float = 0.01
    """Extra decay applied to the running first moment before its EMA update."""
    running_momentum_mode: str = "residual" # residual or snr
    """Which quantity the momentum guidance is built from."""
    running_momentum_source: str = "latent"
    """Frame differences of post-step latents, current-step x0_hat, or their blend."""

    def __post_init__(self) -> None:
        from .guidance import SOURCES
        from .operators import OPERATORS

        if self.few_step_prior_source not in SOURCES:
            raise ValueError(
                f"unknown phaselock.few_step_prior_source {self.few_step_prior_source!r}; "
                f"expected one of {sorted(SOURCES)}"
            )
        if self.running_momentum_source not in {"latent", "x0_hat", "blend"}:
            raise ValueError("phaselock.running_momentum_source must be 'latent', 'x0_hat', or 'blend'")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("phaselock.beta1 and beta2 must be in [0, 1)")
        if not 0 <= self.velocity_decay < 1:
            raise ValueError("phaselock.velocity_decay must be in [0, 1)")
        if self.few_step_prior_type not in OPERATORS:
            raise ValueError(
                f"unknown phaselock.few_step_prior_type {self.few_step_prior_type!r}; "
                f"expected one of {sorted(OPERATORS)}"
            )


@dataclass(frozen=True)
class OutputConfig:
    """Where artefacts land.

    Results are filed as ``root/<backend>/<dataset>/<name>/`` rather than in one flat
    directory. Every number in this project is only meaningful against the model and
    dataset that produced it -- an accuracy from Wan T2V says nothing about CogVideoX
    I2V -- and a flat layout of hand-named runs makes that provenance a naming
    convention, which decays. The path itself carries it here.

    ``name`` is the stage plus any variant: ``detection``, ``detection_pilot6``,
    ``generation``, ``step_sweep``.
    """

    root: str = "/data/experiments/phaselock_geophys"
    run_id: str = ""
    """A label for the whole invocation, e.g. ``20260810_1030_n100``.

    Inserted between ``root`` and the backend segment, so one experiment's five tracks sit
    together under one folder and a later run cannot overwrite or silently append to an
    earlier one. Empty keeps the old flat layout.

    **Set it once per chain, not per stage.** Every stage of a run must share the same
    value or the gate lands somewhere the report cannot find it; the runner script
    computes it once and passes it to each command. That is also why it does not default
    to a timestamp: a default evaluated per process would give each stage its own."""

    name: str = "run"
    backend: str = ""
    dataset: str = ""
    """Filled in from the rest of the config by :meth:`Config.resolved`; not set by hand."""

    def dir(self, *parts: str) -> Path:
        path = self.base()
        for part in parts:
            path = path / part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def base(self) -> Path:
        """The run directory: ``root / run_id / backend / dataset / name``."""
        path = Path(self.root)
        for segment in (self.run_id, self.backend, self.dataset):
            if segment:
                path = path / segment
        return path / self.name


@dataclass(frozen=True)
class Config:
    """A complete experiment configuration."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    inversion: InversionConfig = field(default_factory=InversionConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    phaselock: PhaseLockConfig = field(default_factory=PhaseLockConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.exploration.enabled and self.refinement.enabled:
            raise ValueError("exploration and refinement must be evaluated separately")
        if (self.exploration.enabled or self.refinement.enabled) and self.phaselock.prior_mode != "running_momentum":
            raise ValueError("momentum extensions require prior_mode=running_momentum")

    def resolved(self) -> "Config":
        """Fill the output path's backend and dataset segments from this config.

        Done here rather than in each YAML file so the two can never disagree: a run's
        directory is derived from the backend and dataset it actually used, not from what
        someone typed alongside them.
        """
        if self.output.backend and self.output.dataset:
            return self
        return dataclasses.replace(
            self,
            output=dataclasses.replace(
                self.output,
                backend=self.output.backend or self.backend.name,
                dataset=self.output.dataset or self.data.name,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the resolved config next to the results it produced."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def default_run_id(
    limit: Optional[int] = None,
    resolution: str = "",
    label: str = "",
    now: Optional[Any] = None,
) -> str:
    """A sortable, self-describing folder name for one invocation.

    ``20260810_1030_n100_native``. Date first so a directory listing is chronological, then
    the two settings that most often distinguish two runs of the same code -- how many
    pairs, and at what geometry -- because those are exactly what you need to know when
    looking at two result folders side by side.

    Built here rather than in the shell so the format has one definition and a test.
    """
    import datetime

    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M")
    parts = [stamp]
    if limit is not None:
        parts.append(f"n{limit}")
    if resolution:
        parts.append(resolution)
    if label:
        parts.append(label)
    return "_".join(parts)


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

    return _build(Config, raw).resolved()


def parse_overrides(items: Sequence[str]) -> dict[str, str]:
    """Turn ``["data__limit=24", "probe__pooling=flatten"]`` into a dict."""
    parsed: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"override {item!r} must be of the form section__key=value")
        parsed[key.strip()] = value.strip()
    return parsed
