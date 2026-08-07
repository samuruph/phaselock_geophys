"""Dataset protocols.

Two shapes of dataset are needed and they are deliberately not merged:

* :class:`PairedVideoDataset` -- LikePhys and IntPhys2. Matched pairs sharing initial
  conditions where exactly one member violates physics. Used for detection.
* :class:`GenerationDataset` -- Physics-IQ, and LikePhys in its conditioning role. A
  prompt plus a conditioning image plus a ground-truth continuation. Used for generation.

Forcing both through one interface would mean every consumer branching on which it
really had.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class VideoSample:
    """One video with a known plausibility label."""

    sample_id: str
    path: str
    label: int
    """1 if the video violates physics, 0 if it is plausible."""

    group: str
    """Pairing key. Members of a pair share it."""

    dataset: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoPair:
    """A matched plausible/violated pair sharing initial conditions."""

    plausible: VideoSample
    violated: VideoSample
    scenario: str
    """Coarse grouping used as the bootstrap resampling unit."""

    violation: str
    """What was broken, e.g. "penetration" or "solidity"."""

    @property
    def group(self) -> str:
        return self.plausible.group


@dataclass(frozen=True)
class GenerationSample:
    """One conditioning setup plus its ground-truth continuation."""

    sample_id: str
    prompt: str
    category: str
    image_path: Optional[str] = None
    reference_path: Optional[str] = None
    """Real continuation, used as the fidelity target."""

    mask_path: Optional[str] = None
    """Motion mask, where the benchmark supplies one."""

    meta: Mapping[str, Any] = field(default_factory=dict)


def balanced_subset(
    pairs: Sequence[VideoPair], limit: Optional[int], key: str = "scenario", seed: int = 0
) -> list[VideoPair]:
    """Take ``limit`` pairs spread evenly across ``key`` rather than the first N.

    The first N pairs of any of these datasets are all one scenario, which would make a
    pilot run measure one kind of physics. Deterministic given ``seed``.
    """
    import random

    if limit is None or limit >= len(pairs):
        return list(pairs)
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    buckets: dict[Any, list[VideoPair]] = {}
    for pair in pairs:
        buckets.setdefault(getattr(pair, key), []).append(pair)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    chosen: list[VideoPair] = []
    order = sorted(buckets)
    while len(chosen) < limit:
        progressed = False
        for name in order:
            if buckets[name]:
                chosen.append(buckets[name].pop())
                progressed = True
                if len(chosen) == limit:
                    break
        if not progressed:  # pragma: no cover - guarded by the limit check above
            break
    return chosen


def _check_filter(name: str, requested: Optional[Iterable[str]], available: Iterable[str]) -> Optional[set[str]]:
    """Validate a filter against what the dataset actually contains.

    A typo in a scenario name should fail loudly, not silently select nothing and make a
    run look like it produced no signal.
    """
    if requested is None:
        return None
    requested = {requested} if isinstance(requested, str) else set(requested)
    available = set(available)
    unknown = requested - available
    if unknown:
        raise ValueError(
            f"unknown {name}: {sorted(unknown)}. Available: {sorted(available)}"
        )
    return requested


class PairedVideoDataset(ABC):
    """A benchmark of matched plausible/violated video pairs."""

    name: str

    @abstractmethod
    def pairs(self) -> list[VideoPair]:
        """Every matched pair, in a deterministic order."""

    def samples(self) -> list[VideoSample]:
        """Every distinct video. A plausible clip shared by several pairs appears once."""
        seen: dict[str, VideoSample] = {}
        for pair in self.pairs():
            for sample in (pair.plausible, pair.violated):
                seen.setdefault(sample.sample_id, sample)
        return list(seen.values())

    @property
    def scenarios(self) -> list[str]:
        return sorted({pair.scenario for pair in self.pairs()})

    @property
    def violations(self) -> list[str]:
        return sorted({pair.violation for pair in self.pairs()})

    def select(
        self,
        scenarios: Optional[Iterable[str]] = None,
        violations: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> list[VideoPair]:
        """Filter and optionally subsample, balanced across scenarios.

        ``None`` means "keep everything". Unknown filter values raise.
        """
        wanted_scenarios = _check_filter("scenarios", scenarios, self.scenarios)
        wanted_violations = _check_filter("violations", violations, self.violations)

        chosen = [
            pair
            for pair in self.pairs()
            if (wanted_scenarios is None or pair.scenario in wanted_scenarios)
            and (wanted_violations is None or pair.violation in wanted_violations)
        ]
        return balanced_subset(chosen, limit, seed=seed)


class GenerationDataset(ABC):
    """A benchmark of conditioning setups with ground-truth continuations."""

    name: str

    @abstractmethod
    def generation_samples(self) -> list[GenerationSample]:
        """Every conditioning setup, in a deterministic order."""

    @property
    def categories(self) -> list[str]:
        return sorted({sample.category for sample in self.generation_samples()})

    def select_generation(
        self,
        categories: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> list[GenerationSample]:
        """Filter by category and optionally subsample, balanced across categories."""
        wanted = _check_filter("categories", categories, self.categories)
        chosen = [
            sample
            for sample in self.generation_samples()
            if wanted is None or sample.category in wanted
        ]
        if limit is None or limit >= len(chosen):
            return chosen

        import random

        buckets: dict[str, list[GenerationSample]] = {}
        for sample in chosen:
            buckets.setdefault(sample.category, []).append(sample)
        rng = random.Random(seed)
        for bucket in buckets.values():
            rng.shuffle(bucket)

        picked: list[GenerationSample] = []
        order = sorted(buckets)
        while len(picked) < limit:
            for name in order:
                if buckets[name]:
                    picked.append(buckets[name].pop())
                    if len(picked) == limit:
                        break
        return picked
