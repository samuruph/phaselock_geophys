"""Storage for pooled internal trajectories.

One :class:`ProbeRecord` holds every recorded trajectory for one video, keyed by
``(source, block, step)``, alongside enough provenance to tell two runs apart. Written as
a compressed ``.npz`` with a JSON sidecar, so the arrays stay loadable without this
package and the provenance stays readable without loading the arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

# Sources that are not per-block carry this in place of a block index.
NO_BLOCK = -1


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    """Identifies one pooled trajectory within a record."""

    source: str
    block: int
    step: int

    def to_string(self) -> str:
        return f"{self.source}|b{self.block}|s{self.step}"

    @classmethod
    def from_string(cls, text: str) -> "TrajectoryKey":
        source, block, step = text.split("|")
        return cls(source=source, block=int(block[1:]), step=int(step[1:]))


@dataclass
class ProbeRecord:
    """Every pooled trajectory captured for one video, plus provenance."""

    trajectories: dict[TrajectoryKey, torch.Tensor] = field(default_factory=dict)
    taus: dict[int, float] = field(default_factory=dict)
    """Normalised diffusion time per recorded step index. 0 is noise, 1 is clean data."""

    provenance: dict[str, Any] = field(default_factory=dict)

    def add(self, key: TrajectoryKey, values: torch.Tensor) -> None:
        if key in self.trajectories:
            raise KeyError(f"trajectory {key.to_string()} recorded twice")
        if values.ndim != 2:
            raise ValueError(f"expected a (T, D) trajectory, got {tuple(values.shape)}")
        # fp16 on disk: these are pooled averages of already-noisy activations, and the
        # statistics widen back to float32 before any differencing.
        self.trajectories[key] = values.detach().to(torch.float16).cpu()

    @property
    def sources(self) -> list[str]:
        return sorted({key.source for key in self.trajectories})

    @property
    def steps(self) -> list[int]:
        return sorted({key.step for key in self.trajectories})

    def blocks(self, source: str) -> list[int]:
        return sorted({key.block for key in self.trajectories if key.source == source})

    def get(self, source: str, step: int, block: int = NO_BLOCK) -> torch.Tensor:
        """One trajectory as float32, ready for the metrics."""
        key = TrajectoryKey(source=source, block=block, step=step)
        try:
            return self.trajectories[key].to(torch.float32)
        except KeyError:
            raise KeyError(
                f"no trajectory {key.to_string()}; recorded sources are {self.sources}"
            ) from None

    def iter_source(self, source: str) -> Iterator[tuple[TrajectoryKey, torch.Tensor]]:
        for key in sorted(k for k in self.trajectories if k.source == source):
            yield key, self.trajectories[key].to(torch.float32)

    def nbytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.trajectories.values())

    def save(self, path: str | Path) -> None:
        """Write ``<path>.npz`` and ``<path>.json``."""
        path = Path(path).with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            **{key.to_string(): value.numpy() for key, value in self.trajectories.items()},
        )
        sidecar = dict(self.provenance)
        sidecar["taus"] = {str(step): tau for step, tau in self.taus.items()}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "ProbeRecord":
        path = Path(path).with_suffix("")
        with np.load(path.with_suffix(".npz")) as archive:
            trajectories = {
                TrajectoryKey.from_string(name): torch.from_numpy(archive[name])
                for name in archive.files
            }
        provenance: dict[str, Any] = {}
        taus: dict[int, float] = {}
        sidecar_path = path.with_suffix(".json")
        if sidecar_path.is_file():
            provenance = json.loads(sidecar_path.read_text())
            taus = {int(k): float(v) for k, v in provenance.pop("taus", {}).items()}
        return cls(trajectories=trajectories, taus=taus, provenance=provenance)


@dataclass(frozen=True)
class StatisticRow:
    """One row of the per-video statistics table.

    The tables are what scoring consumes; full trajectories are only reloaded for
    figures and for the linear-probe comparison. A record's statistics are roughly
    40 KB against tens of megabytes of trajectories.
    """

    sample_id: str
    label: int
    group: str
    scenario: str
    violation: str
    source: str
    block: int
    step: int
    tau: float
    statistics: dict[str, float]
    drift: Optional[dict[str, float]] = None
    drift_estimator: Optional[str] = None
    alignment: Optional[float] = None
    erosion: Optional[float] = None

    def flatten(self) -> dict[str, Any]:
        """Flat dict suitable for a CSV row."""
        row: dict[str, Any] = {
            "sample_id": self.sample_id,
            "label": self.label,
            "group": self.group,
            "scenario": self.scenario,
            "violation": self.violation,
            "source": self.source,
            "block": self.block,
            "step": self.step,
            "tau": self.tau,
        }
        row.update({f"phi_{name}": value for name, value in self.statistics.items()})
        if self.drift is not None:
            row.update({f"drift_{name}": value for name, value in self.drift.items()})
            row["drift_estimator"] = self.drift_estimator
        if self.alignment is not None:
            row["alignment"] = self.alignment
        if self.erosion is not None:
            row["erosion"] = self.erosion
        return row
