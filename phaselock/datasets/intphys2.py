"""IntPhys2: photorealistic possible/impossible pairs over four core-knowledge properties.

Driven by ``<split>/metadata.csv``, whose columns are::

    SceneIndex, name, file_name, game_name, condition, env, type, occluder, Difficulty, Camera

``type`` is ``<run>_<Possible|Impossible>``. A scene is filmed twice, so pairing on
``(SceneIndex, run)`` gives **506 pairs** from 253 scenes on the Main split, matching the
count the GeoPhys paper reports.

Clips run 636 frames at 60 fps -- about 10.6 s. Resampling that to a 49-frame model input
is a 13x decimation, which can step straight over a brief violation; ``video_io`` exposes
a ``window`` option to trade coverage for temporal resolution.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .base import PairedVideoDataset, VideoPair, VideoSample

DEFAULT_ROOT = Path("/data/datasets/IntPhys2")
SPLITS = ("Main", "Debug", "HeldOut")

POSSIBLE = "Possible"
IMPOSSIBLE = "Impossible"


class IntPhys2(PairedVideoDataset):
    """The IntPhys2 benchmark, paired from its metadata CSV.

    Args:
        root: Dataset root containing the split directories.
        split: One of ``Main``, ``Debug``, ``HeldOut``. ``HeldOut`` ships videos without
            labels, so it cannot be paired.
    """

    name = "intphys2"

    def __init__(self, root: os.PathLike | str = DEFAULT_ROOT, split: str = "Main"):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.metadata_path = self.split_dir / "metadata.csv"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"IntPhys2 metadata not found: {self.metadata_path}. "
                f"The {split!r} split may not ship labels."
            )
        self._pairs: Optional[list[VideoPair]] = None

    def _rows(self) -> list[dict[str, str]]:
        with open(self.metadata_path, newline="") as handle:
            return list(csv.DictReader(handle))

    def pairs(self) -> list[VideoPair]:
        if self._pairs is not None:
            return self._pairs

        grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in self._rows():
            run, _, outcome = row["type"].partition("_")
            if outcome not in (POSSIBLE, IMPOSSIBLE):
                raise ValueError(f"unexpected type {row['type']!r} in {self.metadata_path}")
            grouped[(row["SceneIndex"], run)][outcome] = row

        pairs: list[VideoPair] = []
        for (scene_index, run), outcomes in sorted(grouped.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
            if set(outcomes) != {POSSIBLE, IMPOSSIBLE}:
                raise ValueError(
                    f"scene {scene_index} run {run} is not a complete pair: {sorted(outcomes)}"
                )
            group = f"{scene_index}/{run}"
            samples = {
                outcome: VideoSample(
                    sample_id=row["name"],
                    path=str(self.split_dir / row["file_name"]),
                    label=int(outcome == IMPOSSIBLE),
                    group=group,
                    dataset=self.name,
                    meta={
                        "scene_index": scene_index,
                        "run": run,
                        "condition": row["condition"],
                        "game_name": row["game_name"],
                        "env": row["env"],
                        "occluder": row["occluder"],
                        "difficulty": row["Difficulty"],
                        "camera": row["Camera"],
                    },
                )
                for outcome, row in outcomes.items()
            }
            pairs.append(
                VideoPair(
                    plausible=samples[POSSIBLE],
                    violated=samples[IMPOSSIBLE],
                    # The core-knowledge property is the natural scenario axis: it is
                    # what the benchmark reports per-column, and it groups scenes that
                    # test the same thing.
                    scenario=outcomes[POSSIBLE]["condition"],
                    violation=outcomes[POSSIBLE]["condition"],
                )
            )

        self._pairs = pairs
        return pairs

    def select_by(
        self,
        difficulty: Optional[str] = None,
        camera: Optional[str] = None,
        **kwargs,
    ) -> list[VideoPair]:
        """:meth:`select` plus the IntPhys2-specific difficulty and camera splits.

        The paper breaks results down along both, and moving-camera clips are the harder
        half, so keeping them separable matters.
        """
        chosen = self.select(**kwargs)
        if difficulty is not None:
            available = {pair.plausible.meta["difficulty"] for pair in self.pairs()}
            if difficulty not in available:
                raise ValueError(f"unknown difficulty {difficulty!r}; available: {sorted(available)}")
            chosen = [p for p in chosen if p.plausible.meta["difficulty"] == difficulty]
        if camera is not None:
            available = {pair.plausible.meta["camera"] for pair in self.pairs()}
            if camera not in available:
                raise ValueError(f"unknown camera {camera!r}; available: {sorted(available)}")
            chosen = [p for p in chosen if p.plausible.meta["camera"] == camera]
        return chosen
