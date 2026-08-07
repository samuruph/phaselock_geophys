"""Physics-IQ: real-world I2V continuation benchmark.

Driven by ``descriptions/best_practice/descriptions_base.csv``, whose columns are
``scenario, description, category, generated_video_name``. Scenario names look like::

    0001_perspective-left_take-1_trimmed-ball-and-block-fall.mp4

Each scene was shot twice under identical conditions. **Take 1 is the benchmark** (198
scenarios, indices 0001-0198); take 2 is the real-vs-real noise floor and carries its own
index range (0199-0396), so a take-2 path cannot be derived by string-substituting the
take field -- it has to be looked up by ``(perspective, scene)``.

Asset filenames are not the scenario name. Each asset type carries its own infix::

    switch-frames/  0001_switch-frames_anyFPS_perspective-left_trimmed-<scene>.jpg
    split-videos/testing/30FPS/
                    0001_testing-videos_30FPS_perspective-left_take-1_trimmed-<scene>.mp4
    video-masks/real/8FPS/
                    0001_video-masks_8FPS_perspective-left_take-1_trimmed-<scene>.mp4

Note that switch frames use ``anyFPS`` and omit the take entirely.

This replaces the loader in the old ``scripts/test_physics_iq.py``, which read the
parallel ``.txt`` file and zipped it positionally against a sorted directory listing.
That happened to line up, but it carried no category and no take number, so it generated
all 396 rows including the take-2 duplicates, and it named outputs after the switch-frame
file rather than the ``generated_video_name`` the official evaluator expects.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Optional

from .base import GenerationDataset, GenerationSample

DEFAULT_ROOT = Path("/data/datasets/physics-IQ-benchmark-verified")

BENCHMARK_TAKE = 1
PAIR_TAKE = 2
AVAILABLE_FPS = (8, 16, 24, 30)
DEFAULT_MASK_FPS = 8

# The scene component keeps its "trimmed-" prefix: it is part of the on-disk name.
_SCENARIO = re.compile(
    r"^(?P<index>\d{4})_perspective-(?P<perspective>[a-z]+)_take-(?P<take>\d+)_(?P<scene>.+)\.mp4$"
)


class PhysicsIQ(GenerationDataset):
    """The Physics-IQ benchmark's 198 take-1 evaluation scenarios."""

    name = "physics_iq"

    def __init__(
        self,
        root: os.PathLike | str = DEFAULT_ROOT,
        testing_fps: int = 30,
        mask_fps: int = DEFAULT_MASK_FPS,
    ):
        for label, value in (("testing_fps", testing_fps), ("mask_fps", mask_fps)):
            if value not in AVAILABLE_FPS:
                raise ValueError(f"{label} must be one of {AVAILABLE_FPS}, got {value}")
        self.root = Path(root)
        self.testing_fps = testing_fps
        self.mask_fps = mask_fps
        if not self.descriptions_path.is_file():
            raise FileNotFoundError(f"Physics-IQ descriptions not found: {self.descriptions_path}")
        self._samples: Optional[list[GenerationSample]] = None

    @property
    def descriptions_path(self) -> Path:
        return self.root / "descriptions" / "best_practice" / "descriptions_base.csv"

    @staticmethod
    def _asset_name(
        index: str,
        asset: str,
        fps: str,
        perspective: str,
        scene: str,
        take: Optional[int],
        extension: str = ".mp4",
    ) -> str:
        """Build an on-disk asset filename from parsed scenario components.

        ``scene`` arrives without its extension because the scenario regex consumes it.
        ``take=None`` omits the take field, which is how switch frames are named.
        """
        take_field = f"take-{take}_" if take is not None else ""
        return f"{index}_{asset}_{fps}_perspective-{perspective}_{take_field}{scene}{extension}"

    def _optional(self, path: Path) -> Optional[str]:
        return str(path) if path.exists() else None

    def generation_samples(self) -> list[GenerationSample]:
        if self._samples is not None:
            return self._samples

        with open(self.descriptions_path, newline="") as handle:
            rows = list(csv.DictReader(handle))

        parsed = []
        for row in rows:
            match = _SCENARIO.match(row["scenario"])
            if not match:
                raise ValueError(f"unparsable Physics-IQ scenario name: {row['scenario']!r}")
            parsed.append((match.groupdict(), row))

        # Take 2 shares (perspective, scene) with take 1 but has a different index.
        pair_index = {
            (fields["perspective"], fields["scene"]): fields["index"]
            for fields, _ in parsed
            if int(fields["take"]) == PAIR_TAKE
        }

        samples: list[GenerationSample] = []
        for fields, row in parsed:
            if int(fields["take"]) != BENCHMARK_TAKE:
                continue
            index, perspective, scene = fields["index"], fields["perspective"], fields["scene"]

            testing_dir = self.root / "split-videos" / "testing" / f"{self.testing_fps}FPS"
            mask_dir = self.root / "video-masks" / "real" / f"{self.mask_fps}FPS"

            pair_path = None
            if (perspective, scene) in pair_index:
                pair_path = self._optional(
                    testing_dir
                    / (
                        self._asset_name(
                            pair_index[(perspective, scene)],
                            "testing-videos",
                            f"{self.testing_fps}FPS",
                            perspective,
                            scene,
                            PAIR_TAKE,
                        )
                    )
                )

            samples.append(
                GenerationSample(
                    sample_id=index,
                    prompt=row["description"],
                    category=row["category"],
                    image_path=self._optional(
                        self.root
                        / "switch-frames"
                        / self._asset_name(
                            index, "switch-frames", "anyFPS", perspective, scene, None, ".jpg"
                        )
                    ),
                    reference_path=self._optional(
                        testing_dir
                        / self._asset_name(
                            index,
                            "testing-videos",
                            f"{self.testing_fps}FPS",
                            perspective,
                            scene,
                            BENCHMARK_TAKE,
                        )
                    ),
                    mask_path=self._optional(
                        mask_dir
                        / self._asset_name(
                            index, "video-masks", f"{self.mask_fps}FPS", perspective, scene, BENCHMARK_TAKE
                        )
                    ),
                    meta={
                        "scenario": row["scenario"],
                        # The filename the official Physics-IQ evaluator expects.
                        "output_name": row["generated_video_name"],
                        "perspective": perspective,
                        "scene": scene,
                        "conditioning_path": self._optional(
                            self.root
                            / "split-videos"
                            / "conditioning"
                            / f"{self.testing_fps}FPS"
                            / self._asset_name(
                                index,
                                "conditioning-videos",
                                f"{self.testing_fps}FPS",
                                perspective,
                                scene,
                                BENCHMARK_TAKE,
                            )
                        ),
                        # Take 2 of the same scene: the real-vs-real ceiling that
                        # normalises the benchmark score to 100%.
                        "pair_path": pair_path,
                    },
                )
            )

        self._samples = sorted(samples, key=lambda sample: sample.sample_id)
        return self._samples
