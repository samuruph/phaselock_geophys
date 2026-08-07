"""LikePhys: matched valid/violated renders across 12 dynamic scenarios.

Layout on disk, which is the only metadata there is -- the release ships no manifest::

    data/<scenario>_videos/subgroup_NNN/valid_00.mp4
    data/<scenario>_videos/subgroup_NNN/<violation>_00.mp4

Each subgroup holds exactly one valid clip and 5-8 violations of it, and every subgroup
within a scenario carries the same violation set. Pairing each violation against its own
subgroup's valid clip gives **800 pairs** over 12 scenarios x 10 subgroups.

The paper cites 650 pairs, so this local copy is a different release. Absolute numbers
will not match published ones even with a correct implementation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .base import PairedVideoDataset, VideoPair, VideoSample

DEFAULT_ROOT = Path("/data/datasets/LikePhys-Benchmark/data")

_SUBGROUP = re.compile(r"^subgroup_(\d+)$")
_VIDEO = re.compile(r"^(?P<kind>.+?)_(?P<take>\d+)\.mp4$")

VALID_KIND = "valid"

# Prompts for the I2V continuation experiment. LikePhys ships no captions, and
# CogVideoX-5B-I2V degrades badly on an empty prompt, so one short description per
# scenario is written here. They are deliberately generic: naming the outcome would
# leak the physics the model is being tested on.
SCENARIO_PROMPTS: dict[str, str] = {
    "ball_collision": "Two balls roll across a flat surface and collide.",
    "ball_drop": "A ball falls onto a flat surface below it.",
    "block_slide": "A block slides along a flat surface.",
    "cloth_drape": "A piece of cloth falls and settles over an object.",
    "faucet": "Water pours from a faucet into the container below.",
    "flag": "A flag hangs from a pole and moves in the wind.",
    "fluid": "Liquid moves inside a transparent container.",
    "pendulum": "A pendulum bob swings from a fixed pivot.",
    "pyramid": "A stack of spheres arranged in a pyramid rests on a surface.",
    "river": "Water flows along a channel past obstacles.",
    "shadow": "An object rests on a surface under a light, casting a shadow.",
    "shadow_camera": "A camera moves around an object casting a shadow on a surface.",
}


class LikePhys(PairedVideoDataset):
    """The LikePhys benchmark, paired from its directory layout."""

    name = "likephys"

    def __init__(self, root: os.PathLike | str = DEFAULT_ROOT):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"LikePhys root not found: {self.root}")
        self._pairs: Optional[list[VideoPair]] = None

    def _scenario_dirs(self) -> list[tuple[str, Path]]:
        found = [
            (path.name[: -len("_videos")], path)
            for path in sorted(self.root.iterdir())
            if path.is_dir() and path.name.endswith("_videos")
        ]
        if not found:
            raise FileNotFoundError(f"no '*_videos' scenario directories under {self.root}")
        return found

    def pairs(self) -> list[VideoPair]:
        if self._pairs is not None:
            return self._pairs

        pairs: list[VideoPair] = []
        for scenario, scenario_dir in self._scenario_dirs():
            for subgroup_dir in sorted(scenario_dir.iterdir()):
                match = _SUBGROUP.match(subgroup_dir.name)
                if not match or not subgroup_dir.is_dir():
                    continue
                subgroup = match.group(1)
                group = f"{scenario}/{subgroup}"

                clips: dict[str, Path] = {}
                for video in sorted(subgroup_dir.iterdir()):
                    video_match = _VIDEO.match(video.name)
                    if video_match:
                        clips[video_match.group("kind")] = video

                if VALID_KIND not in clips:
                    raise FileNotFoundError(f"{subgroup_dir} has no {VALID_KIND}_*.mp4")

                plausible = VideoSample(
                    sample_id=f"{group}/{VALID_KIND}",
                    path=str(clips[VALID_KIND]),
                    label=0,
                    group=group,
                    dataset=self.name,
                    meta={"scenario": scenario, "subgroup": subgroup, "kind": VALID_KIND},
                )
                for kind, path in clips.items():
                    if kind == VALID_KIND:
                        continue
                    pairs.append(
                        VideoPair(
                            plausible=plausible,
                            violated=VideoSample(
                                sample_id=f"{group}/{kind}",
                                path=str(path),
                                label=1,
                                group=group,
                                dataset=self.name,
                                meta={"scenario": scenario, "subgroup": subgroup, "kind": kind},
                            ),
                            scenario=scenario,
                            violation=kind,
                        )
                    )

        self._pairs = pairs
        return pairs

    def valid_clips(self) -> list[VideoSample]:
        """The one plausible clip per subgroup, deduplicated.

        These are the conditioning sources for the generation experiment: their first
        frame seeds the model and their continuation is the physically plausible target.
        """
        seen: dict[str, VideoSample] = {}
        for pair in self.pairs():
            seen.setdefault(pair.plausible.sample_id, pair.plausible)
        return sorted(seen.values(), key=lambda sample: sample.sample_id)

    def prompt_for(self, scenario: str) -> str:
        try:
            return SCENARIO_PROMPTS[scenario]
        except KeyError:
            raise KeyError(
                f"no prompt registered for LikePhys scenario {scenario!r}; "
                f"add one to SCENARIO_PROMPTS"
            ) from None
