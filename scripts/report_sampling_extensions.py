#!/usr/bin/env python
"""Compare extension runs on their common Physics-IQ samples and report diversity.

Usage: python scripts/report_sampling_extensions.py RUN_DIR [RUN_DIR ...] --out comparison.csv
The pooled Physics-IQ score is recomputed only on the shared sample intersection.
Saturation follows measure_oversaturation.py; diversity is pairwise mean absolute RGB
frame difference across auxiliary seeds of the same method and generation seed.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from measure_oversaturation import measure_video
from report_guidance import score_of, VARIANCE_FIELDS


def load_run(directory: Path):
    files = list(directory.glob("physics_iq_*.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one physics_iq_*.csv in {directory}; found {len(files)}")
    with files[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["setting"] for r in rows}) != 1:
        raise ValueError(f"expected one guidance setting in {files[0]}")
    parsed = {}
    for row in rows:
        for key in (*VARIANCE_FIELDS, *VARIANCE_FIELDS.values(), "motion", "raw_score"):
            row[key] = float(row[key]) if row.get(key) not in (None, "") else float("nan")
        parsed[row["sample_id"]] = row
    return parsed


def frames(path: Path, max_frames=49, longest_edge=256):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError(f"cannot open {path}")
    output = []
    try:
        for _ in range(max_frames):
            ok, frame = video.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = min(1.0, longest_edge / max(h, w))
            output.append(cv2.resize(frame, (round(w * scale), round(h * scale)),
                                     interpolation=cv2.INTER_AREA))
    finally:
        video.release()
    if not output:
        raise ValueError(f"no frames in {path}")
    return output


def pairwise_diversity(a: Path, b: Path) -> float:
    first, second = frames(a), frames(b)
    if len(first) != len(second) or first[0].shape != second[0].shape:
        raise ValueError(f"video dimensions differ: {a} and {b}")
    return float(np.mean([np.abs(x.astype(np.float32) - y.astype(np.float32)).mean() / 255
                          for x, y in zip(first, second)]))


def report(directories: list[Path]) -> list[dict]:
    runs = {directory: load_run(directory) for directory in directories}
    common = set.intersection(*(set(rows) for rows in runs.values()))
    if not common:
        raise ValueError("no common sample IDs across runs")
    summaries = []
    diversity_groups = defaultdict(list)
    for directory, rows in runs.items():
        selected = [rows[sample] for sample in sorted(common)]
        setting = selected[0]["setting"]
        metrics = []
        videos = {}
        for sample in sorted(common):
            row = rows[sample]
            video = directory / "videos" / setting / row["output_name"]
            if not video.exists():
                raise ValueError(f"missing evaluator video {video}")
            videos[sample] = video
            metrics.append(measure_video(video, 49, 256, .90, .80))
        metadata_by_sample = {}
        for sample in sorted(common):
            metadata_path = (directory / "sampling" / setting
                             / f"{Path(rows[sample]['output_name']).stem}.json")
            if metadata_path.exists():
                metadata_by_sample[sample] = json.loads(metadata_path.read_text())
        metadata = next(iter(metadata_by_sample.values()), {})
        extension = metadata.get("exploration", {})
        refinement = metadata.get("refinement", {})
        generation_seed = metadata.get("generation_seed")
        key = (
            json.dumps({k: v for k, v in extension.items() if k != "seed"}, sort_keys=True),
            json.dumps({k: v for k, v in refinement.items() if k != "seed"}, sort_keys=True),
            generation_seed, setting,
        )
        if extension.get("enabled"):
            diversity_groups[key].append((directory, extension.get("seed"), videos))
        summaries.append({
            "run": str(directory), "samples": len(common),
            "physics_iq": score_of(selected),
            "mean_motion": float(np.mean([row["motion"] for row in selected])),
            "mean_saturation": float(np.mean([m["mean_saturation"] for m in metrics])),
            "rgb_clipping_fraction": float(np.mean([m["rgb_clipping_fraction"] for m in metrics])),
            "high_saturation_fraction": float(np.mean([m["high_saturation_fraction"] for m in metrics])),
            "mean_value": float(np.mean([m["mean_value"] for m in metrics])),
            "mean_guided_predictions": (float(np.mean([
                m["guided_predictions"] for m in metadata_by_sample.values()
                if isinstance(m.get("guided_predictions"), (int, float))]))
                if any(isinstance(m.get("guided_predictions"), (int, float))
                       for m in metadata_by_sample.values()) else ""),
            "mean_transformer_calls": (float(np.mean([
                m["transformer_calls"] for m in metadata_by_sample.values()
                if isinstance(m.get("transformer_calls"), (int, float))]))
                if any(isinstance(m.get("transformer_calls"), (int, float))
                       for m in metadata_by_sample.values()) else ""),
            "mean_wall_seconds": (float(np.mean([
                m["wall_seconds"] for m in metadata_by_sample.values()
                if isinstance(m.get("wall_seconds"), (int, float))]))
                if any(isinstance(m.get("wall_seconds"), (int, float))
                       for m in metadata_by_sample.values()) else ""),
            "mean_pairwise_diversity": "",
        })
    by_run = {row["run"]: row for row in summaries}
    for group in diversity_groups.values():
        values = defaultdict(list)
        for (a, seed_a, clips_a), (b, seed_b, clips_b) in itertools.combinations(group, 2):
            if seed_a == seed_b:
                continue
            distances = [pairwise_diversity(clips_a[s], clips_b[s]) for s in sorted(common)]
            values[str(a)].extend(distances)
            values[str(b)].extend(distances)
        for run, distances in values.items():
            by_run[run]["mean_pairwise_diversity"] = float(np.mean(distances))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("sampling_comparison.csv"))
    args = parser.parse_args()
    rows = report(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} for {rows[0]['samples']} paired samples")


if __name__ == "__main__":
    main()
