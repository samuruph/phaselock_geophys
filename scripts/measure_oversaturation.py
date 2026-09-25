#!/usr/bin/env python
"""Measure colour oversaturation in matched Physics-IQ videos.

The evaluator reports both ordinary saturation and the more specific failure mode of
bright, highly saturated pixels.  All methods are restricted to the intersection of
sample IDs present in the real, baseline, PhaseLock, and Ours directories.

Example::

    python scripts/measure_oversaturation.py \
        --real-root /data/datasets/physics-IQ-benchmark-verified \
        --baseline-root /data/experiments/.../videos/baseline \
        --phaselock-root /data/experiments/phaselock/physics_iq \
        --ours-root /data/experiments/.../videos/motion_on_latent \
        --output-dir /data/experiments/oversaturation/physics_iq

The real root is the Physics-IQ dataset root, not a directory of manually renamed
videos.  Generated videos may be nested below their supplied root; their sample ID is
the leading four-digit component of the filename.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# When invoked as ``python scripts/measure_oversaturation.py``, Python puts ``scripts/``
# on sys.path rather than the repository root.  Keep the evaluator runnable from any
# shell/conda environment without requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from phaselock.datasets.physics_iq import PhysicsIQ


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SAMPLE_ID = re.compile(r"^(?P<id>\d{4})(?:_|\.)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True,
                        help="Physics-IQ dataset root containing split-videos/")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--phaselock-root", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=49,
                        help="maximum uniformly sampled frames per clip (default: 49)")
    parser.add_argument("--resize", type=int, default=256,
                        help="resize the longest frame edge before measuring (default: 256)")
    parser.add_argument("--saturation-threshold", type=float, default=0.90,
                        help="HSV saturation threshold for high-saturation rates")
    parser.add_argument("--brightness-threshold", type=float, default=0.80,
                        help="HSV value threshold for bright high-saturation rate")
    parser.add_argument("--no-plots", action="store_true",
                        help="write CSVs only; skip PNG plots")
    return parser.parse_args()


def generated_videos(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise SystemExit(f"video root does not exist or is not a directory: {root}")
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        match = SAMPLE_ID.match(path.name)
        if not match:
            continue
        sample_id = match.group("id")
        if sample_id in found:
            raise SystemExit(f"multiple videos found for sample {sample_id} under {root}: "
                             f"{found[sample_id]} and {path}")
        found[sample_id] = path
    return found


def real_videos(root: Path) -> dict[str, Path]:
    dataset = PhysicsIQ(root=root)
    videos = {}
    for sample in dataset.generation_samples():
        if sample.reference_path:
            videos[sample.sample_id] = Path(sample.reference_path)
    return videos


def frame_indices(count: int, max_frames: int) -> list[int]:
    if count <= 0:
        return []
    if count <= max_frames:
        return list(range(count))
    return np.linspace(0, count - 1, max_frames).round().astype(int).tolist()


def measure_video(path: Path, max_frames: int, resize: int,
                  saturation_threshold: float, brightness_threshold: float) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = frame_indices(count, max_frames)
    if not indices:
        capture.release()
        raise ValueError(f"video has no frames: {path}")

    sums = defaultdict(float)
    pixels = 0
    used = 0
    wanted = set(indices)
    try:
        for index in range(max(indices) + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index not in wanted:
                continue
            if resize > 0:
                height, width = frame.shape[:2]
                longest = max(height, width)
                if longest > resize:
                    scale = resize / longest
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                       interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            # For float32 input OpenCV already returns HSV S and V in [0, 1].
            saturation = hsv[..., 1]
            value = hsv[..., 2]
            max_rgb = rgb.max(axis=2)
            min_rgb = rgb.min(axis=2)
            high = saturation >= saturation_threshold
            bright_high = high & (value >= brightness_threshold)
            clipped = (max_rgb >= 0.98) & ((max_rgb - min_rgb) >= 0.05)
            sums["mean_saturation"] += float(saturation.sum())
            sums["mean_value"] += float(value.sum())
            sums["high_saturation_fraction"] += float(high.sum())
            sums["bright_high_saturation_fraction"] += float(bright_high.sum())
            sums["rgb_clipping_fraction"] += float(clipped.sum())
            sums["mean_chroma"] += float((max_rgb - min_rgb).sum())
            pixels += saturation.size
            used += 1
    finally:
        capture.release()

    if pixels == 0:
        raise ValueError(f"could not decode sampled frames from {path}")
    return {
        "video": str(path),
        "frames": used,
        "pixels": pixels,
        "mean_saturation": sums["mean_saturation"] / pixels,
        "mean_value": sums["mean_value"] / pixels,
        "high_saturation_fraction": sums["high_saturation_fraction"] / pixels,
        "bright_high_saturation_fraction": sums["bright_high_saturation_fraction"] / pixels,
        "rgb_clipping_fraction": sums["rgb_clipping_fraction"] / pixels,
        "mean_chroma": sums["mean_chroma"] / pixels,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, rows: list[dict], summary: list[dict], metrics: list[str]) -> None:
    """Write small summary bars and a per-sample heatmap when Matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed; skipped PNG plots")
        return

    sources = [row["source"] for row in summary]
    labels = {
        "mean_saturation": "Mean saturation",
        "high_saturation_fraction": "Saturation >= 0.90",
        "bright_high_saturation_fraction": "Bright + saturated",
        "rgb_clipping_fraction": "RGB clipping",
        "mean_chroma": "Mean chroma",
    }
    plot_metrics = ["mean_saturation", "high_saturation_fraction",
                    "bright_high_saturation_fraction", "rgb_clipping_fraction", "mean_chroma"]

    figure, axes = plt.subplots(1, len(plot_metrics), figsize=(14, 3.4), constrained_layout=True)
    for axis, metric in zip(axes, plot_metrics):
        values = [[float(row[f"mean_{metric}"]) for row in summary]]
        axis.bar(sources, values[0], color=["#777777", "#4c78a8", "#f58518", "#54a24b"])
        axis.set_title(labels[metric], fontsize=9)
        axis.tick_params(axis="x", labelrotation=45, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "oversaturation_summary.png", dpi=160)
    plt.close(figure)

    sample_ids = sorted({row["sample_id"] for row in rows})
    heat_metric = "bright_high_saturation_fraction"
    matrix = np.asarray([
        [next(row[heat_metric] for row in rows
             if row["sample_id"] == sample_id and row["source"] == source)
         for source in sources]
        for sample_id in sample_ids
    ], dtype=float)
    figure, axis = plt.subplots(
        figsize=(6.5, max(3.0, 0.22 * len(sample_ids))), constrained_layout=True
    )
    image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=max(0.01, float(matrix.max())))
    axis.set_xticks(range(len(sources)), sources, rotation=45, ha="right")
    axis.set_yticks(range(len(sample_ids)), sample_ids, fontsize=7)
    axis.set_title("Bright/high-saturation fraction per sample")
    figure.colorbar(image, ax=axis, label="fraction")
    figure.savefig(output_dir / "oversaturation_per_sample.png", dpi=160)
    plt.close(figure)
    print(f"wrote {output_dir / 'oversaturation_summary.png'}")
    print(f"wrote {output_dir / 'oversaturation_per_sample.png'}")


def main() -> None:
    args = parse_args()
    if args.max_frames < 1 or args.resize < 0:
        raise SystemExit("--max-frames must be positive and --resize must be non-negative")
    if not 0 < args.saturation_threshold <= 1 or not 0 < args.brightness_threshold <= 1:
        raise SystemExit("thresholds must be in (0, 1]")

    sources = {
        "real": real_videos(args.real_root),
        "baseline": generated_videos(args.baseline_root),
        "phaselock": generated_videos(args.phaselock_root),
        "ours": generated_videos(args.ours_root),
    }
    common = sorted(set.intersection(*(set(paths) for paths in sources.values())))
    if not common:
        details = ", ".join(f"{name}={len(paths)}" for name, paths in sources.items())
        raise SystemExit(f"no common sample IDs ({details})")
    print("common samples:", " ".join(common))

    rows = []
    metrics = ["mean_saturation", "mean_value", "high_saturation_fraction",
               "bright_high_saturation_fraction", "rgb_clipping_fraction", "mean_chroma"]
    for sample_id in common:
        measured = {}
        for name, paths in sources.items():
            print(f"measuring {name:9s} {sample_id} {paths[sample_id]}")
            measured[name] = measure_video(paths[sample_id], args.max_frames, args.resize,
                                           args.saturation_threshold, args.brightness_threshold)
        for name, result in measured.items():
            row = {"sample_id": sample_id, "source": name, **result}
            for metric in metrics:
                row[f"delta_vs_real_{metric}"] = (
                    result[metric] - measured["real"][metric] if name != "real" else ""
                )
            rows.append(row)

    summary = []
    for source in sources:
        source_rows = [row for row in rows if row["source"] == source]
        row = {"source": source, "samples": len(source_rows)}
        for metric in metrics:
            values = np.asarray([item[metric] for item in source_rows], dtype=float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"median_{metric}"] = float(np.median(values))
        if source != "real":
            for metric in metrics:
                values = np.asarray([item[f"delta_vs_real_{metric}"] for item in source_rows], dtype=float)
                row[f"mean_delta_vs_real_{metric}"] = float(values.mean())
        summary.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "oversaturation_per_sample.csv", rows)
    write_csv(args.output_dir / "oversaturation_summary.csv", summary)
    if not args.no_plots:
        write_plots(args.output_dir, rows, summary, metrics)
    print(f"wrote {args.output_dir / 'oversaturation_per_sample.csv'}")
    print(f"wrote {args.output_dir / 'oversaturation_summary.csv'}")


if __name__ == "__main__":
    main()
