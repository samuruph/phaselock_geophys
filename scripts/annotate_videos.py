#!/usr/bin/env python
"""Draw a run's computed signals underneath the videos it already wrote.

    python scripts/annotate_videos.py /data/experiments/.../likephys/inversion
    python scripts/annotate_videos.py <run_dir> --location hidden_states/b22

Pure post-processing: reads ``statistics.csv`` and the mp4s in ``visuals/``, and writes
``*_signals.mp4`` beside them. Seconds of CPU, no GPU, nothing recomputed. Safe to re-run
with a different ``--location`` to look at another probe point.

What it can and cannot show is set by what the run stored; see the module docstring of
``phaselock/analysis/signal_overlay.py``. In short: the x-axis is the **denoising step**,
because the per-video-frame intermediates are summarised away before anything is written
to disk, and there is no spatial map, because pooling happens inside the forward hook.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phaselock.analysis import signal_overlay
from phaselock.datasets import get_paired_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--location", default=None, metavar="SOURCE[/bN]",
        help="probe point to draw, e.g. hidden_states/b22 or latent. "
             "Default: the middle hidden-state block.",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--static", action="store_true",
        help="one fixed strip at a single block, instead of the animated depth summary",
    )
    parser.add_argument(
        "--no-timeline", action="store_true",
        help="skip the per-video-frame view even when trajectories were saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run_dir

    statistics = run / "statistics.csv"
    if not statistics.is_file():
        raise SystemExit(f"no statistics.csv under {run}")
    videos = next((d for d in (run / "visuals", run / "videos") if d.is_dir()), None)
    if videos is None:
        raise SystemExit(f"no visuals/ or videos/ under {run}; run with --save-visuals")

    with open(statistics, newline="") as handle:
        rows = list(csv.DictReader(handle))
    source, block = signal_overlay.choose_location(rows, args.location)
    taus = {int(r["step"]): float(r["tau"]) for r in rows}
    print(f"drawing {source} block {block} over {len({r['step'] for r in rows})} steps")

    config = json.loads((run / "config.json").read_text())
    dataset = get_paired_dataset(config["data"]["name"])
    pairs = dataset.select(limit=config["data"]["limit"], seed=config["data"]["seed"])

    # Trajectories are what make video time a usable axis: statistics.csv only keeps the
    # per-clip summaries, and a mean cannot be inverted into the values behind it.
    trajectories = run / "trajectories"
    spec = None
    if trajectories.is_dir() and not args.no_timeline:
        from phaselock.backends import get_spec

        spec = get_spec(config["backend"]["name"])
        print(f"found trajectories: also drawing per-video-frame signals")

    written = 0
    for pair in pairs:
        stem = pair.violated.sample_id.replace("/", "_")
        plausible = signal_overlay.per_step(rows, pair.plausible.sample_id, source, block)
        violated = signal_overlay.per_step(rows, pair.violated.sample_id, source, block)
        if not plausible or not violated:
            continue

        # One strip per pair, reused across that pair's videos: the signals belong to the
        # clips, not to a particular rendering of them.
        depth_plausible = signal_overlay.across_depth(rows, pair.plausible.sample_id, source)
        depth_violated = signal_overlay.across_depth(rows, pair.violated.sample_id, source)

        # GeoPhys Figure 4's view: each signal against video time for both clips, with the
        # signed gap filled, so *when* the detector fires can be lined up against what is
        # happening on screen.
        timeline = None
        if spec is not None:
            from phaselock.probes import ProbeRecord

            try:
                low = signal_overlay.per_video_frame(
                    ProbeRecord.load(trajectories / pair.plausible.sample_id.replace("/", "_")),
                    spec, source)
                high = signal_overlay.per_video_frame(
                    ProbeRecord.load(trajectories / pair.violated.sample_id.replace("/", "_")),
                    spec, source)
            except FileNotFoundError:
                low = high = {}
            if low and high:
                timeline = (low, high)
                figure = signal_overlay.timeline_figure(
                    low, high, videos / f"{stem}_timeline.png",
                    title=f"{pair.scenario} - {pair.violation}   ({source})")
                if figure:
                    print(f"  wrote {figure.name}")
                    written += 1

        cached = None
        for suffix in ("pair", "inversion", "roundtrip"):
            path = videos / f"{stem}_{suffix}.mp4"
            if not path.is_file():
                continue

            import cv2

            capture = cv2.VideoCapture(str(path))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

            out_path = videos / f"{stem}_{suffix}_signals.mp4"
            if args.static:
                caption = (
                    f"{source} block {block}. x = denoising timestep (0 = clean video, "
                    f"1000 = noise), not video time. No spatial map: activations are "
                    f"mean-pooled over space before any statistic exists."
                )
                if cached is None or cached[0] != width:
                    cached = (width, signal_overlay.signal_strip(
                        plausible, violated, width, taus=taus, caption=caption))
                out = signal_overlay.attach(path, out_path, cached[1], fps=args.fps)
            elif timeline is not None:
                strips = signal_overlay.timeline_strips(
                    timeline[0], timeline[1], width,
                    caption=(
                        f"Per-frame signal from {source}, averaged over depth. Green = "
                        f"plausible, red = violated, violet = the gap (positive means the "
                        f"detector is right at that frame). The line marks the frame above."
                    ))
                out = signal_overlay.attach_animated(path, out_path, strips, fps=args.fps)
            else:
                caption = (
                    f"Line = mean over all {source} blocks, band = 1 s.d. across depth. "
                    f"x = denoising timestep (0 = clean video, 1000 = noise), which does "
                    f"not advance with video time -- the reveal is a reading aid, not a "
                    f"correspondence. No spatial map: activations are mean-pooled over "
                    f"space before any statistic exists."
                )
                strips = signal_overlay.animated_strips(
                    depth_plausible or plausible, depth_violated or violated,
                    width, frames=count, taus=taus, caption=caption)
                out = signal_overlay.attach_animated(path, out_path, strips, fps=args.fps)
            print(f"  wrote {out.name}")
            written += 1

    print(f"\n{written} annotated videos in {videos}")


if __name__ == "__main__":
    main()
