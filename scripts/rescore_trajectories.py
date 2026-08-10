#!/usr/bin/env python
"""Recompute every statistic from saved trajectories, without touching the GPU.

    python scripts/rescore_trajectories.py <run_dir>

A run stores its statistics as computed at the time. When a new statistic is added --
energy, momentum and jerk were -- every earlier run is missing it, and re-inverting to get
it back costs hours per hundred clips. The full `(T, D)` trajectories are on disk whenever
`probe.save_trajectories` was set, and every statistic is a pure function of them, so the
whole table can be rebuilt on CPU in seconds.

Writes `statistics_rescored.csv` beside the original rather than over it, so the numbers a
run actually reported stay recoverable, and prints the per-source accuracy table.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phaselock.datasets import get_paired_dataset
from phaselock.experiments.detection import statistics_from_record
from phaselock.metrics.geophys import STATISTICS
from phaselock.probes import ProbeRecord, StatisticRow
from phaselock.progress import track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", default="statistics_rescored.csv")
    args = parser.parse_args()

    trajectories = args.run_dir / "trajectories"
    if not trajectories.is_dir():
        raise SystemExit(
            f"no trajectories/ under {args.run_dir}. Only runs with "
            "probe__save_trajectories=true can be rescored."
        )

    config = json.loads((args.run_dir / "config.json").read_text())
    dataset = get_paired_dataset(config["data"]["name"])
    pairs = dataset.select(limit=config["data"]["limit"], seed=config["data"]["seed"])
    saved = {p.stem for p in trajectories.glob("*.npz")}

    usable = [p for p in pairs
              if p.plausible.sample_id.replace("/", "_") in saved
              and p.violated.sample_id.replace("/", "_") in saved]
    if not usable:
        raise SystemExit(f"none of the {len(pairs)} pairs have both clips saved")
    print(f"{len(usable)} pairs with saved trajectories, "
          f"{len(STATISTICS)} statistics: {', '.join(STATISTICS)}\n")

    rows: list[StatisticRow] = []
    for pair in track(usable, "rescoring", unit="pair"):
        for sample in (pair.plausible, pair.violated):
            record = ProbeRecord.load(trajectories / sample.sample_id.replace("/", "_"))
            rows.extend(statistics_from_record(
                record, sample, pair,
                order=config["metrics"]["ar_order"],
                fit=config["metrics"]["residual_fit"],
            ))

    out = args.run_dir / args.out
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=StatisticRow.fieldnames())
        writer.writeheader()
        writer.writerows(row.flatten() for row in rows)
    print(f"\nwrote {out} ({len(rows)} rows)\n")

    # Accuracy per (source, statistic), meaned over the probe grid -- the same summary the
    # source-comparison bars use.
    by_cell: dict[tuple, dict[str, float]] = collections.defaultdict(dict)
    labels = {}
    for row in rows:
        labels[row.sample_id] = row.label
        for name, value in row.statistics.items():
            by_cell[(row.source, row.block, row.step, name)][row.sample_id] = value

    hits: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for (source, _, _, name), values in by_cell.items():
        for pair in usable:
            low = values.get(pair.plausible.sample_id)
            high = values.get(pair.violated.sample_id)
            if low is None or high is None:
                continue
            hits[(source, name)][0] += int(high > low)
            hits[(source, name)][1] += 1

    print("pairwise accuracy, meaned over the probe grid")
    print(f"{'source':<16}" + "".join(f"{n:>10}" for n in STATISTICS))
    for source in sorted({s for s, _ in hits}):
        line = f"{source:<16}"
        for name in STATISTICS:
            got, total = hits.get((source, name), (0, 0))
            line += f"{100 * got / total:9.1f}%" if total else "        - "
        print(line)
    print("\nBelow 50% is not failure: it is the same discriminative power with the sign "
          "inverted.\nThe orientation assumed everywhere is larger = less plausible.")


if __name__ == "__main__":
    main()
