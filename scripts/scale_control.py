#!/usr/bin/env python
"""How much of each statistic is just the trajectory's scale?

    python scripts/scale_control.py <run_dir>

Every GeoPhys statistic is computed on `v_t = z_{t+1} - z_t`, and most of them grow with
`‖v_t‖`: `phi_accel` and `phi_jerk` are means of squared norms, `phi_perr` a mean of norms.
So a representation whose trajectory simply *moves further* for violated clips will score
well on all of them without any of them measuring what they claim to.

This divides each statistic by the matching power of `mean(‖v_t‖)` and rescores. What
survives is the part that is not scale. What does not survive was one effect measured
several times.

It also explains `phi_energy`, which reads far *below* chance: it is `std(E)/mean(E)`, and
if the numerator is at chance while the denominator is a strong detector, the ratio is
inverted by construction. Nothing about energy exchange is required to produce that.

CPU only, and needs `probe.save_trajectories`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as stats_module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from phaselock.datasets import get_paired_dataset
from phaselock.metrics.geophys import geophys_signals
from phaselock.progress import track

# The power of the scale each statistic carries, from its definition. `phi_speed` is a
# standard deviation of norms and `phi_curv`/`phi_ang` are angles, so dividing by the mean
# speed is the right correction for the first and a no-op in spirit for the others -- they
# are already scale-free, which is why they are here as a control on the control.
SCALE_POWER = {
    "speed": 1, "curv": 0, "ang": 0, "accel": 2, "perr": 1,
    "energy": 0, "momentum": 0, "jerk": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source", default="hidden_states")
    parser.add_argument("--block-stride", type=int, default=5,
                        help="subsample blocks; every cell is a full statistic evaluation")
    parser.add_argument("--out", default="scale_control.csv")
    args = parser.parse_args()

    trajectories = args.run_dir / "trajectories"
    if not trajectories.is_dir():
        raise SystemExit(f"no trajectories/ under {args.run_dir}")

    config = json.loads((args.run_dir / "config.json").read_text())
    dataset = get_paired_dataset(config["data"]["name"])
    pairs = dataset.select(limit=config["data"]["limit"], seed=config["data"]["seed"])
    saved = {p.stem for p in trajectories.glob("*.npz")}
    usable = [p for p in pairs
              if p.plausible.sample_id.replace("/", "_") in saved
              and p.violated.sample_id.replace("/", "_") in saved]
    if not usable:
        raise SystemExit("no pairs with both clips saved")

    load = lambda s: np.load(trajectories / (s.sample_id.replace("/", "_") + ".npz"))
    available = [k for k in load(usable[0].plausible).keys()
                 if k.split("|")[0] == args.source]
    if not available:
        raise SystemExit(f"no {args.source} trajectories in {trajectories}")

    def block_of(key: str) -> int:
        part = key.split("|")[1]
        return int(part[1:]) if part.startswith("b") else -1

    blocks = sorted({block_of(k) for k in available})
    keep = set(blocks[:: max(1, args.block_stride)])
    cells = [k for k in available if block_of(k) in keep]
    print(f"{len(usable)} pairs x {len(cells)} cells of {args.source}\n")

    def quantities(record, key: str) -> dict[str, float]:
        z = torch.as_tensor(record[key].astype(np.float64))
        signals = geophys_signals(z, order=config["metrics"]["ar_order"],
                                  fit=config["metrics"]["residual_fit"])
        speed = np.linalg.norm(z[1:].numpy() - z[:-1].numpy(), axis=-1)
        scale = float(speed.mean())
        out = {"mean_speed": scale}
        for name, value in signals.statistics.items():
            if value is None:
                continue
            out[name] = float(value)
            out[f"{name} / scale"] = float(value) / scale ** SCALE_POWER.get(name, 0)
        return out

    hits: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))
    for pair in track(usable, "scale control", unit="pair"):
        plausible, violated = load(pair.plausible), load(pair.violated)
        for key in cells:
            low, high = quantities(plausible, key), quantities(violated, key)
            for name in low:
                hits[name][key][0] += int(high[name] > low[name])
                hits[name][key][1] += 1

    rows = []
    for name, per_cell in hits.items():
        values = [100 * got / total for got, total in per_cell.values()]
        rows.append({
            "quantity": name,
            "accuracy": round(stats_module.mean(values), 1),
            "std": round(stats_module.stdev(values) if len(values) > 1 else 0.0, 1),
            "n_cells": len(values),
            "n_pairs": len(usable),
        })
    rows.sort(key=lambda r: -r["accuracy"])

    out = args.run_dir / args.out
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'quantity':<26}{'accuracy':>10}{'s.d.':>8}")
    print("-" * 44)
    for row in rows:
        print(f"{row['quantity']:<26}{row['accuracy']:>9.1f}%{row['std']:>8.1f}")
    print(f"\nwrote {out}")
    print(
        "\n'X / scale' divides out mean(||v_t||) to the power X carries by definition.\n"
        "A statistic that keeps its margin measures something beyond how far the\n"
        "trajectory moved; one that falls to chance was measuring the distance."
    )


if __name__ == "__main__":
    main()
