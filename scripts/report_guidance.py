#!/usr/bin/env python
"""One table comparing every guidance setting's Physics-IQ result.

    python scripts/report_guidance.py <run_dir>
    python scripts/report_guidance.py <run_dir> --official /data/experiments/.../results

Each setting is a `(few_step_prior_type, few_step_prior_source)` pair plus the unguided
baseline, and every one generated the same clips from the same seed. So the number that
matters is the **paired delta against baseline**, not the absolute score: per-clip
differences remove the clip-to-clip variance that otherwise swamps a few points.

Two things are printed alongside every score, and both can retire a result:

* **motion**, against the real continuation's. Every geometric statistic is minimised by a
  static video, so a setting that gains fidelity while its motion collapses has found the
  degenerate optimum rather than better physics.
* **prior RMS**, the size of the target the setting matched. `lambda` is shared across
  settings, so if these differ by orders of magnitude the same `lambda` was a different
  intervention and the ranking is not yet a fair comparison.

``--official`` folds in the Physics-IQ repo's own score (`final_score_view` from its
``<run>_metrics.json``) so the two implementations sit in one table. They should agree
closely; a gap is a bug in one of them and worth chasing before reporting anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats_module
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASELINE = "baseline"

# (column, label, higher_is_better). `motion` is a control, never a score.
METRICS = [
    ("physics_iq_score", "PhysicsIQ score %", True),
    ("raw_score", "raw (un-normalised)", True),
    ("spatial_iou", "spatial IoU", True),
    ("spatiotemporal_iou", "spatiotemporal IoU", True),
    ("weighted_spatial_iou", "weighted IoU", True),
    ("mse", "MSE", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a guidance run, or any parent of one")
    parser.add_argument("--official", type=Path, default=None,
                        help="the Physics-IQ repo's results directory, to fold in its own "
                             "score for settings that have been evaluated there")
    parser.add_argument("--out", default="guidance_comparison.csv")
    args, extra = parser.parse_known_args()
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args


def setting_of(row: dict) -> str:
    """A stable, readable name for one cell of the grid."""
    if row.get("guidance") == BASELINE:
        return BASELINE
    kind = row.get("few_step_prior_type") or row.get("guidance", "?")
    source = row.get("few_step_prior_source") or "latent"
    return f"{kind} on {source}"


def load_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.rglob("physics_iq_*.csv")):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou",
                            "mse", "raw_score", "motion", "reference_motion", "prior_rms"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                row["setting"] = setting_of(row)
                rows.append(row)
    return rows


def official_scores(results_dir: Path) -> dict[str, float]:
    """``final_score_view`` per run name -- the number the leaderboard reports."""
    scores: dict[str, float] = {}
    for path in sorted(results_dir.rglob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        value = payload.get("final_score_view", payload.get("final_score_raw"))
        if value is not None:
            scores[path.name.removesuffix("_metrics.json")] = float(value)
    return scores


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_dir)
    if not rows:
        raise SystemExit(f"no physics_iq_*.csv under {args.run_dir}")

    by_setting: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_setting[row["setting"]].append(row)

    # Baseline first, then the guided settings by score. Sorting by the thing being
    # compared is the point of the table.
    guided = sorted(
        (s for s in by_setting if s != BASELINE),
        key=lambda s: -stats_module.mean(r["physics_iq_score"] for r in by_setting[s]),
    )
    order = ([BASELINE] if BASELINE in by_setting else []) + guided

    control = {}
    if BASELINE in by_setting:
        for column, _, _ in METRICS:
            control[column] = {r["sample_id"]: r[column] for r in by_setting[BASELINE]}

    official = official_scores(args.official) if args.official else {}

    clips = min(len(by_setting[s]) for s in order)
    print(f"\n{len(order)} settings, {clips} clips each, paired on one seed\n")

    width = max(len(s) for s in order) + 2
    header = (f"{'setting':<{width}}{'PhysicsIQ':>11}{'vs base':>10}{'motion':>9}"
              f"{'prior RMS':>11}")
    if official:
        header += f"{'official':>10}"
    print(header)
    print("-" * len(header))

    reference_motion = stats_module.mean(r["reference_motion"] for r in rows)
    for setting in order:
        group = by_setting[setting]
        score = stats_module.mean(r["physics_iq_score"] for r in group)
        motion = stats_module.mean(r["motion"] for r in group)

        delta = ""
        if control and setting != BASELINE:
            paired = [r["physics_iq_score"] - control["physics_iq_score"][r["sample_id"]]
                      for r in group if r["sample_id"] in control["physics_iq_score"]]
            if paired:
                delta = f"{stats_module.mean(paired):+.4f}"

        rms = [r["prior_rms"] for r in group
               if isinstance(r.get("prior_rms"), float) and r["prior_rms"] == r["prior_rms"]]
        line = (f"{setting:<{width}}{score:>11.4f}{delta:>10}{motion:>9.4f}"
                f"{(stats_module.mean(rms) if rms else float('nan')):>11.5f}")
        if official:
            line += f"{official.get(setting, float('nan')):>10.4f}"
        # A collapse is the failure mode the whole control exists for, so it is called out
        # on the row rather than left to be noticed.
        if motion < 0.5 * reference_motion:
            line += "   MOTION COLLAPSE"
        print(line)

    print(f"\nreal continuations move {reference_motion:.4f} per frame.")
    print("A setting that scores better while moving markedly less has found the static\n"
          "optimum, not better physics.")

    if not official:
        print("\nFor the official score, point the Physics-IQ repo at a setting's videos:")
        print("  cd /home/ec2-user/code/physics-IQ-benchmark")
        print("  ./run_eval.sh <run_dir>/videos/<setting> <setting>")
        print("then re-run this with --official <its results dir> to see both columns.")

    # The full breakdown, for anything that wants to pivot rather than read.
    path = args.run_dir / args.out
    fields = ["setting", "metric", "mean", "paired_delta", "n_clips"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in order:
            group = by_setting[setting]
            for column, label, _ in METRICS + [("motion", "motion (control)", None)]:
                paired = ""
                if control.get(column) and setting != BASELINE:
                    values = [r[column] - control[column][r["sample_id"]]
                              for r in group if r["sample_id"] in control[column]]
                    if values:
                        paired = round(stats_module.mean(values), 6)
                writer.writerow({
                    "setting": setting, "metric": label,
                    "mean": round(stats_module.mean(r[column] for r in group), 6),
                    "paired_delta": paired, "n_clips": len(group),
                })
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
