#!/usr/bin/env python
"""One table across every guidance run, with the real Physics-IQ score.

    python scripts/summarise_guidance.py /data/experiments/phaselock_prior_ablation
    python scripts/summarise_guidance.py <root> --out summary.csv

Runs written before the score was wired up carry only the four raw metrics. The benchmark's
score divides each IoU by its **physical variance** -- the same metric measured between two
real takes of the scene -- and subtracts MSE after the same correction::

    score = mean(ST/var_ST, spatial/var_spatial, weighted/var_weighted) - (MSE - var_MSE)

That variance depends only on the real footage, so it can be computed after the fact, on
CPU, from take 2. This back-fills it for every row that lacks it and writes a single summary
comparing every (run, setting).

The pooling is the part that matters and the part that is easy to get wrong: **every term
above is a mean over all clips before any division.** Scoring clips individually and
averaging the results is a different and much worse estimator -- a scene where nothing moves
has a near-zero variance, and its per-clip ratio runs to thousands of percent and then
dominates the mean. Pooled, that scene contributes a small numerator and a small denominator
and carries exactly its own weight.

Variances are cached per sample, since the same clip appears in every setting of every run.
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

from phaselock.datasets.physics_iq import PhysicsIQ
from phaselock.datasets.video_io import load_video
from phaselock.metrics.motion_mask import (MotionMaskScores, motion_mask_scores,
                                           physics_iq_score)
from phaselock.progress import track

BENCHMARK_SECONDS = 5.0

# metric column -> physical-variance column, as run_physics_iq.py writes them.
VARIANCE_FIELDS = {
    "spatial_iou": "variance_spatial_iou",
    "spatiotemporal_iou": "variance_spatiotemporal_iou",
    "weighted_spatial_iou": "variance_weighted_spatial_iou",
    "mse": "variance_mse",
}
NUMERIC = (*VARIANCE_FIELDS, *VARIANCE_FIELDS.values(),
           "raw_score", "motion", "reference_motion", "prior_rms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="a directory containing guidance runs")
    parser.add_argument("--out", default="summary.csv")
    parser.add_argument("--no-backfill", action="store_true",
                        help="skip the variance computation and report raw metrics only")
    args, extra = parser.parse_known_args()
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args


def setting_of(row: dict) -> str:
    if row.get("guidance") == "baseline":
        return "baseline"
    if row.get("setting"):
        return row["setting"]
    kind = row.get("few_step_prior_type") or row.get("guidance", "?")
    return f"{kind}_on_{row.get('few_step_prior_source') or 'latent'}"


def run_of(path: Path, root: Path) -> str:
    """The run id: the first path segment under the root."""
    return path.relative_to(root).parts[0]


class Variances:
    """Real-vs-real scores per sample, computed once and reused everywhere.

    The physical variance is a property of the benchmark footage, not of any run or
    setting, so a hundred CSV rows referring to the same clip need one computation between
    them.
    """

    def __init__(self, geometry: tuple[int, int, int], cache_path: Path,
                 testing_fps: int = 30):
        self.frames, self.height, self.width = geometry
        # 30 FPS is the dataset default, and therefore what run_physics_iq.py scored the
        # generations against. The variance has to be measured on the same footage or the
        # ratio mixes two different references.
        self.dataset = PhysicsIQ(testing_fps=testing_fps)
        self.by_id = {s.sample_id: s for s in self.dataset.generation_samples()}
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path.is_file():
            self._cache = json.loads(cache_path.read_text())

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1, sort_keys=True))

    def get(self, sample_id: str) -> MotionMaskScores | None:
        if sample_id not in self._cache:
            sample = self.by_id.get(sample_id)
            scores = None
            if sample is not None and sample.reference_path and sample.meta.get("pair_path"):
                kwargs = dict(num_frames=self.frames, height=self.height, width=self.width)
                take_one = load_video(sample.reference_path, **kwargs)
                take_two = load_video(sample.meta["pair_path"], **kwargs)
                scores = motion_mask_scores(take_two, take_one)
            self._cache[sample_id] = scores.__dict__ if scores else None
        cached = self._cache[sample_id]
        return MotionMaskScores(**cached) if cached else None


def geometry_from(paths: list[Path]) -> tuple[int, int, int]:
    """Score at the geometry the runs used, read from a run's own config.json."""
    from phaselock.backends import get_spec

    for path in paths:
        config_path = path.parent / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        spec = get_spec(config["backend"]["name"])
        return (round(BENCHMARK_SECONDS * spec.default_fps),
                spec.default_height, spec.default_width)
    raise SystemExit("no config.json beside any CSV; cannot tell what geometry to score at")


def score(rows: list[dict]) -> float:
    """The pooled Physics-IQ score for one (run, setting)."""
    return physics_iq_score(
        [MotionMaskScores(**{k: r[k] for k in VARIANCE_FIELDS}) for r in rows],
        [MotionMaskScores(**{k: r[v] for k, v in VARIANCE_FIELDS.items()}) for r in rows],
    )


def main() -> None:
    args = parse_args()
    paths = sorted(args.root.rglob("physics_iq*.csv"))
    if not paths:
        raise SystemExit(f"no physics_iq*.csv under {args.root}")

    rows: list[dict] = []
    for path in paths:
        run = run_of(path, args.root)
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                for key in NUMERIC:
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                row["run"] = run
                row["setting"] = setting_of(row)
                rows.append(row)

    def complete(row: dict) -> bool:
        return all(isinstance(row.get(v), float) and row[v] == row[v]
                   for v in VARIANCE_FIELDS.values())

    missing = [r for r in rows if not complete(r)]
    if missing and not args.no_backfill:
        variances = Variances(geometry_from(paths), args.root / "physical_variance.json")
        ids = sorted({r["sample_id"] for r in missing})
        fresh = [i for i in ids if i not in variances._cache]
        print(f"back-filling the real-vs-real physical variance for {len(ids)} clips "
              f"({len(missing)} rows lack it)")
        if fresh:
            print(f"  {len(ids) - len(fresh)} already cached, computing {len(fresh)} "
                  f"(each decodes two 4K clips)")
        for sample_id in track(ids, "variance", unit="clip"):
            variances.get(sample_id)
        variances.save()
        for row in missing:
            reference = variances.get(row["sample_id"])
            for key, column in VARIANCE_FIELDS.items():
                row[column] = getattr(reference, key) if reference else float("nan")

    scored = [r for r in rows if complete(r)]
    dropped = sorted({r["sample_id"] for r in rows} - {r["sample_id"] for r in scored})
    if dropped:
        print(f"\n{len(dropped)} clips have no second real take and are excluded: "
              f"{', '.join(dropped)}")
    if not scored:
        raise SystemExit("no rows could be scored; is the Physics-IQ dataset present?")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored:
        grouped[(row["run"], row["setting"])].append(row)

    # Per run: the baseline, then everything else by score. Absolute scores move with the
    # clip draw, so only settings within one run are comparable to each other.
    print()
    out = []
    for run in sorted({r for r, _ in grouped}):
        settings = [s for r, s in grouped if r == run]
        base = grouped.get((run, "baseline"), [])
        baseline_score = score(base) if base else float("nan")

        ordered = ["baseline"] if base else []
        ordered += sorted((s for s in settings if s != "baseline"),
                          key=lambda s: -score(grouped[(run, s)]))

        clips = min(len(grouped[(run, s)]) for s in ordered)
        width = max(len(s) for s in ordered) + 2
        print(f"{run}   ({clips} clips per setting)")
        print(f"  {'setting':<{width}}{'PhysicsIQ %':>13}{'vs base':>10}{'motion':>9}"
              f"{'prior RMS':>11}")
        print("  " + "-" * (width + 43))
        reference = stats_module.mean(r["reference_motion"] for r in scored
                                      if r["run"] == run)
        for setting in ordered:
            group = grouped[(run, setting)]
            value = score(group)
            motion = stats_module.mean(r["motion"] for r in group)
            delta = ("" if setting == "baseline" or baseline_score != baseline_score
                     else f"{value - baseline_score:+.2f}")
            rms = [r["prior_rms"] for r in group
                   if isinstance(r.get("prior_rms"), float) and r["prior_rms"] == r["prior_rms"]]
            line = (f"  {setting:<{width}}{value:>13.2f}{delta:>10}{motion:>9.4f}"
                    f"{(stats_module.mean(rms) if rms else float('nan')):>11.5f}")
            if motion < 0.5 * reference:
                line += "   MOTION COLLAPSE"
            print(line)
            out.append({
                "run": run, "setting": setting, "n_clips": len(group),
                "physics_iq_score": round(value, 3),
                "vs_baseline": delta,
                **{k: round(stats_module.mean(r[k] for r in group), 5)
                   for k in VARIANCE_FIELDS},
                **{v: round(stats_module.mean(r[v] for r in group), 5)
                   for v in VARIANCE_FIELDS.values()},
                "motion": round(motion, 5),
                "reference_motion": round(reference, 5),
                "prior_rms": round(stats_module.mean(rms), 5) if rms else "",
            })
        print(f"\n  real continuations move {reference:.4f} per frame.\n")

    path = args.root / args.out
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {path}")
    print("\nPhysicsIQ % is the benchmark's own score: each IoU against the level two real\n"
          "takes of the scene agree, MSE subtracted. 0 = no overlap with reality,\n"
          "100 = matched a second real take. Pooled over clips, then clipped to [0, 100].")


if __name__ == "__main__":
    main()
