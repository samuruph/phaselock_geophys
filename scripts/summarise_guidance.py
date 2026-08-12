#!/usr/bin/env python
"""One table across every guidance run, with the real Physics-IQ score.

    python scripts/summarise_guidance.py /data/experiments/phaselock_prior_ablation
    python scripts/summarise_guidance.py <root> --out summary.csv

Runs written before the score was wired up carry only `raw_score`, which is
``sum(IoU) - MSE`` and is **not** the benchmark's number: it is un-normalised, so it
cannot be compared across scenes. The benchmark divides it by the real-vs-real ceiling --
a second real recording of the same scene scored against the first -- because reality
repeating itself does not score 100% on a motion mask.

That ceiling depends only on the real footage, so it can be computed after the fact, on
CPU, from take-2. This back-fills it for every row that lacks one and writes a single
summary comparing every (run, setting).

Ceilings are cached per sample, since the same clip appears in every setting of every run.
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
from phaselock.metrics.motion_mask import motion_mask_scores
from phaselock.progress import track

BENCHMARK_SECONDS = 5.0

# A ceiling this small means the two real takes of the scene barely overlap in motion-mask
# terms -- nothing moved, or the takes are not aligned -- so the scene has no usable scale
# and dividing by it turns a 0.1 raw score into 3000%. Those clips are reported as
# unscoreable rather than as spectacular results.
MIN_CEILING = 0.05
NUMERIC = ("spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou", "mse",
           "raw_score", "physics_iq_score", "ceiling_raw_score", "motion",
           "reference_motion", "prior_rms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="a directory containing guidance runs")
    parser.add_argument("--out", default="summary.csv")
    parser.add_argument("--no-backfill", action="store_true",
                        help="skip the ceiling computation and report raw scores only")
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


class Ceilings:
    """Real-vs-real scores per sample, computed once and reused everywhere.

    The ceiling is a property of the benchmark footage, not of any run or setting, so a
    hundred CSV rows referring to the same clip need one computation between them.
    """

    def __init__(self, geometry: tuple[int, int, int], cache_path: Path,
                 testing_fps: int = 30):
        self.frames, self.height, self.width = geometry
        # 30 FPS is the dataset default, and therefore what run_physics_iq.py scored the
        # generations against. The ceiling has to be measured on the same footage or the
        # ratio mixes two different references. (The 8 FPS variant is the same scene at
        # exactly the benchmark window; it is a defensible choice for future runs, but it
        # is not the one these CSVs used.)
        self.dataset = PhysicsIQ(testing_fps=testing_fps)
        self.by_id = {s.sample_id: s for s in self.dataset.generation_samples()}
        # Cached on disk, not just in memory: the ceiling is a property of the benchmark
        # footage, so it is the same for every setting, every run, and every future run.
        self.cache_path = cache_path
        self._cache: dict[str, float] = {}
        if cache_path.is_file():
            self._cache = {k: float(v) for k, v in json.loads(cache_path.read_text()).items()}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1, sort_keys=True))

    def get(self, sample_id: str) -> float:
        if sample_id in self._cache:
            return self._cache[sample_id]
        sample = self.by_id.get(sample_id)
        value = float("nan")
        if sample is not None and sample.reference_path and sample.meta.get("pair_path"):
            kwargs = dict(num_frames=self.frames, height=self.height, width=self.width)
            take_one = load_video(sample.reference_path, **kwargs)
            take_two = load_video(sample.meta["pair_path"], **kwargs)
            value = motion_mask_scores(take_two, take_one).raw_score
        self._cache[sample_id] = value
        return value


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

    missing = [r for r in rows if not isinstance(r.get("physics_iq_score"), float)]
    if missing and not args.no_backfill:
        ceilings = Ceilings(geometry_from(paths), args.root / "ceilings.json")
        ids = sorted({r["sample_id"] for r in missing})
        print(f"back-filling the real-vs-real ceiling for {len(ids)} clips "
              f"({len(missing)} rows lack a score)")
        fresh = [i for i in ids if i not in ceilings._cache]
        if fresh:
            print(f"  {len(ids) - len(fresh)} already cached, computing {len(fresh)} "
                  f"(each decodes two 4K clips, ~10 s)")
        for sample_id in track(ids, "ceilings", unit="clip"):
            ceilings.get(sample_id)
        ceilings.save()
        for row in missing:
            ceiling = ceilings.get(row["sample_id"])
            row["ceiling_raw_score"] = ceiling
            usable = ceiling == ceiling and ceiling > MIN_CEILING
            row["physics_iq_score"] = (
                100.0 * row["raw_score"] / ceiling if usable else float("nan")
            )

    scored = [r for r in rows if isinstance(r.get("physics_iq_score"), float)
              and r["physics_iq_score"] == r["physics_iq_score"]]
    dropped = sorted({r["sample_id"] for r in rows} - {r["sample_id"] for r in scored})
    if dropped:
        print(f"\n{len(dropped)} clips have no usable real-vs-real ceiling and are "
              f"excluded: {', '.join(dropped)}")
        print("  (a ceiling near zero means the two real takes barely overlap, so the "
              "scene\n   has no scale to normalise against)")
    if not scored:
        raise SystemExit("no rows could be scored; is the Physics-IQ dataset present?")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored:
        grouped[(row["run"], row["setting"])].append(row)

    # Per run: the baseline, then everything else by score. The paired delta against that
    # run's own baseline is the number to read -- absolute scores move with the clip draw.
    print()
    out = []
    for run in sorted({r for r, _ in grouped}):
        settings = [s for r, s in grouped if r == run]
        base = grouped.get((run, "baseline"), [])
        control = {r["sample_id"]: r["physics_iq_score"] for r in base}

        ordered = ["baseline"] if base else []
        ordered += sorted(
            (s for s in settings if s != "baseline"),
            key=lambda s: -stats_module.mean(x["physics_iq_score"] for x in grouped[(run, s)]),
        )

        clips = min(len(grouped[(run, s)]) for s in ordered)
        width = max(len(s) for s in ordered) + 2
        print(f"{run}   ({clips} clips per setting)")
        print(f"  {'setting':<{width}}{'PhysicsIQ %':>13}{'vs base':>10}{'motion':>9}"
              f"{'prior RMS':>11}")
        print("  " + "-" * (width + 43))
        reference = stats_module.mean(r["reference_motion"] for r in scored if r["run"] == run)
        for setting in ordered:
            group = grouped[(run, setting)]
            score = stats_module.mean(r["physics_iq_score"] for r in group)
            motion = stats_module.mean(r["motion"] for r in group)
            delta = ""
            if control and setting != "baseline":
                paired = [r["physics_iq_score"] - control[r["sample_id"]]
                          for r in group if r["sample_id"] in control]
                if paired:
                    delta = f"{stats_module.mean(paired):+.2f}"
            rms = [r["prior_rms"] for r in group
                   if isinstance(r.get("prior_rms"), float) and r["prior_rms"] == r["prior_rms"]]
            line = (f"  {setting:<{width}}{score:>13.2f}{delta:>10}{motion:>9.4f}"
                    f"{(stats_module.mean(rms) if rms else float('nan')):>11.5f}")
            if motion < 0.5 * reference:
                line += "   MOTION COLLAPSE"
            print(line)
            out.append({
                "run": run, "setting": setting, "n_clips": len(group),
                "physics_iq_score": round(score, 3),
                "vs_baseline": delta or "",
                "raw_score": round(stats_module.mean(r["raw_score"] for r in group), 5),
                "spatial_iou": round(stats_module.mean(r["spatial_iou"] for r in group), 5),
                "spatiotemporal_iou": round(
                    stats_module.mean(r["spatiotemporal_iou"] for r in group), 5),
                "weighted_spatial_iou": round(
                    stats_module.mean(r["weighted_spatial_iou"] for r in group), 5),
                "mse": round(stats_module.mean(r["mse"] for r in group), 6),
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
    print("\nPhysicsIQ % is the raw score as a percentage of the real-vs-real ceiling:\n"
          "0 = no overlap with reality, 100 = matched a second real take of the scene.")


if __name__ == "__main__":
    main()
