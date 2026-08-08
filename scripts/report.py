#!/usr/bin/env python
"""Turn a detection run's CSVs into readable tables and block x step heatmaps.

    python scripts/report.py /data/experiments/phaselock_geophys/wan21_t2v_1_3b/likephys/inversion
    python scripts/report.py <run_dir> --statistic curv --source hidden_states

The headline output is the per-source comparison: which internal representation carries
the geometry. The heatmaps are the geometric analogue of the Invisible Hand's Figure 4,
which found linear-probe accuracy peaking in the middle third of the network.

Reads ``signals.csv`` (written by ``run_inversion.py``) and, if present,
``external/external_signals.csv`` for the DINOv2 correctness gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phaselock.metrics.geophys import STATISTICS

# GeoPhys's reported single-backbone range on LikePhys.
PUBLISHED_RANGE = (0.776, 0.808)

# Coarse ramp; the eye reads density here better than it reads numbers.
SHADES = " .:-=+*#%@"


def load(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="output directory of a detection run")
    parser.add_argument("--statistic", default=None, choices=list(STATISTICS), help="restrict heatmaps")
    parser.add_argument("--source", default=None, help="restrict heatmaps to one source")
    parser.add_argument("--kind", default="phi", choices=["phi", "drift"])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--figures", action="store_true", help="also write PNG heatmaps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load(args.run_dir / "signals.csv")
    if not rows:
        raise SystemExit(f"no signals.csv under {args.run_dir}; run scripts/run_inversion.py first")

    for row in rows:
        row["accuracy"] = float(row["accuracy"])
        row["ci_low"] = float(row["ci_low"])
        row["ci_high"] = float(row["ci_high"])
        row["block"] = int(row["block"])
        row["step"] = int(row["step"])
        row["n_pairs"] = int(row["n_pairs"])

    print(f"\n{len(rows)} signals from {rows[0]['n_pairs']} pairs\n")
    correctness_gate(args.run_dir)
    per_source(rows)
    ranked(rows, args.top)
    heatmaps(rows, args.statistic, args.source, args.kind)
    if args.figures:
        write_figures(rows, args.run_dir, args.kind)


def correctness_gate(run_dir: Path) -> None:
    """The DINOv2 baseline decides whether anything else is interpretable."""
    external = load(run_dir / "external" / "external_signals.csv")
    print("=" * 78)
    if not external:
        print("CORRECTNESS GATE: not run.")
        print("  Run scripts/run_external.py before trusting any internal number -- if the")
        print("  five statistics do not reproduce GeoPhys's published 77.6-80.8% on frozen")
        print("  DINOv2 features, the implementation is wrong and nothing below means anything.")
        print("=" * 78 + "\n")
        return

    best = max(external, key=lambda row: float(row["accuracy"]))
    accuracy = float(best["accuracy"])
    low, high = PUBLISHED_RANGE
    verdict = (
        "consistent with the published range"
        if low - 0.08 <= accuracy <= high + 0.08
        else "OUTSIDE the published range -- internal results are unvalidated"
    )
    print(f"CORRECTNESS GATE (DINOv2, layer {best['layer']}, '{best['statistic']}'): "
          f"{100 * accuracy:.1f}%")
    print(f"  published 77.6-80.8% single-backbone -> {verdict}")
    print("=" * 78 + "\n")


def per_source(rows: list[dict]) -> None:
    """Best signal per source: the comparison the project exists to make."""
    best: dict[str, dict] = {}
    for row in rows:
        current = best.get(row["source"])
        if current is None or row["accuracy"] > current["accuracy"]:
            best[row["source"]] = row

    print("Best signal per source")
    print("-" * 78)
    print(f"{'source':<16}{'block':>6}{'step':>6}{'statistic':>12}{'kind':>7}{'accuracy':>22}")
    for source, row in sorted(best.items(), key=lambda item: item[1]["accuracy"], reverse=True):
        block = "-" if row["block"] < 0 else str(row["block"])
        interval = f"[{100 * row['ci_low']:.1f}, {100 * row['ci_high']:.1f}]"
        print(
            f"{source:<16}{block:>6}{row['step']:>6}{row['statistic']:>12}{row['kind']:>7}"
            f"{100 * row['accuracy']:>11.1f}% {interval:>10}"
        )

    latent = best.get("latent")
    hidden = best.get("hidden_states")
    if latent and hidden:
        print()
        print("  The control comparison. 'The Invisible Hand of Physics' reports linear probes")
        print("  at chance (48-53%) on VAE latents. Geometry also landing near chance there")
        print("  while working on hidden states would say PhaseLock's latent delta operates in")
        print("  a representation that carries no physical signal.")
        print(f"    latent        {100 * latent['accuracy']:.1f}%")
        print(f"    hidden_states {100 * hidden['accuracy']:.1f}%")
    print()


def ranked(rows: list[dict], top: int) -> None:
    print(f"Top {top} signals")
    print("-" * 78)
    for row in sorted(rows, key=lambda r: r["accuracy"], reverse=True)[:top]:
        block = "" if row["block"] < 0 else f"/b{row['block']}"
        label = f"{row['source']}{block}/s{row['step']}/{row['kind']}_{row['statistic']}"
        auc = f"{float(row['auc']):.3f}" if row.get("auc") else "  -  "
        print(
            f"  {label:<48}{100 * row['accuracy']:>6.1f}% "
            f"[{100 * row['ci_low']:>5.1f},{100 * row['ci_high']:>5.1f}]  AUC {auc}"
        )
    print()


def heatmaps(rows: list[dict], statistic: str | None, source: str | None, kind: str) -> None:
    """Block x step accuracy grids, the geometric analogue of Invisible Hand Fig. 4."""
    grids: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for row in rows:
        if row["kind"] != kind:
            continue
        if statistic and row["statistic"] != statistic:
            continue
        if source and row["source"] != source:
            continue
        grids[(row["source"], row["statistic"])][(row["block"], row["step"])] = row["accuracy"]

    for (source_name, statistic_name), grid in sorted(grids.items()):
        blocks = sorted({block for block, _ in grid})
        steps = sorted({step for _, step in grid})
        if len(blocks) < 2 and len(steps) < 2:
            continue

        values = list(grid.values())
        low, high = min(values), max(values)
        span = max(high - low, 1e-9)

        print(f"{source_name} / {kind}_{statistic_name}   "
              f"(accuracy {100 * low:.1f}% {SHADES[0]!r} .. {100 * high:.1f}% {SHADES[-1]!r})")
        print(f"  {'block':>6} | steps {steps[0]} -> {steps[-1]}")
        for block in blocks:
            cells = "".join(
                SHADES[min(int((grid.get((block, step), low) - low) / span * (len(SHADES) - 1)), len(SHADES) - 1)]
                for step in steps
            )
            best_here = max((grid.get((block, step), low) for step in steps))
            label = "-" if block < 0 else str(block)
            print(f"  {label:>6} | {cells}  max {100 * best_here:.1f}%")
        print()


def write_figures(rows: list[dict], run_dir: Path, kind: str) -> None:
    """Render the figure set. See phaselock/analysis/figures.py for what each asks."""
    import csv as _csv

    from phaselock.analysis import render_all
    from phaselock.datasets import get_paired_dataset
    from phaselock.experiments.detection import delta_matrix, score_signals, summarise_grid
    from phaselock.metrics.scoring import selection_null

    config = json.loads((run_dir / "config.json").read_text())
    dataset = get_paired_dataset(config["data"]["name"])
    pairs = dataset.select(limit=config["data"]["limit"], seed=config["data"]["seed"])

    statistics = load(run_dir / "statistics.csv")
    summaries, null, steps_to_timestep = None, None, None
    if statistics:
        results = score_signals(statistics, pairs, resamples=200)
        summaries = summarise_grid(results)
        deltas = delta_matrix(statistics, pairs)
        if deltas.size:
            null = selection_null(deltas, resamples=300)
        # Recorded step index -> the actual diffusion timestep, so axes are readable.
        steps_to_timestep = {
            int(row["step"]): round((1.0 - float(row["tau"])) * 1000) for row in statistics
        }

    # A pooled external run, if present, is the one that is actually comparable to the
    # internal path; fall back to the unpooled one and say so in the figures.
    external_dir, external_pooling = run_dir / "external_latent", "latent"
    if not (external_dir / "external_signals.csv").is_file():
        external_dir, external_pooling = run_dir / "external", "none"

    external_summaries = None
    external_rows = load(external_dir / "external_signals.csv")
    if external_rows:
        from phaselock.experiments.detection import SourceSummary

        grouped: dict[str, list[float]] = {}
        for row in external_rows:
            grouped.setdefault(row["statistic"], []).append(float(row["accuracy"]))
        import statistics as _stats

        external_summaries = [
            SourceSummary(
                source="dinov2", statistic=name, kind="phi",
                mean=sum(values) / len(values),
                std=_stats.pstdev(values) if len(values) > 1 else 0.0,
                n_cells=len(values), best=max(values), best_label=f"dinov2/{name}",
            )
            for name, values in sorted(grouped.items())
        ]

    # Map the external per-clip table onto the internal schema (block = readout layer,
    # one step) so DINOv2 gets the same per-source figures as everything else.
    external_statistics = [
        dict(row, source="dinov2", block=row.get("encoder_layer", 0), step=0)
        for row in load(external_dir / "external_statistics.csv")
    ]
    external_signal_rows = [
        {"source": "dinov2", "block": row["layer"], "step": 0, "statistic": row["statistic"],
         "kind": "phi", "accuracy": row["accuracy"], "ci_low": row["ci_low"],
         "ci_high": row["ci_high"], "auc": row.get("auc", ""), "n_pairs": row["n_pairs"]}
        for row in external_rows
    ]

    from phaselock.backends import get_spec

    sweep = load(run_dir / "sweep.csv") or None
    written = render_all(
        rows, run_dir / "figures", summaries=summaries,
        external_summaries=external_summaries, null=null,
        steps_to_timestep=steps_to_timestep, sweep=sweep,
        statistics_rows=statistics + external_statistics,
        external_rows=external_signal_rows,
        external_pooling=external_pooling,
        temporal_ratio=get_spec(config["backend"]["name"]).temporal_ratio,
    )
    print(f"\nwrote {len(written)} figures to {run_dir / 'figures'}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
