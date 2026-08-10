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
    # Both by default. Family survives a small draw where scenario does not, but at n=100
    # every scenario has 8-9 pairs and the finer table is the more useful one -- so which
    # is worth reading depends on the run, and both are cheap once the signals are scored.
    parser.add_argument("--category", default=["family", "scenario"], nargs="+",
                        choices=["family", "scenario", "violation"],
                        help="grouping(s) for the per-category breakdown")
    # A run written before a statistic existed can be rebuilt from its trajectories by
    # scripts/rescore_trajectories.py. Pointing at that output beats copying it over the
    # original, which loses the numbers the run actually reported.
    parser.add_argument("--statistics", default="statistics.csv",
                        help="statistics file to read, relative to run_dir "
                             "(e.g. statistics_rescored.csv)")
    parser.add_argument("--signals", default="signals.csv",
                        help="scored-signal file to read, relative to run_dir. Pair it "
                             "with --statistics; rescore_trajectories.py writes both.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Dispatch on what the run actually produced, so one command serves every stage.
    # Requiring signals.csv meant the generation and step-sweep runs wrote their CSVs and
    # then silently had no figures, which is how they sat unreported for a day.
    if not (args.run_dir / args.signals).is_file():
        for filename, handler in (
            ("sweep.csv", step_sweep_report),
            ("candidates.csv", generation_report),
        ):
            if (args.run_dir / filename).is_file():
                handler(args.run_dir, args.figures)
                return
        raise SystemExit(
            f"nothing to report under {args.run_dir}: expected signals.csv (inversion), "
            "sweep.csv (step sweep) or candidates.csv (generation)"
        )

    rows = load(args.run_dir / args.signals)
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
        write_figures(rows, args.run_dir, args.kind, args.statistics)
        for key in args.category:
            write_category_table(args.run_dir, key, args.statistics)


def write_category_table(run_dir: Path, key: str,
                         statistics_name: str = "statistics.csv") -> None:
    """Where each signal succeeds and fails, as a figure and a spreadsheet.

    The aggregate hides this: at n=96 the best cell scored 100% on four scenarios and
    50% -- chance -- on `river`. A method that works on half the physics and one that
    works everywhere have the same mean.
    """
    import csv as _csv
    import json as _json

    from phaselock.analysis import category_table
    from phaselock.datasets import get_paired_dataset
    from phaselock.experiments.detection import score_by_category

    statistics = load(run_dir / statistics_name)
    if not statistics:
        return
    config = _json.loads((run_dir / "config.json").read_text())
    pairs = get_paired_dataset(config["data"]["name"]).select(
        limit=config["data"]["limit"], seed=config["data"]["seed"])

    cells = score_by_category(statistics, pairs, key=key)

    # The external baseline belongs in the same table: "which representation is best per
    # family" is not answerable without the thing it is being compared against. DINOv2's
    # rows are mapped onto the same schema, layer standing in for block.
    directory, _ = external_dir(run_dir)
    external = load(directory / "external_statistics.csv")
    if external:
        cells += score_by_category(
            [dict(row, source="dinov2", block=row.get("encoder_layer", "0"), step="0")
             for row in external],
            pairs, key=key,
        )

    if not cells:
        print(f"  (no category breakdown: fewer than 2 pairs in every {key})")
        return

    from phaselock.analysis.spreadsheet import write_category_workbook

    book = write_category_workbook(cells, run_dir / f"category_{key}.xlsx", key=key)
    print(f"  {book.name}")

    produced = category_table(
        cells, run_dir / "figures" / f"06_by_{key}.png",
        title=f"Violation detection by {key}",
    )
    if produced:
        print(f"  {produced.name}")


def step_sweep_report(run_dir: Path, figures: bool) -> None:
    """Stage 7. The blur control is the whole point, so it leads the table."""
    cells = load(run_dir / "sweep.csv")
    print(f"\n{len(cells)} cells from {len({c['sample_id'] for c in cells})} clips\n")

    from phaselock.analysis.figures import SWEEP_METRICS

    steps = sorted({int(c["num_steps"]) for c in cells})
    sigmas = sorted({float(c["blur_sigma"]) for c in cells})
    for column, label, higher in SWEEP_METRICS:
        values = [c for c in cells if c.get(column) not in (None, "")]
        if not values:
            continue
        print(f"{label}   ({'higher' if higher else 'lower'} is better)")
        print("       " + "".join(f"{'K=' + str(k):>12}" for k in steps))
        for sigma in sigmas:
            row = []
            for k in steps:
                got = [float(c[column]) for c in values
                       if int(c["num_steps"]) == k and float(c["blur_sigma"]) == sigma]
                row.append(sum(got) / len(got) if got else float("nan"))
            print(f"  s={sigma:<4g}" + "".join(f"{v:12.4f}" for v in row))
        print()

    if figures:
        from phaselock.analysis import step_sweep, step_sweep_panels

        directory = run_dir / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        for produced in (
            step_sweep_panels(cells, directory / "01_step_sweep_panels.png"),
            step_sweep(cells, directory / "02_step_sweep_curvature.png", metric="phi_curv"),
        ):
            if produced:
                print(f"  {produced.name}")


def generation_report(run_dir: Path, figures: bool) -> None:
    """Stage 6. Fidelity of the generated continuation to the real one."""
    import statistics as stats

    cells = load(run_dir / "candidates.csv")
    print(f"\n{len(cells)} generations from {len({c['sample_id'] for c in cells})} clips\n")
    for column in ("spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou", "mse", "raw_score"):
        values = [float(c[column]) for c in cells if c.get(column) not in (None, "")]
        if values:
            spread = stats.stdev(values) if len(values) > 1 else 0.0
            print(f"  {column:<24}{stats.mean(values):8.4f} +/- {spread:.4f}"
                  f"   [{min(values):.4f}, {max(values):.4f}]")

    ranked = load(run_dir / "signal_fidelity.csv")
    if ranked:
        print("\nSignals most predictive of fidelity  (negative rho = larger statistic "
              "means worse match, the expected direction)")
        print("-" * 78)
        for row in sorted(ranked, key=lambda r: float(r["spearman_rho"]))[:12]:
            block = "" if int(row["block"]) < 0 else f"/b{row['block']}"
            label = f"{row['source']}{block}/s{row['step']}/{row['kind']}_{row['statistic']}"
            print(f"  {label:<48} rho {float(row['spearman_rho']):+.3f}  (n={row['n_clips']})")
    else:
        print("\nno signal_fidelity.csv: a rank correlation needs at least 3 clips.")

    if figures:
        from phaselock.analysis import generation_quality, render_all

        directory = run_dir / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        produced = generation_quality(cells, directory / "00_generation_quality.png")
        if produced:
            print(f"\n  {produced.name}")

        statistics = load(run_dir / "statistics.csv")
        if statistics:
            from phaselock.analysis import signal_quality_correlation

            # The plainest statement of what generation measures: signal against
            # ground-truth quality, no rescaling, with the scatter beside it.
            produced = signal_quality_correlation(
                statistics, cells, directory / "02_signal_vs_quality.png")
            if produced:
                print(f"  {produced.name}")
        if ranked and statistics:
            for path in generation_figures(ranked, statistics, directory):
                print(f"  {path.relative_to(directory)}")


def generation_figures(ranked, statistics, directory):
    """The inversion figure set, on generation's own scoring.

    Generation has no matched pair, so its signals are ranked by Spearman correlation
    against ground-truth fidelity rather than by pairwise accuracy. Those live on
    different scales, and rather than build a parallel set of figures the rho is mapped
    onto the accuracy axis the existing ones already use:

        accuracy = (1 - rho) / 2

    which is the concordance probability -- the chance that a randomly chosen pair of
    generations is ordered correctly by this signal. rho = -1, a signal that perfectly
    predicts *worse* fidelity as it grows, becomes 100%; rho = 0 becomes 50%, chance, in
    the same place the pairwise figures put it. The sign flip is deliberate: every
    statistic is oriented so larger means less regular.
    """
    from phaselock.analysis import render_all
    from phaselock.experiments.detection import SignalKey, summarise_grid
    from phaselock.metrics.scoring import PairwiseResult

    results, rows = {}, []
    for row in ranked:
        rho = float(row["spearman_rho"])
        accuracy = (1.0 - rho) / 2.0
        key = SignalKey(row["source"], int(row["block"]), int(row["step"]),
                        row["statistic"], row["kind"])
        results[key] = PairwiseResult(accuracy, accuracy, accuracy, int(row["n_clips"]))
        rows.append({
            "source": key.source, "block": key.block, "step": key.step,
            "statistic": key.statistic, "kind": key.kind, "accuracy": accuracy,
            "ci_low": accuracy, "ci_high": accuracy, "auc": "",
            "n_pairs": int(row["n_clips"]),
        })

    steps = {int(r["step"]): round((1.0 - float(r["tau"])) * 1000) for r in statistics}
    return render_all(
        rows, directory, summaries=summarise_grid(results),
        steps_to_timestep=steps, statistics_rows=statistics,
        value_label="concordance with ground-truth fidelity (%)",
        title="Which signal predicts a faithful generation?",
        formula=(
            r"per probe cell:  $\rho$ = Spearman$\left(\varphi_\sigma^{\,(b,s)},"
            r"\ \mathrm{raw\_score}\right)$ over clips" "\n"
            r"concordance $=\frac{1-\rho}{2}\times 100$,   bar = mean over all cells"
            "\n"
            r"raw_score $=\Sigma\,$IoU$\,-\,$MSE  vs the real continuation" "\n"
            "50% = chance.  Correlation, not an accuracy: nothing is counted."
        ),
    )


def external_dir(run_dir: Path) -> tuple[Path, str]:
    """The external baseline to use, preferring the temporally matched one.

    A pooled run is the only one directly comparable to the internal path -- unpooled
    DINOv2 is per video frame, four times finer in time than the latents -- so it wins
    when both are present. One lookup, shared by the table and the figures: they read
    different directories for a while, and the table reported the gate as "not run" while
    the figures were drawing it.
    """
    pooled = run_dir / "external_latent"
    if (pooled / "external_signals.csv").is_file():
        return pooled, "latent"
    return run_dir / "external", "none"


def correctness_gate(run_dir: Path) -> None:
    """The DINOv2 baseline decides whether anything else is interpretable."""
    directory, pooling = external_dir(run_dir)
    external = load(directory / "external_signals.csv")
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
    matched = "temporally matched to the latent grid" if pooling == "latent" else (
        "per-frame, NOT temporally matched to the internal path")
    print(f"CORRECTNESS GATE (DINOv2, layer {best['layer']}, '{best['statistic']}'): "
          f"{100 * accuracy:.1f}%")
    print(f"  published 77.6-80.8% single-backbone -> {verdict}")
    print(f"  baseline is {matched}")
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


def write_figures(rows: list[dict], run_dir: Path, kind: str,
                  statistics_name: str = "statistics.csv") -> None:
    """Render the figure set. See phaselock/analysis/figures.py for what each asks."""
    import csv as _csv

    from phaselock.analysis import render_all
    from phaselock.datasets import get_paired_dataset
    from phaselock.experiments.detection import delta_matrix, score_signals, summarise_grid
    from phaselock.metrics.scoring import selection_null

    config = json.loads((run_dir / "config.json").read_text())
    dataset = get_paired_dataset(config["data"]["name"])
    pairs = dataset.select(limit=config["data"]["limit"], seed=config["data"]["seed"])

    statistics = load(run_dir / statistics_name)
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
    ext_dir, external_pooling = external_dir(run_dir)

    external_summaries = None
    external_rows = load(ext_dir / "external_signals.csv")
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
        for row in load(ext_dir / "external_statistics.csv")
    ]
    external_signal_rows = [
        {"source": "dinov2", "block": row["layer"], "step": 0, "statistic": row["statistic"],
         "kind": "phi", "accuracy": row["accuracy"], "ci_low": row["ci_low"],
         "ci_high": row["ci_high"], "auc": row.get("auc", ""), "n_pairs": row["n_pairs"]}
        for row in external_rows
    ]

    from phaselock.backends import get_spec

    # The aggregate table: sources down, signals across, one view of everything. The
    # per-category tables break this down; this is the thing they break down, and every
    # cell here rests on the full sample rather than a quarter of it.
    if summaries:
        from phaselock.analysis import summary_table
        from phaselock.analysis.spreadsheet import write_summary_workbook

        combined = list(summaries) + list(external_summaries or [])
        book = write_summary_workbook(combined, run_dir / "overall.xlsx")
        print(f"  {book.name}")
        produced = summary_table(combined, run_dir / "figures" / "00_overall.png")
        if produced:
            print(f"  {produced.name}")

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
