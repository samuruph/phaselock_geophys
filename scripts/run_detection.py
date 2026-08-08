#!/usr/bin/env python
"""Stage 5: score GeoPhys geometry on internal representations of labelled clips.

    python scripts/run_detection.py --config configs/experiments/detection_likephys.yaml
    python scripts/run_detection.py --config configs/experiments/detection_likephys.yaml \
        data__limit=24 inversion__num_steps=100

Extraction is resumable: clips already present in ``statistics.csv`` are skipped, so an
interrupted ten-hour run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must run before torch is imported: a system CUDA install ahead of torch's bundled
# libraries on LD_LIBRARY_PATH aborts the process inside the VAE encode.
from phaselock.runtime import prepare

prepare()

from phaselock import load, set_seed
from phaselock.backends import load_backend
from phaselock.config import parse_overrides
from phaselock.datasets import get_paired_dataset
from phaselock.experiments.detection import (
    delta_matrix,
    extract_sample,
    reconstruction_check,
    score_ensembles,
    score_signals,
    summarise_by_source,
    unique_samples,
    write_rows,
)
from phaselock.metrics.scoring import selection_null
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML experiment config")
    parser.add_argument("--score-only", action="store_true", help="skip extraction, score the existing CSV")
    parser.add_argument("--top", type=int, default=25, help="how many signals to print")
    parser.add_argument("--save-visuals", type=int, default=0, metavar="N",
                        help="write contact sheets for the first N pairs, to inspect by eye")
    parser.add_argument("overrides", nargs="*", help="section__key=value overrides")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    config = load(args.config, **parse_overrides(args.overrides))
    set_seed(config.data.seed)

    output = config.output.dir()
    config.save(output / "config.json")
    statistics_path = output / "statistics.csv"

    dataset_kwargs = {}
    if config.data.root:
        dataset_kwargs["root"] = config.data.root
    if config.data.split:
        dataset_kwargs["split"] = config.data.split
    dataset = get_paired_dataset(config.data.name, **dataset_kwargs)

    pairs = dataset.select(
        scenarios=config.data.scenarios,
        violations=config.data.violations,
        limit=config.data.limit,
        seed=config.data.seed,
    )
    logger.info("%s: %d pairs over %d scenarios", dataset.name, len(pairs), len({p.scenario for p in pairs}))

    if args.save_visuals:
        write_visuals(config, pairs[: args.save_visuals], output)

    if not args.score_only:
        extract(config, pairs, statistics_path, output)

    if not statistics_path.exists():
        logger.error("no statistics at %s", statistics_path)
        return

    with open(statistics_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    logger.info("scoring %d rows", len(rows))

    results = score_signals(rows, pairs, resamples=config.metrics.bootstrap_resamples)
    if not results:
        logger.error("no scorable signals; check that both members of each pair were extracted")
        return

    # OR and Majority across the five statistics. These carry GeoPhys's headline numbers
    # (98.3% on LikePhys vs 77.6-80.8% for the best single signal), so reporting only
    # single signals would understate the method by roughly 18 points.
    ensembles = score_ensembles(rows, pairs)
    logger.info("scored %d single signals and %d ensembles", len(results), len(ensembles))

    # Reporting the best of thousands of signals is a selection procedure, and its
    # output is biased upward. The permutation null says how high it would climb anyway.
    null = None
    deltas = delta_matrix(rows, pairs)
    if deltas.size:
        null = selection_null(deltas, resamples=300)
        logger.info("selection null: %s", null)

    report(results, ensembles, args.top, output, null)


def write_visuals(config, pairs, output: Path) -> None:
    """Contact sheets for eyeballing preprocessing and inversion before trusting numbers."""
    from phaselock.experiments.detection import save_visuals

    backend = load_backend(
        config.backend.name, model_id=config.backend.model_id,
        torch_dtype=resolve_dtype(config.backend.dtype), enable_offload=config.backend.offload,
    )
    directory = config.output.dir("visuals")
    for index, pair in enumerate(pairs, start=1):
        for path in save_visuals(backend, pair, config, directory):
            logger.info("[%d/%d] wrote %s", index, len(pairs), path.name)


def extract(config, pairs, statistics_path, output) -> None:
    """Invert every distinct clip and append its statistics."""
    done: set[str] = set()
    if statistics_path.exists():
        with open(statistics_path, newline="") as handle:
            done = {row["sample_id"] for row in csv.DictReader(handle)}
        logger.info("resuming: %d clips already extracted", len(done))

    todo = [(s, p) for s, p in unique_samples(pairs) if s.sample_id not in done]
    logger.info("%d clips to invert", len(todo))
    if not todo:
        return

    backend = load_backend(
        config.backend.name,
        model_id=config.backend.model_id,
        torch_dtype=resolve_dtype(config.backend.dtype),
        enable_offload=config.backend.offload,
    )
    trajectory_dir = config.output.dir("trajectories") if config.probe.save_trajectories else None

    if config.inversion.reconstruction_check:
        sample, _ = todo[0]
        scores = reconstruction_check(backend, sample, config)
        logger.info(
            "reconstruction check on %s: PSNR %.2f dB at %d steps "
            "(rules out a broken inversion; sweep num_steps for trajectory fidelity)",
            sample.sample_id, scores["psnr"], int(scores["num_steps"]),
        )
        (output / "reconstruction.json").write_text(json.dumps(scores, indent=2))

    for index, (sample, pair) in enumerate(todo, start=1):
        rows = extract_sample(backend, sample, pair, config, trajectory_dir)
        written = write_rows(rows, statistics_path)
        logger.info("[%d/%d] %s -> %d rows", index, len(todo), sample.sample_id, written)


def report(results, ensembles, top: int, output: Path, null=None) -> None:
    """Print the ranked signals, the per-source comparison and the ensembles."""
    ranked = sorted(results.items(), key=lambda item: item[1].accuracy, reverse=True)

    if null is not None:
        print("\n" + "=" * 88)
        print("SELECTION-CORRECTED SIGNIFICANCE")
        print(f"  {null}")
        if not null.significant:
            print("  The best signal is INSIDE the band the best-of-N reaches by chance.")
            print("  Treat the table below as exploratory: more pairs, not more signals.")
        else:
            print("  The best signal exceeds what selection alone produces.")
        print("=" * 88)

    print(f"\n{'signal':<52} {'pairwise acc':>26} {'AUC':>7}")
    print("-" * 88)
    for key, result in ranked[:top]:
        auc = f"{result.auc:.3f}" if result.auc is not None else "  -  "
        print(f"{key.label():<52} {str(result):>26} {auc:>7}")

    print("\nBest signal per source -- which internal representation carries the geometry:")
    print("-" * 88)
    for source, (key, result) in sorted(
        summarise_by_source(results).items(), key=lambda item: item[1][1].accuracy, reverse=True
    ):
        print(f"  {source:<16} {key.label():<44} {result}")

    if ensembles:
        print("\nEnsembles across the five statistics (GeoPhys's headline combination):")
        print("-" * 88)
        best_or = max(
            (item for item in ensembles.items() if item[0].statistic == "or"),
            key=lambda item: item[1].accuracy,
            default=None,
        )
        best_majority = max(
            (item for item in ensembles.items() if item[0].statistic == "majority"),
            key=lambda item: item[1].accuracy,
            default=None,
        )
        for label, best in (("OR", best_or), ("Majority", best_majority)):
            if best is not None:
                print(f"  {label:<10} {best[0].label():<44} {best[1]}")

    combined = {**results, **ensembles}
    with open(output / "signals.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "block", "step", "statistic", "kind", "accuracy", "ci_low", "ci_high", "auc", "n_pairs"])
        for key, result in sorted(combined.items(), key=lambda i: i[1].accuracy, reverse=True):
            writer.writerow(
                [key.source, key.block, key.step, key.statistic, key.kind,
                 f"{result.accuracy:.6f}", f"{result.ci_low:.6f}", f"{result.ci_high:.6f}",
                 "" if result.auc is None else f"{result.auc:.6f}", result.n_pairs]
            )
    print(f"\nwrote {output / 'signals.csv'}")


if __name__ == "__main__":
    main()
