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
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phaselock import load, set_seed
from phaselock.backends import load_backend
from phaselock.config import parse_overrides
from phaselock.datasets import get_paired_dataset
from phaselock.experiments.detection import (
    extract_sample,
    score_signals,
    summarise_by_source,
    unique_samples,
    write_rows,
)
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML experiment config")
    parser.add_argument("--score-only", action="store_true", help="skip extraction, score the existing CSV")
    parser.add_argument("--top", type=int, default=25, help="how many signals to print")
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

    report(results, args.top, output)


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

    for index, (sample, pair) in enumerate(todo, start=1):
        rows = extract_sample(backend, sample, pair, config, trajectory_dir)
        written = write_rows(rows, statistics_path)
        logger.info("[%d/%d] %s -> %d rows", index, len(todo), sample.sample_id, written)


def report(results, top: int, output: Path) -> None:
    """Print the ranked signals and the per-source comparison, and save both."""
    ranked = sorted(results.items(), key=lambda item: item[1].accuracy, reverse=True)

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

    with open(output / "signals.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "block", "step", "statistic", "kind", "accuracy", "ci_low", "ci_high", "auc", "n_pairs"])
        for key, result in ranked:
            writer.writerow(
                [key.source, key.block, key.step, key.statistic, key.kind,
                 f"{result.accuracy:.6f}", f"{result.ci_low:.6f}", f"{result.ci_high:.6f}",
                 "" if result.auc is None else f"{result.auc:.6f}", result.n_pairs]
            )
    print(f"\nwrote {output / 'signals.csv'}")


if __name__ == "__main__":
    main()
