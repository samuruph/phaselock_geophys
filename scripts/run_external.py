#!/usr/bin/env python
"""The published GeoPhys path: five statistics on frozen DINOv2 features.

    python scripts/run_external.py --config configs/experiments/detection_likephys.yaml

This is the correctness gate for everything else. GeoPhys reports 78-81% pairwise accuracy
on LikePhys from a single frozen backbone. If this run lands far from that, the statistics,
the pairing, the preprocessing or the scoring rule is wrong, and no number measured on
internal representations means anything.

It also performs the paper's readout-layer selection, choosing the layer that maximises
the violated-minus-plausible curvature gap.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phaselock import load, set_seed
from phaselock.config import parse_overrides
from phaselock.datasets import get_paired_dataset
from phaselock.encoders import DINOv2Encoder
from phaselock.experiments.detection import unique_samples
from phaselock.experiments.external import encode_sample, score_external
from phaselock.metrics.geophys import STATISTICS

logger = logging.getLogger("run_external")

# GeoPhys's reported single-backbone range on LikePhys.
PUBLISHED_RANGE = (0.776, 0.808)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="facebook/dinov2-large")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="restrict the readout sweep")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    config = load(args.config, **parse_overrides(args.overrides))
    set_seed(config.data.seed)

    output = config.output.dir("external")
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
    samples = unique_samples(pairs)
    logger.info("%s: %d pairs, %d distinct clips", dataset.name, len(pairs), len(samples))

    encoder = DINOv2Encoder(model_id=args.model)
    logger.info("%s: %d layers, %d prefix tokens", args.model, encoder.num_layers, encoder.num_prefix_tokens)

    rows = []
    for index, (sample, pair) in enumerate(samples, start=1):
        rows.extend(encode_sample(encoder, sample, pair, config, layers=args.layers))
        if index % 20 == 0 or index == len(samples):
            logger.info("[%d/%d] encoded", index, len(samples))

    with open(output / "external_statistics.csv", "w", newline="") as handle:
        flattened = [row.flatten() for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

    results = score_external(rows, pairs, resamples=config.metrics.bootstrap_resamples)
    report(results, output)


def report(results, output: Path) -> None:
    by_layer: dict[int, dict[str, object]] = defaultdict(dict)
    for (layer, statistic), result in results.items():
        by_layer[layer][statistic] = result

    print(f"\n{'layer':>6} " + " ".join(f"{name:>14}" for name in STATISTICS))
    print("-" * (7 + 15 * len(STATISTICS)))
    for layer in sorted(by_layer):
        cells = " ".join(f"{100 * by_layer[layer][n].accuracy:>13.1f}%" for n in STATISTICS)
        print(f"{layer:>6} {cells}")

    best_key, best = max(results.items(), key=lambda item: item[1].accuracy)
    print(f"\nbest readout: layer {best_key[0]} on '{best_key[1]}' -> {best}")

    low, high = PUBLISHED_RANGE
    if low - 0.08 <= best.accuracy <= high + 0.08:
        verdict = "consistent with the published 77.6-80.8% single-backbone range"
    else:
        verdict = (
            f"OUTSIDE the published {100 * low:.1f}-{100 * high:.1f}% range -- "
            "treat internal-representation results as unvalidated until this is resolved"
        )
    print(f"correctness gate: {verdict}")

    with open(output / "external_signals.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "statistic", "accuracy", "ci_low", "ci_high", "auc", "n_pairs"])
        for (layer, statistic), result in sorted(results.items()):
            writer.writerow(
                [layer, statistic, f"{result.accuracy:.6f}", f"{result.ci_low:.6f}",
                 f"{result.ci_high:.6f}", "" if result.auc is None else f"{result.auc:.6f}",
                 result.n_pairs]
            )
    print(f"wrote {output / 'external_signals.csv'}")


if __name__ == "__main__":
    main()
