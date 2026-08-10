#!/usr/bin/env python
"""Stage 6: baseline I2V continuation of LikePhys valid clips.

    python scripts/run_generation.py --config configs/experiments/generation_likephys.yaml
    python scripts/run_generation.py --config ... data__limit=10 generation__num_candidates=4

Conditions on the first frame of each *valid* clip, so the real continuation is a
physically plausible reference and every generation can be scored against ground truth.
Sampling is plain baseline throughout -- PhaseLock guidance is never applied.

With ``generation.num_candidates > 1`` this also runs best-of-N: candidates are ranked by
a verifier score and compared against the no-verifier baseline and the oracle.
"""

from __future__ import annotations

import argparse
import csv
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
from phaselock.datasets import LikePhys
from phaselock.experiments.generation import best_of_n, generate_candidate
from phaselock.experiments.detection import statistics_from_record
from phaselock.experiments.generation import score_against_fidelity
from phaselock.progress import track
from phaselock.metrics.geophys import geophys_statistics
from phaselock.probes import StatisticRow
from phaselock.probes import LATENT
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_generation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--verifier-source", default=LATENT, help="probe source used to rank candidates")
    parser.add_argument("--verifier-statistic", default="curv")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def verifier_score(candidate, source: str, statistic: str, config) -> float:
    """Mean of one GeoPhys statistic over the recorded steps. Lower is more plausible."""
    record = candidate.record
    if record is None or source not in record.sources:
        return 0.0
    blocks = record.blocks(source)
    values = [
        float(
            geophys_statistics(
                record.get(source, step, block),
                order=config.metrics.ar_order,
                fit=config.metrics.residual_fit,
            )[statistic]
        )
        for step in record.steps
        for block in blocks
    ]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    config = load(args.config, **parse_overrides(args.overrides))
    set_seed(config.generation.seed)

    output = config.output.dir()
    config.save(output / "config.json")
    video_dir = config.output.dir("videos") if args.save_videos else None
    # Needed for the per-frame timeline figures: statistics.csv keeps only each clip's
    # temporal summary, and a mean cannot be inverted into the values behind it.
    trajectory_dir = config.output.dir("trajectories") if config.probe.save_trajectories else None

    dataset = LikePhys(**({"root": config.data.root} if config.data.root else {}))
    clips = dataset.valid_clips()
    if config.data.scenarios:
        clips = [c for c in clips if c.meta["scenario"] in set(config.data.scenarios)]
    if config.data.limit:
        by_scenario: dict[str, list] = {}
        for clip in clips:
            by_scenario.setdefault(clip.meta["scenario"], []).append(clip)
        picked, order = [], sorted(by_scenario)
        while len(picked) < config.data.limit and any(by_scenario.values()):
            for scenario in order:
                if by_scenario[scenario]:
                    picked.append(by_scenario[scenario].pop(0))
                    if len(picked) == config.data.limit:
                        break
        clips = picked

    n = config.generation.num_candidates
    logger.info("%d clips x %d candidates = %d generations", len(clips), n, len(clips) * n)

    backend = load_backend(
        config.backend.name,
        model_id=config.backend.model_id,
        torch_dtype=resolve_dtype(config.backend.dtype),
        enable_offload=config.backend.offload,
    )

    rows, selection, statistics = [], [], []
    for index, clip in enumerate(track(clips, 'generating'), start=1):
        candidates = [
            generate_candidate(
                backend, clip, dataset, config,
                seed=config.generation.seed + offset,
                video_dir=video_dir,
            )
            for offset in range(n)
        ]
        rows.extend(candidate.flatten() for candidate in candidates)
        # The same per-(source, block, step) table the inversion stage writes. Generation
        # was recording every probe and keeping one verifier scalar, throwing thousands of
        # signals away; they are the whole point of probing during generation at all.
        for candidate in candidates:
            if candidate.record is None:
                continue
            if trajectory_dir is not None:
                # Named by (clip, seed) rather than clip alone: best-of-N writes several
                # candidates per clip and they must not overwrite each other.
                candidate.record.save(
                    trajectory_dir
                    / f"{clip.sample_id.replace('/', '_')}_s{candidate.seed}"
                )
            statistics.extend(
                row.flatten()
                for row in statistics_from_record(
                    candidate.record, clip,
                    order=config.metrics.ar_order, fit=config.metrics.residual_fit,
                )
            )

        if n > 1:
            scores = [
                verifier_score(c, args.verifier_source, args.verifier_statistic, config)
                for c in candidates
            ]
            picked, baseline, oracle = best_of_n(candidates, scores)
            selection.append(
                {
                    "sample_id": clip.sample_id,
                    "scenario": clip.meta["scenario"],
                    "selected": picked.scores.raw_score,
                    "baseline": baseline.scores.raw_score,
                    "oracle": oracle.scores.raw_score,
                }
            )
        logger.info("[%d/%d] %s", index, len(clips), clip.sample_id)

    with open(output / "candidates.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", output / "candidates.csv")

    if statistics:
        with open(output / "statistics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=StatisticRow.fieldnames())
            writer.writeheader()
            writer.writerows(statistics)
        logger.info("wrote %s (%d rows)", output / "statistics.csv", len(statistics))

        # No matched pair exists here, so pairwise accuracy does not apply. Ground truth
        # does: rank every signal by how well it predicts fidelity to the real
        # continuation, which is the question a verifier actually has to answer.
        ranked = score_against_fidelity(statistics, rows)
        if ranked:
            with open(output / "signal_fidelity.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ranked[0].flatten()))
                writer.writeheader()
                writer.writerows(item.flatten() for item in ranked)
            logger.info("wrote %s (%d signals)", output / "signal_fidelity.csv", len(ranked))
            print("\nSignals most predictive of fidelity (negative = larger statistic "
                  "means worse match, the expected direction)")
            print("-" * 78)
            for item in ranked[:10]:
                block = "" if item.block < 0 else f"/b{item.block}"
                label = f"{item.source}{block}/s{item.step}/{item.kind}_{item.statistic}"
                print(f"  {label:<48} rho {item.rho:+.3f}  (n={item.n})")

    if selection:
        report_selection(selection, output, args)


def report_selection(selection, output: Path, args) -> None:
    """Best-of-N gain against the no-verifier baseline and the oracle ceiling."""
    mean = lambda key: sum(row[key] for row in selection) / len(selection)
    baseline, selected, oracle = mean("baseline"), mean("selected"), mean("oracle")
    headroom = oracle - baseline

    print(f"\nBest-of-N with verifier {args.verifier_source}/{args.verifier_statistic}")
    print("-" * 62)
    print(f"  no verifier (first draw)  {baseline:.4f}")
    print(f"  selected                  {selected:.4f}   ({selected - baseline:+.4f})")
    print(f"  oracle (upper bound)      {oracle:.4f}   ({headroom:+.4f} headroom)")
    if headroom > 1e-9:
        print(f"  closed {100 * (selected - baseline) / headroom:.1f}% of the gap to the oracle")
    else:
        print("  candidates are indistinguishable; selection cannot help here")

    with open(output / "selection.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection[0]))
        writer.writeheader()
        writer.writerows(selection)


if __name__ == "__main__":
    main()
