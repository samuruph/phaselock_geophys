#!/usr/bin/env python
"""What lives inside a single latent, and is it a usable prior?

    python scripts/run_latent_motion.py --config configs/experiments/inversion_likephys_wan.yaml \
        data__limit=24

A causal VAE bundles four video frames into one latent, so motion *inside* a latent is
invisible to a frame difference along the latent axis -- which is exactly what PhaseLock's
latent delta and GeoPhys's first-order velocity both are. This asks two things:

  1. Which cheap statistic of the latent predicts the motion hidden inside it?
     (Correlated against ground truth obtained by decoding.)
  2. Do plausible and violated clips differ in those statistics, along the latent axis
     and along the denoising trajectory?

A statistic that scores well on (1) and separates on (2) is a prior that frame
differencing cannot see.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phaselock.runtime import prepare

prepare()

import torch

from phaselock import load, set_seed
from phaselock.analysis import figures, palette
from phaselock.analysis.latent_motion import (
    build_profiles,
    correlation,
    hidden_motion,
    inter_latent_motion,
    latent_statistics,
)
from phaselock.backends import load_backend
from phaselock.config import parse_overrides
from phaselock.datasets import get_paired_dataset, load_video
from phaselock.pipelines.inversion import invert
from phaselock.probes import LATENT
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_latent_motion")

STATISTICS = ("channel_std", "spatial_std", "spatial_gradient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--trajectory", action="store_true",
                        help="also invert each clip to profile the statistics along tau (slow)")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    config = load(args.config, **parse_overrides(args.overrides))
    set_seed(config.data.seed)

    output = config.output.dir("latent_motion")
    dataset_kwargs = {}
    if config.data.root:
        dataset_kwargs["root"] = config.data.root
    if config.data.split:
        dataset_kwargs["split"] = config.data.split
    dataset = get_paired_dataset(config.data.name, **dataset_kwargs)
    pairs = dataset.select(limit=config.data.limit, seed=config.data.seed)

    backend = load_backend(
        config.backend.name,
        model_id=config.backend.model_id,
        torch_dtype=resolve_dtype(config.backend.dtype),
        enable_offload=config.backend.offload,
    )
    spec = backend.spec
    logger.info("%d pairs, %s, %d frames -> %d latent frames",
                len(pairs), spec.name, spec.default_num_frames,
                (spec.default_num_frames - 1) // spec.temporal_ratio + 1)

    rows: list[dict] = []
    series = {"plausible": [], "violated": []}
    taus = {"plausible": [], "violated": []}

    for index, pair in enumerate(pairs, start=1):
        for label, sample in (("plausible", pair.plausible), ("violated", pair.violated)):
            frames = load_video(sample.path, num_frames=spec.default_num_frames,
                                height=spec.default_height, width=spec.default_width)
            latents = backend.encode(frames).cpu()

            stats = latent_statistics(latents)
            # Ground truth: motion the latent axis cannot see, obtained by decoding.
            inside = hidden_motion(backend.decode(latents).cpu(), spec)
            between = inter_latent_motion(latents)

            row = {"sample_id": sample.sample_id, "label": label, "scenario": pair.scenario}
            for name, value in stats.as_dict().items():
                row[f"corr_{name}_vs_hidden"] = correlation(value[1:], inside[1:])
                row[f"mean_{name}"] = float(value.mean())
            # The control: does the frame difference itself predict the hidden motion?
            row["corr_interlatent_vs_hidden"] = correlation(between, inside[1:])
            row["mean_hidden_motion"] = float(inside[1:].mean())
            rows.append(row)
            series[label].append(stats.channel_std)

            if args.trajectory:
                result = invert(backend, latents=latents, num_steps=config.inversion.num_steps,
                                record_steps=config.probe.record_steps, sources=[LATENT],
                                prompt=config.inversion.prompt)
                steps = result.record.steps
                taus[label].append((
                    [result.record.taus[s] for s in steps],
                    [float(result.record.get(LATENT, s).std()) for s in steps],
                ))
        logger.info("[%d/%d] %s", index, len(pairs), pair.scenario)

    with open(output / "latent_motion.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    palette.apply_style()
    figures.latent_motion_profile(
        build_profiles(series["plausible"], series["violated"],
                       taus["plausible"] or None, taus["violated"] or None),
        output / "07_latent_motion.png",
    )
    report(rows, output)


def report(rows: list[dict], output: Path) -> None:
    mean = lambda key, rs: sum(r[key] for r in rs) / len(rs)

    print("\nCan a latent statistic predict the motion hidden inside it?")
    print("=" * 72)
    print("  Correlation against decoded intra-latent pixel motion, averaged over clips.")
    print(f"\n  {'statistic':<26}{'corr':>8}")
    print("  " + "-" * 34)
    ranked = sorted(
        [(f"{name}", mean(f"corr_{name}_vs_hidden", rows)) for name in STATISTICS]
        + [("inter-latent delta (control)", mean("corr_interlatent_vs_hidden", rows))],
        key=lambda item: -abs(item[1]),
    )
    for name, value in ranked:
        print(f"  {name:<26}{value:>8.3f}")
    print("\n  The control is PhaseLock's latent delta. A statistic that beats it is seeing")
    print("  motion the delta cannot.")

    print("\nDo plausible and violated clips differ in these statistics?")
    print("=" * 72)
    good = [r for r in rows if r["label"] == "plausible"]
    bad = [r for r in rows if r["label"] == "violated"]
    print(f"  {'statistic':<26}{'plausible':>11}{'violated':>11}{'rel diff':>10}")
    print("  " + "-" * 58)
    for name in list(STATISTICS) + ["hidden_motion"]:
        key = f"mean_{name}"
        a, b = mean(key, good), mean(key, bad)
        rel = (b - a) / abs(a) if abs(a) > 1e-12 else float("nan")
        print(f"  {name:<26}{a:>11.4f}{b:>11.4f}{rel:>+9.1%}")
    print(f"\nwrote {output / 'latent_motion.csv'} and 07_latent_motion.png")


if __name__ == "__main__":
    main()
