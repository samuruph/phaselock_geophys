#!/usr/bin/env python
"""Stage 7: does trajectory geometry reproduce the few-step effect?

    python scripts/run_step_sweep.py --config configs/experiments/step_sweep_likephys.yaml

Generates each LikePhys valid clip at K in {2, 10, 30, 50} from the same seed, guidance
off in every arm, then measures three things at every (K, sigma) cell with Gaussian blur
applied to *both* the generation and the real reference:

  * DINOv2 GeoPhys -- the published external path, the primary result
  * PhaseLock's inter-frame phase-difference correlation -- their Fig. 3a metric, which
    doubles as a reproduction check (0.358 at K=2 vs 0.100 at K=50, under sigma=16)
  * motion-mask fidelity -- ground-truth based, so it arbitrates if the two disagree

The blur control is the point. A 2-step output is blurrier, so any feature trajectory
could look "more regular" purely from having less texture to move around. If the K=2
advantage vanishes at sigma=16, the ordering is a sharpness artefact and not a physics
result -- worth reporting either way, since PhaseLock's spectral metric does survive.
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

import torch

from phaselock import load, set_seed
from phaselock.backends import load_backend
from phaselock.config import parse_overrides
from phaselock.datasets import LikePhys
from phaselock.encoders import DINOv2Encoder
from phaselock.experiments.detection import frame_geometry
from phaselock.experiments.generation import frames_to_tensor, reference_continuation
from phaselock.experiments.step_sweep import (
    DEFAULT_BLUR,
    DEFAULT_STEPS,
    SweepCell,
    blur_survival,
    measure_cell,
)
from phaselock.progress import track
from phaselock.metrics.geophys import STATISTICS
from phaselock.pipelines.generation import generate_with_probes
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_step_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--encoder", default="facebook/dinov2-large")
    parser.add_argument("--layer", type=int, default=None, help="DINOv2 readout layer")
    parser.add_argument("overrides", nargs="*")
    # parse_known_args, not parse_args: argparse fills a positional nargs="*"
    # from the FIRST run of positionals it meets and rejects any later run, so an
    # override that lands after a flag kills the whole stage with "unrecognized
    # arguments". Leftovers are still validated by parse_overrides, which rejects
    # anything that is not section__key=value.
    args, extra = parser.parse_known_args()
    args.overrides = list(args.overrides) + extra
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    config = load(args.config, **parse_overrides(args.overrides))
    set_seed(config.generation.seed)

    steps = tuple(config.generation.step_sweep or DEFAULT_STEPS)
    blurs = tuple(config.generation.blur_sweep or DEFAULT_BLUR)
    output = config.output.dir()
    config.save(output / "config.json")

    dataset = LikePhys(**({"root": config.data.root} if config.data.root else {}))
    clips = dataset.valid_clips()
    if config.data.limit:
        clips = clips[:: max(1, len(clips) // config.data.limit)][: config.data.limit]

    logger.info(
        "%d clips x %d step counts = %d generations, each scored at %d blur levels",
        len(clips), len(steps), len(clips) * len(steps), len(blurs),
    )

    backend = load_backend(
        config.backend.name,
        model_id=config.backend.model_id,
        torch_dtype=resolve_dtype(config.backend.dtype),
        enable_offload=config.backend.offload,
    )
    encoder = DINOv2Encoder(model_id=args.encoder, layer=args.layer)

    cells: list[SweepCell] = []
    for index, clip in enumerate(track(clips, 'step sweep'), start=1):
        from PIL import Image

        reference = reference_continuation(clip, backend, config=config)
        first_frame = Image.fromarray((reference[0].permute(1, 2, 0) * 255).byte().cpu().numpy())
        scenario = clip.meta["scenario"]
        # Same geometry on both arms. The reference honours data.height/width; without
        # passing it on, the generator falls back to the backend default and measure_cell
        # rejects the pair on shape -- which is exactly how this stage died at n=100.
        _, height, width = frame_geometry(backend.spec, config)

        for num_steps in steps:
            result = generate_with_probes(
                backend,
                prompt=dataset.prompt_for(scenario),
                image=first_frame,
                num_steps=num_steps,
                height=height,
                width=width,
                record_steps=min(config.probe.record_steps, num_steps),
                sources=config.probe.sources,
                blocks=config.probe.blocks,
                block_stride=config.probe.block_stride,
                pooling=config.probe.pooling,
                guidance_scale=config.generation.guidance_scale,
                seed=config.generation.seed,
                provenance={"sample_id": clip.sample_id, "num_steps": num_steps},
            )
            generated = frames_to_tensor(result.frames)

            for sigma in blurs:
                cells.append(
                    measure_cell(
                        generated, reference, encoder,
                        sample_id=clip.sample_id,
                        scenario=scenario,
                        seed=config.generation.seed,
                        num_steps=num_steps,
                        blur_sigma=sigma,
                        layer=args.layer,
                        ar_order=config.metrics.ar_order,
                        residual_fit=config.metrics.residual_fit,
                    )
                )
        logger.info("[%d/%d] %s", index, len(clips), clip.sample_id)

    with open(output / "sweep.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SweepCell.fieldnames())
        writer.writeheader()
        writer.writerows(cell.flatten() for cell in cells)
    logger.info("wrote %s", output / "sweep.csv")

    report(cells, steps, blurs)


def report(cells, steps, blurs) -> None:
    low, high = steps[0], steps[-1]

    print(f"\nDoes the K={low} advantage survive the blur control?")
    print("=" * 74)
    print(f"{'metric':<26}{'sigma':>7}{f'K={low}':>11}{f'K={high}':>11}{'gap':>10}{'favoured':>9}")
    print("-" * 74)

    metrics = [(name, True) for name in STATISTICS] + [
        ("phase_difference_corr", False),
        ("raw_score", False),
    ]
    for name, lower_is_better in metrics:
        survival = blur_survival(
            cells, name, low_steps=low, high_steps=high, lower_is_more_regular=lower_is_better
        )
        for sigma in blurs:
            row = survival.get(sigma)
            if row is None:
                continue
            mark = "yes" if row["few_step_favoured"] else "no"
            print(
                f"{name:<26}{sigma:>7.0f}{row[f'k{low}']:>11.4f}{row[f'k{high}']:>11.4f}"
                f"{row['gap']:>10.4f}{mark:>9}"
            )
        print()

    print("Reading this: 'phase_difference_corr' is PhaseLock's own metric and is expected to")
    print(f"favour K={low} at every sigma (they report 0.358 vs 0.100 at sigma=16). A geometric")
    print("statistic that agrees at sigma=0 but flips under blur was measuring sharpness.")
    print("'raw_score' is ground-truth motion-mask fidelity and needs no assumption that")
    print("geometry measures physics, so it arbitrates when the two disagree.")


if __name__ == "__main__":
    main()
