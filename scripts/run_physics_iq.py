#!/usr/bin/env python
"""PhaseLock on Physics-IQ: guided against unguided, on the paper's own benchmark.

    python scripts/run_physics_iq.py --config configs/experiments/physics_iq.yaml
    python scripts/run_physics_iq.py --config ... data__limit=8 --arms baseline phaselock

Restored from `scripts/test_physics_iq.py`, which was in the initial commit and removed by
78f8f28 when the config-driven drivers landed. Physics-IQ was the one track that never got
a replacement, so this is the gap being filled rather than a new experiment.

Three things are different from the original, and all three are the point:

* **It scores.** The original exported two mp4s per sample and stopped, so it could not
  produce a Physics-IQ number. `phaselock/metrics/motion_mask.py` has been built and
  tested since, so every generation is now scored against its real continuation.
* **Both arms, one seed.** Guided and unguided are generated from the same seed and the
  same conditioning frame, so the comparison is *paired* and a per-sample difference is
  meaningful. The original ran the guided arm alone.
* **A motion control.** Every geometric statistic is minimised by a static video, so
  guidance that "improves physics" by deleting the motion would score well on a mostly
  static scene. Mean absolute frame difference is reported per arm; a fidelity gain that
  arrives with a motion collapse is a failure, not a result.

The paper's settings are the defaults: `few_steps=2`, `full_steps=50`,
`guidance_strength=0.05`, `guide_start=0`, `guide_end=25`, `guidance_scale=6.0`.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
from phaselock.datasets import PhysicsIQ
from phaselock.datasets.video_io import load_video, save_video
from phaselock.experiments.detection import frame_geometry
from phaselock.experiments.generation import frames_to_tensor
from phaselock.metrics.motion_mask import motion_mask_scores
from phaselock.pipelines.phaselock import PhaseLockPipeline
from phaselock.progress import track
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_physics_iq")

ARMS = ("baseline", "phaselock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                        help="which arms to run; both, on the same seed, is the point")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("overrides", nargs="*")
    # parse_known_args, not parse_args: argparse fills a positional nargs="*" from the
    # FIRST run of positionals it meets and rejects any later run, so an override that
    # lands after a flag kills the stage with "unrecognized arguments".
    args, extra = parser.parse_known_args()
    args.overrides = list(args.overrides) + extra
    return args


def motion_magnitude(frames: torch.Tensor) -> float:
    """Mean absolute frame difference -- the anti-freeze control.

    Not a quality measure. It exists so that "the guided arm scored better" can be
    separated from "the guided arm stopped moving", which every geometric statistic
    rewards.
    """
    if frames.shape[0] < 2:
        return 0.0
    return float((frames[1:] - frames[:-1]).abs().mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    config = load(args.config, **parse_overrides(args.overrides))
    set_seed(config.generation.seed)

    output = config.output.dir()
    config.save(output / "config.json")
    videos = output / "videos"
    if args.save_videos:
        videos.mkdir(parents=True, exist_ok=True)

    dataset = PhysicsIQ(**({"root": config.data.root} if config.data.root else {}))
    samples = dataset.select_generation(
        categories=config.data.categories, limit=config.data.limit, seed=config.data.seed
    )
    # A sample with no switch frame cannot be conditioned and one with no continuation
    # cannot be scored, so neither can contribute a row.
    usable = [s for s in samples if s.image_path and s.reference_path]
    if len(usable) < len(samples):
        logger.info("%d of %d samples dropped: missing switch frame or continuation",
                    len(samples) - len(usable), len(samples))
    if not usable:
        raise SystemExit(f"no usable Physics-IQ samples under {dataset.root}")

    backend = load_backend(
        config.backend.name,
        model_id=config.backend.model_id,
        torch_dtype=resolve_dtype(config.backend.dtype),
        enable_offload=config.backend.offload,
    )
    num_frames, height, width = frame_geometry(backend.spec, config)

    pipeline = PhaseLockPipeline(
        backend,
        few_steps=config.phaselock.few_steps,
        full_steps=config.generation.num_steps,
        guidance_strength=config.phaselock.guidance_strength,
        guide_start=config.phaselock.guide_start,
        guide_end=config.phaselock.guide_end,
    )

    logger.info("%d samples x %d arms on %s", len(usable), len(args.arms),
                config.backend.name)

    from PIL import Image

    rows: list[dict] = []
    for sample in track(usable, "physics-iq", unit="clip"):
        # The reference is loaded at the geometry the generation will have, so the two are
        # directly comparable without a second resample.
        reference = load_video(sample.reference_path, num_frames=num_frames,
                               height=height, width=width)
        image = Image.open(sample.image_path).convert("RGB")

        for arm in args.arms:
            strength = 0.0 if arm == "baseline" else config.phaselock.guidance_strength
            pipeline.guidance_strength = strength
            # Same seed in both arms: the arms differ only by guidance, so the difference
            # per sample is attributable rather than a draw from two unrelated samples.
            frames = pipeline(
                prompt=sample.prompt,
                image=image,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=config.generation.guidance_scale,
                negative_prompt=config.generation.negative_prompt,
                seed=config.generation.seed,
            )
            generated = frames_to_tensor(frames)
            scores = motion_mask_scores(generated, reference)

            rows.append({
                "sample_id": sample.sample_id,
                "category": sample.category,
                "arm": arm,
                "guidance_strength": strength,
                "spatial_iou": scores.spatial_iou,
                "spatiotemporal_iou": scores.spatiotemporal_iou,
                "weighted_spatial_iou": scores.weighted_spatial_iou,
                "mse": scores.mse,
                "raw_score": scores.raw_score,
                "motion": motion_magnitude(generated),
                "reference_motion": motion_magnitude(reference),
            })

            if args.save_videos:
                save_video(generated, str(videos / f"{sample.sample_id}_{arm}.mp4"),
                           fps=8)

        torch.cuda.empty_cache()
        gc.collect()

    path = output / "physics_iq.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", path)

    report(rows, args.arms)


def report(rows: list[dict], arms: list[str]) -> None:
    """Per-arm means, and the paired difference that is the actual claim."""
    import statistics as stats_module

    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    metrics = [("raw_score", True), ("spatial_iou", True), ("spatiotemporal_iou", True),
               ("weighted_spatial_iou", True), ("mse", False), ("motion", None)]

    print(f"\n{len(by_arm[arms[0]])} clips\n")
    print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in arms) +
          (f"{'paired diff':>14}" if len(arms) == 2 else ""))
    print("-" * (24 + 14 * (len(arms) + (1 if len(arms) == 2 else 0))))
    for name, higher_is_better in metrics:
        line = f"{name:<24}"
        for arm in arms:
            line += f"{stats_module.mean(r[name] for r in by_arm[arm]):>14.4f}"
        if len(arms) == 2:
            paired = {r["sample_id"]: r[name] for r in by_arm[arms[0]]}
            deltas = [r[name] - paired[r["sample_id"]] for r in by_arm[arms[1]]
                      if r["sample_id"] in paired]
            line += f"{stats_module.mean(deltas):>+14.4f}"
        if higher_is_better is None:
            line += "   (control, not a score)"
        print(line)

    if len(arms) == 2:
        reference = stats_module.mean(r["reference_motion"] for r in by_arm[arms[0]])
        print(f"\nreal continuations move {reference:.4f} per frame. A guided arm that "
              "scores\nbetter while moving markedly less has found the static optimum, "
              "not physics.")


if __name__ == "__main__":
    main()
