#!/usr/bin/env python
"""PhaseLock on Physics-IQ: generate the benchmark's videos, and score them.

    python scripts/run_physics_iq.py --config configs/experiments/physics_iq.yaml
    python scripts/run_physics_iq.py --config ... --categories "Fluid Dynamics" --limit 8
    python scripts/run_physics_iq.py --config ... --arms phaselock      # guided only

Restored from `scripts/test_physics_iq.py`, which was in the initial commit and removed by
78f8f28 when the config-driven drivers landed. Physics-IQ was the one track that never got
a replacement, which is why the dataset class and the motion-mask metrics have sat built
and tested with nothing driving them.

**Only the 198 take-1 scenarios are generated.** Those are what the benchmark scores;
take-2 is the second recording of the same scenario -- the real-vs-real noise floor the
benchmark normalises against -- not extra prompts.

**Videos are written under the name the official evaluator expects**, the
`generated_video_name` column of `descriptions/best_practice/descriptions_base.csv`, one
directory per arm. So `videos/phaselock/` can be handed to the Physics-IQ repo with no
renaming, and its score compared against the one printed here. Two independent scorers
over the same files is the check that neither is quietly wrong.

Three things differ from the original, and all three are the point:

* **It scores.** The original exported mp4s and stopped, so it could not produce a number
  without a second tool. Every generation is now also scored here against its real
  continuation with `phaselock/metrics/motion_mask.py`.
* **Both arms, one seed.** Guided and unguided run from the same seed and the same
  conditioning frame, so the per-sample difference is attributable. The original ran the
  guided arm alone, which cannot support a claim about guidance.
* **A motion control.** Every geometric statistic is minimised by a static video, so
  guidance that "improves physics" by deleting the motion would score well on a mostly
  static scene. Mean absolute frame difference is reported per arm against the real
  continuation's; a fidelity gain that arrives with a motion collapse is a failure.

The paper's settings are the defaults, in `configs/experiments/physics_iq.yaml`:
`few_steps=2`, `full_steps=50`, `guidance_strength=0.05`, `guide_start=0`, `guide_end=25`,
`guidance_scale=6.0`, 49 frames at 8 fps.
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
from diffusers.utils import export_to_video, load_image

from phaselock import load, set_seed
from phaselock.backends import load_backend
from phaselock.config import parse_overrides
from phaselock.datasets.physics_iq import PhysicsIQ
from phaselock.datasets.video_io import load_video
from phaselock.experiments.detection import frame_geometry
from phaselock.experiments.generation import frames_to_tensor
from phaselock.metrics.motion_mask import motion_mask_scores
from phaselock.pipelines.phaselock import PhaseLockPipeline
from phaselock.progress import track
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_physics_iq")

ARMS = ("baseline", "phaselock")


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                        help="both, on the same seed, is what makes the difference paired")
    parser.add_argument("--categories", type=_list,
                        help="names from the CSV's category column; default is all five")
    parser.add_argument("--perspectives", type=_list, help="left,center,right; default all")
    parser.add_argument("--limit", type=int,
                        help="cap the sample count, balanced across categories")
    parser.add_argument("--save-prior", action="store_true",
                        help="also write the few-step prior as few_<name>.mp4")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate videos that are already on disk")
    parser.add_argument("overrides", nargs="*")
    # parse_known_args, not parse_args: argparse fills a positional nargs="*" from the
    # FIRST run of positionals it meets and rejects any later run, so an override that
    # lands after a flag kills the stage with "unrecognized arguments".
    args, extra = parser.parse_known_args()
    args.overrides = list(args.overrides) + extra
    return args


def motion_magnitude(frames: torch.Tensor) -> float:
    """Mean absolute frame difference -- the anti-freeze control.

    Not a quality measure. It exists so that "the guided arm scored better" can be told
    apart from "the guided arm stopped moving", which every geometric statistic rewards.
    """
    if frames.shape[0] < 2:
        return 0.0
    return float((frames[1:] - frames[:-1]).abs().mean())


def generate(pipeline: PhaseLockPipeline, arm: str, want_prior: bool, seed: int, **kwargs):
    """One clip from one arm, as ``(frames, prior_frames_or_None)``.

    The baseline skips the few-step pass rather than running it at strength 0. Both give
    identical pixels -- `LatentDeltaGuidance.compute_schedule` returns 0 at every step and
    the callback then hands its kwargs back untouched -- but the prior pass plus its VAE
    round trip is real compute for a result that is thrown away.
    """
    if arm == "baseline":
        shared = pipeline.backend.generation_kwargs(**kwargs)
        frames = pipeline.pipe(
            **shared,
            num_inference_steps=pipeline.full_steps,
            generator=torch.Generator(device=pipeline.backend.device).manual_seed(seed),
        ).frames[0]
        return frames, None

    result = pipeline(**kwargs, seed=seed, return_few_result=want_prior)
    return result if want_prior else (result, None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    config = load(args.config, **parse_overrides(args.overrides))
    set_seed(config.generation.seed)

    output = config.output.dir()
    config.save(output / "config.json")

    dataset = PhysicsIQ(**({"root": config.data.root} if config.data.root else {}))
    samples = dataset.select_benchmark(
        categories=args.categories or config.data.categories,
        perspectives=args.perspectives,
        limit=args.limit if args.limit is not None else config.data.limit,
    )
    # A sample with no switch frame cannot be conditioned, and one with no continuation
    # cannot be scored; neither can contribute a row.
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
    spec = backend.spec
    # CogVideoX refuses any geometry but its own, so this returns the native one whatever
    # the config asks -- which is also the geometry the benchmark's references are read at.
    num_frames, height, width = frame_geometry(spec, config)

    pipeline = PhaseLockPipeline(
        backend,
        few_steps=config.phaselock.few_steps,
        full_steps=config.generation.num_steps,
        guidance_strength=config.phaselock.guidance_strength,
        guide_start=config.phaselock.guide_start,
        guide_end=config.phaselock.guide_end,
    )

    logger.info(
        "%d of %d take-1 scenarios x [%s] on %s, %d frames at %d fps",
        len(usable), len(dataset.generation_samples()), " ".join(args.arms),
        config.backend.name, num_frames, spec.default_fps,
    )
    for arm in args.arms:
        (output / "videos" / arm).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sample in track(usable, "physics-iq", unit="clip"):
        # Loaded at the geometry the generation will have, so the two are directly
        # comparable without a second resample.
        reference = load_video(sample.reference_path, num_frames=num_frames,
                               height=height, width=width)
        image = load_image(sample.image_path)
        name = sample.meta["output_name"]

        for arm in args.arms:
            path = output / "videos" / arm / name
            if path.exists() and not args.overwrite:
                continue

            frames, prior = generate(
                pipeline, arm, args.save_prior, config.generation.seed,
                prompt=sample.prompt,
                image=image,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=config.generation.guidance_scale,
                negative_prompt=config.generation.negative_prompt,
            )

            # The evaluator-facing file: the pipeline's own frames, under the name the
            # Physics-IQ repo expects.
            export_to_video(frames, str(path), fps=spec.default_fps)
            if prior is not None:
                export_to_video(prior, str(path.parent / f"few_{name}"),
                                fps=spec.default_fps)

            generated = frames_to_tensor(frames)
            scores = motion_mask_scores(generated, reference)
            rows.append({
                "sample_id": sample.sample_id,
                "output_name": name,
                "category": sample.category,
                "perspective": sample.meta["perspective"],
                "arm": arm,
                "guidance_strength": (0.0 if arm == "baseline"
                                      else config.phaselock.guidance_strength),
                "spatial_iou": scores.spatial_iou,
                "spatiotemporal_iou": scores.spatiotemporal_iou,
                "weighted_spatial_iou": scores.weighted_spatial_iou,
                "mse": scores.mse,
                "raw_score": scores.raw_score,
                "motion": motion_magnitude(generated),
                "reference_motion": motion_magnitude(reference),
            })

        torch.cuda.empty_cache()
        gc.collect()

    if not rows:
        logger.info("every video already on disk; nothing regenerated. "
                    "Pass --overwrite to redo them.")
        return

    path = output / "physics_iq.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", path)

    report(rows, [a for a in args.arms if any(r["arm"] == a for r in rows)])
    print(
        f"\nVideos are under {output / 'videos'}/<arm>/, named as the official evaluator\n"
        "expects, so a directory can be handed to the Physics-IQ repo unchanged. Its score\n"
        "and the one above are two independent implementations over the same files."
    )


def report(rows: list[dict], arms: list[str]) -> None:
    """Per-arm means, and the paired difference that is the actual claim."""
    import statistics as stats_module

    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    metrics = [("raw_score", True), ("spatial_iou", True), ("spatiotemporal_iou", True),
               ("weighted_spatial_iou", True), ("mse", False), ("motion", None)]

    print(f"\n{len(by_arm[arms[0]])} clips\n")
    paired = len(arms) == 2
    print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in arms) +
          (f"{'paired diff':>14}" if paired else ""))
    print("-" * (24 + 14 * (len(arms) + (1 if paired else 0))))
    for name, higher_is_better in metrics:
        line = f"{name:<24}"
        for arm in arms:
            line += f"{stats_module.mean(r[name] for r in by_arm[arm]):>14.4f}"
        if paired:
            first = {r["sample_id"]: r[name] for r in by_arm[arms[0]]}
            deltas = [r[name] - first[r["sample_id"]] for r in by_arm[arms[1]]
                      if r["sample_id"] in first]
            if deltas:
                line += f"{stats_module.mean(deltas):>+14.4f}"
        if higher_is_better is None:
            line += "   (control, not a score)"
        print(line)

    reference = stats_module.mean(r["reference_motion"] for r in by_arm[arms[0]])
    print(f"\nreal continuations move {reference:.4f} per frame. An arm that scores better "
          "while\nmoving markedly less has found the static optimum, not physics.")


if __name__ == "__main__":
    main()
