#!/usr/bin/env python
"""PhaseLock on Physics-IQ: generate the benchmark's videos, and score them.

    python scripts/run_physics_iq.py --config configs/experiments/physics_iq.yaml
    python scripts/run_physics_iq.py --config ... --categories "Fluid Dynamics" --limit 8
    python scripts/run_physics_iq.py --config ... --arms baseline motion accel jerk perr

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
* **Every arm, one seed.** Each arm generates the same clips from the same seed and the
  same conditioning frame, so a per-sample delta is attributable. The original ran the
  guided arm alone, which cannot support a claim about guidance.
* **The arms are frame operators.** `motion` is PhaseLock verbatim -- its latent delta
  operator is the first difference along frames. `accel`, `jerk` and `perr` hold a
  different quantity fixed through the identical mechanism; see `phaselock/operators.py`.
  The detection study says `motion` is the weakest of the four in this space, which is
  what this run tests.
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
from phaselock.operators import OPERATORS
from phaselock.pipelines.phaselock import PhaseLockPipeline
from phaselock.progress import track
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_physics_iq")

# An arm is either the unguided control or one frame operator. "phaselock" is kept as an
# alias for "motion" so the flag reads the same as before the ablation existed.
ARMS = ("baseline",) + tuple(OPERATORS)
ARM_ALIASES = {"phaselock": "motion"}


def resolve_arm(name: str) -> str:
    return ARM_ALIASES.get(name, name)


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--arms", nargs="+", default=["baseline", "motion"],
                        choices=list(ARMS) + list(ARM_ALIASES),
                        help="one control plus any frame operators; all run on the same "
                             "seed and clips, so every difference is paired")
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

    Every guided arm runs the identical mechanism -- PhaseLock's equation (2) -- and
    differs only in which frame operator supplies the target, so `pipeline.operator` is
    the whole of the ablation.

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

    pipeline.operator = arm
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
    arms = [resolve_arm(a) for a in args.arms]
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
        len(usable), len(dataset.generation_samples()), " ".join(arms),
        config.backend.name, num_frames, spec.default_fps,
    )
    for arm in arms:
        (output / "videos" / arm).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sample in track(usable, "physics-iq", unit="clip"):
        # Loaded at the geometry the generation will have, so the two are directly
        # comparable without a second resample.
        reference = load_video(sample.reference_path, num_frames=num_frames,
                               height=height, width=width)
        image = load_image(sample.image_path)
        name = sample.meta["output_name"]

        for arm in arms:
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
                "operator": "" if arm == "baseline" else arm,
                "prior_rms": (float("nan") if arm == "baseline"
                              else pipeline.last_prior_rms),
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

    report(rows, [a for a in arms if any(r["arm"] == a for r in rows)])
    print(
        f"\nVideos are under {output / 'videos'}/<arm>/, named as the official evaluator\n"
        "expects, so a directory can be handed to the Physics-IQ repo unchanged. Its score\n"
        "and the one above are two independent implementations over the same files."
    )


def report(rows: list[dict], arms: list[str]) -> None:
    """Per-arm means, then every guided arm's paired delta against the baseline.

    Paired, not just a difference of means: each arm generated the same clips from the
    same seed, so a per-sample delta removes the clip-to-clip variance that otherwise
    swamps a few points of guidance effect.
    """
    import statistics as stats_module

    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    metrics = [("raw_score", True), ("spatial_iou", True), ("spatiotemporal_iou", True),
               ("weighted_spatial_iou", True), ("mse", False), ("motion", None)]

    print(f"\n{len(by_arm[arms[0]])} clips\n")
    print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in arms))
    print("-" * (24 + 14 * len(arms)))
    for name, higher_is_better in metrics:
        line = f"{name:<24}"
        for arm in arms:
            line += f"{stats_module.mean(r[name] for r in by_arm[arm]):>14.4f}"
        if higher_is_better is None:
            line += "   (control, not a score)"
        print(line)

    priors = {a: [r["prior_rms"] for r in by_arm[a]] for a in arms if a != "baseline"}
    if priors:
        print("\nprior magnitude (RMS of the target the arm matches)")
        for arm, values in priors.items():
            print(f"  {arm:<10}{stats_module.mean(values):>12.5f}")
        print("  lambda is shared across arms, so magnitudes differing by orders of "
              "magnitude\n  mean the same lambda is a different intervention -- see the "
              "run's notes.")

    guided = [a for a in arms if a != "baseline"]
    if "baseline" in by_arm and guided:
        print(f"\npaired delta against baseline (+ is better, except mse)\n")
        print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in guided))
        print("-" * (24 + 14 * len(guided)))
        control = {
            name: {r["sample_id"]: r[name] for r in by_arm["baseline"]}
            for name, _ in metrics
        }
        for name, _ in metrics:
            line = f"{name:<24}"
            for arm in guided:
                deltas = [r[name] - control[name][r["sample_id"]] for r in by_arm[arm]
                          if r["sample_id"] in control[name]]
                line += (f"{stats_module.mean(deltas):>+14.4f}" if deltas
                         else f"{'-':>14}")
            print(line)

    reference = stats_module.mean(r["reference_motion"] for r in by_arm[arms[0]])
    print(f"\nreal continuations move {reference:.4f} per frame. An arm that scores better "
          "while\nmoving markedly less has found the static optimum, not physics.")


if __name__ == "__main__":
    main()
