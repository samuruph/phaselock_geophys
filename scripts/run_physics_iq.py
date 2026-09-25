#!/usr/bin/env python
"""PhaseLock on Physics-IQ: generate the benchmark's videos, and score them.

    python scripts/run_physics_iq.py --config configs/experiments/physics_iq.yaml
    python scripts/run_physics_iq.py --config ... --categories "Fluid Dynamics" --limit 8
    python scripts/run_physics_iq.py --config ... --guidance baseline motion accel jerk perr

Restored from `scripts/test_physics_iq.py`, which was in the initial commit and removed by
78f8f28 when the config-driven drivers landed. Physics-IQ was the one track that never got
a replacement, which is why the dataset class and the motion-mask metrics have sat built
and tested with nothing driving them.

**Only the 198 take-1 scenarios are generated.** Those are what the benchmark scores;
take-2 is the second recording of the same scenario -- the real-vs-real noise floor the
benchmark normalises against -- not extra prompts.

**Videos are written under the name the official evaluator expects**, the
`generated_video_name` column of `descriptions/best_practice/descriptions_base.csv`, one
directory per setting. So `videos/motion/` can be handed to the Physics-IQ repo with no
renaming, and its score compared against the one printed here. Two independent scorers
over the same files is the check that neither is quietly wrong.

`videos/few_step/` holds the 2-step generations the priors are taken from -- the prior in
RGB, which is the only way to see what a setting is actually matching against. Kept out of
the per-setting directories deliberately: it is the same clip for every setting, and the
evaluator asserts a setting's directory holds exactly 198 videos.

Three things differ from the original, and all three are the point:

* **It scores.** The original exported mp4s and stopped, so it could not produce a number
  without a second tool. Every generation is now also scored here against its real
  continuation with `phaselock/metrics/motion_mask.py`.
* **Every setting, one seed.** Each generates the same clips from the same seed and the
  same conditioning frame, so a per-sample delta is attributable. The original ran the
  guided pass alone, which cannot support a claim about guidance.
* **`--guidance` selects a frame operator.** `motion` is PhaseLock verbatim -- its latent delta
  operator is the first difference along frames. `accel`, `jerk` and `perr` hold a
  different quantity fixed through the identical mechanism; see `phaselock/operators.py`.
  The detection study says `motion` is the weakest of the four in this space, which is
  what this run tests.
* **A motion control.** Every geometric statistic is minimised by a static video, so
  guidance that "improves physics" by deleting the motion would score well on a mostly
  static scene. Mean absolute frame difference is reported per setting against the real
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
from diffusers.utils import load_image

from phaselock import load, set_seed
from phaselock.backends import load_backend
from phaselock.analysis.video import prompted_video
from phaselock.config import parse_overrides
from phaselock.datasets.physics_iq import PhysicsIQ
from phaselock.datasets.video_io import load_video
from phaselock.experiments.detection import frame_geometry
from phaselock.experiments.generation import frames_to_tensor
from phaselock.metrics.motion_mask import (MotionMaskScores, motion_mask_scores,
                                           physics_iq_score)
from phaselock.guidance import SOURCES
from phaselock.analysis.momentum_diagnostics import (
    MomentumTrace, StepPredictionCapture, render_dashboard,
)
from phaselock.operators import OPERATORS
from phaselock.pipelines.phaselock import PhaseLockPipeline
from phaselock.progress import track
from phaselock.utils import resolve_dtype

logger = logging.getLogger("run_physics_iq")

# Physics-IQ scores exactly the first 5.000 seconds; its `validate_generations` asserts
# that duration to 1 ms and its reference clips are 40 frames at 8 fps. CogVideoX emits 49
# frames -- 6.125 s -- so the generation is TRUNCATED to the benchmark window before
# scoring. Resampling the 5 s reference up to 49 frames instead would compare generated
# t=5.0s against real t=4.08s, misaligning progressively through the clip, and would not
# be the quantity the official evaluator reports.
BENCHMARK_SECONDS = 5.0

# A guidance setting is either the unguided control or one frame operator. "phaselock" is kept as an
# alias for "motion" so the flag reads the same as before the ablation existed.
GUIDANCES = ("baseline",) + tuple(OPERATORS)
GUIDANCE_ALIASES = {"phaselock": "motion"}


def resolve_guidance(name: str) -> str:
    return GUIDANCE_ALIASES.get(name, name)


def _variance_of(variance, field: str) -> float:
    """One physical-variance metric, or NaN for a scenario with no second real take."""
    return float("nan") if variance is None else getattr(variance, field)


def setting_name(prior_type: str, source: str) -> str:
    """The directory and row key for one cell of the grid.

    It must carry the *source* as well as the prior type. Keyed by type alone,
    `motion_on_x0_hat` found `videos/motion/` already full from the latent run and skipped
    every clip -- reporting OK in 58 seconds and producing results that were silently the
    latent ones. Two of the three sources had never actually run.
    """
    if prior_type == "baseline":
        return "baseline"
    return f"{prior_type}_on_{source}"


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--guidance", nargs="+", default=["baseline", "motion"],
                        choices=list(GUIDANCES) + list(GUIDANCE_ALIASES),
                        help="one control plus any frame operators; all run on the same "
                             "seed and clips, so every difference is paired")
    parser.add_argument("--source", default=None, choices=list(SOURCES) + ["blend"],
                        help="frame-difference source: running momentum uses same-timestep "
                             "post-step latent, current-step x0_hat, or their blend; few-step mode also accepts velocity")
    parser.add_argument("--categories", type=_list,
                        help="comma-separated, from: Solid Mechanics, Fluid Dynamics, "
                             "Optics, Thermodynamics, Magnetism. Default is all five")
    parser.add_argument("--perspectives", type=_list, help="left,center,right; default all")
    parser.add_argument("--limit", type=int,
                        help="cap the sample count, balanced across categories")
    parser.add_argument("--sample-id", nargs="+", type=str,
                        help="run exact take-1 sample IDs, e.g. --sample-id 0001 0007; "
                             "takes precedence over the balanced --limit")
    parser.add_argument("--save-prior", action=argparse.BooleanOptionalAction, default=True,
                        help="also write the few-step generation the prior is taken from, "
                             "to videos/few_step/. On by default: it is the prior in RGB, "
                             "and it costs only the export since the pass runs anyway")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate videos that are already on disk")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=None,
                        help="enable running-momentum denoising dashboards")
    parser.add_argument("overrides", nargs="*")
    # parse_known_args, not parse_args: argparse fills a positional nargs="*" from the
    # FIRST run of positionals it meets and rejects any later run, so an override that
    # lands after a flag kills the stage with "unrecognized arguments".
    args, extra = parser.parse_known_args()
    args.overrides = list(args.overrides) + extra
    return args


def motion_magnitude(frames: torch.Tensor) -> float:
    """Mean absolute frame difference -- the anti-freeze control.

    Not a quality measure. It exists so that "guidance scored better" can be told apart
    from "guidance stopped the video moving", which every geometric statistic rewards.
    """
    if frames.shape[0] < 2:
        return 0.0
    return float((frames[1:] - frames[:-1]).abs().mean())


def generate(pipeline: PhaseLockPipeline, name: str, want_prior: bool, seed: int,
             diagnostic_recorder=None, **kwargs):
    """One clip under one guidance setting, as ``(frames, prior_frames_or_None)``.

    Every guided setting runs the identical mechanism -- PhaseLock's equation (2) -- and
    differs only in which frame operator supplies the target, so `pipeline.few_step_prior_type`
    is the whole of the ablation.

    The baseline skips the few-step pass rather than running it at strength 0. Both give
    identical pixels -- `LatentDeltaGuidance.compute_schedule` returns 0 at every step and
    the callback then hands its kwargs back untouched -- but the prior pass plus its VAE
    round trip is real compute for a result that is thrown away.
    """
    if name == "baseline":
        shared = pipeline.backend.generation_kwargs(**kwargs)
        frames = pipeline.pipe(
            **shared,
            num_inference_steps=pipeline.full_steps,
            generator=torch.Generator(device=pipeline.backend.device).manual_seed(seed),
        ).frames[0]
        return frames, None

    pipeline.few_step_prior_type = name
    result = pipeline(**kwargs, seed=seed, return_few_result=want_prior,
                      diagnostic_recorder=diagnostic_recorder)
    return result if want_prior else (result, None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    overrides = parse_overrides(args.overrides)
    if args.diagnostics is not None:
        overrides["diagnostics__enabled"] = str(args.diagnostics).lower()
    config = load(args.config, **overrides)
    set_seed(config.generation.seed)

    output = config.output.dir()
    config.save(output / "config.json")

    dataset = PhysicsIQ(**({"root": config.data.root} if config.data.root else {}))
    guidances = [resolve_guidance(a) for a in args.guidance]
    samples = dataset.select_benchmark(
        categories=args.categories or config.data.categories,
        perspectives=args.perspectives,
        limit=(None if args.sample_id else
               (args.limit if args.limit is not None else config.data.limit)),
    )
    if args.sample_id:
        requested_ids = set(args.sample_id)
        samples = [sample for sample in samples if sample.sample_id in requested_ids]
        found_ids = {sample.sample_id for sample in samples}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise SystemExit(f"Physics-IQ take-1 sample IDs not found: {', '.join(missing_ids)}")
    # A sample with no switch frame cannot be conditioned, and one with no continuation
    # cannot be scored; neither can contribute a row.
    usable = [s for s in samples if s.image_path and s.reference_path]
    if len(usable) < len(samples):
        logger.info("%d of %d samples dropped: missing switch frame or continuation",
                    len(samples) - len(usable), len(samples))
    if not usable:
        raise SystemExit(f"no usable Physics-IQ samples under {dataset.root}")
    if args.sample_id:
        logger.info("selected samples: %s", ", ".join(sample.sample_id for sample in usable))

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
    # Generate the model's native length, score the benchmark's window.
    score_frames = round(BENCHMARK_SECONDS * spec.default_fps)
    if score_frames > num_frames:
        raise SystemExit(
            f"the benchmark scores {BENCHMARK_SECONDS}s = {score_frames} frames but the "
            f"backend only generates {num_frames}"
        )

    running = config.phaselock.prior_mode == "running_momentum"
    source = args.source or (config.phaselock.running_momentum_source if running
                             else config.phaselock.few_step_prior_source)
    if running and source not in {"latent", "x0_hat", "blend"}:
        raise SystemExit("running momentum --source must be latent, x0_hat, or blend")
    if not running and source == "blend":
        raise SystemExit("few-step --source must be latent, x0_hat, or velocity")
    pipeline = PhaseLockPipeline(
        backend,
        prior_mode=config.phaselock.prior_mode,
        betas=(config.phaselock.beta1, config.phaselock.beta2),
        running_momentum_mode=config.phaselock.running_momentum_mode,
        velocity_decay=config.phaselock.velocity_decay,
        few_steps=config.phaselock.few_steps,
        full_steps=config.generation.num_steps,
        guidance_strength=config.phaselock.guidance_strength,
        guide_start=config.phaselock.guide_start,
        guide_end=config.phaselock.guide_end,
        source=source,
    )
    if pipeline.source != source:
        raise SystemExit(f"asked for source {source!r} but the pipeline has "
                         f"{pipeline.source!r}")

    logger.info(
        "%d of %d take-1 scenarios x [%s] on %s, generating %d frames at %d fps, "
        "scoring the first %d (%.1fs, the benchmark window)",
        len(usable), len(dataset.generation_samples()), " ".join(guidances),
        config.backend.name, num_frames, spec.default_fps, score_frames,
        BENCHMARK_SECONDS,
    )
    # `latent` keeps its established post-step setting name; x0_hat is the
    # prediction made from the pre-step input at t and gets an explicit suffix.
    setting_source = "x0_hat_pred_at_t" if running and source == "x0_hat" else source
    settings = {name: setting_name(name, setting_source) for name in guidances}
    for folder in settings.values():
        (output / "videos" / folder).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sample in track(usable, "physics-iq", unit="clip"):
        # Loaded at the benchmark's own length, which is what its reference clips are,
        # so this is an identity resample rather than a stretch.
        reference = load_video(sample.reference_path, num_frames=score_frames,
                               height=height, width=width)
        # The benchmark's PHYSICAL VARIANCE: a SECOND real recording of the same scene,
        # scored against the first. Reality repeating itself does not score 1.0 on a motion
        # mask -- lighting flickers, the camera is not bit-identical -- so each metric is
        # only interpretable against this. It depends on the real footage alone, so it is
        # computed once per clip and reused by every setting.
        #
        # Stored per row rather than folded into a per-clip score, because the benchmark
        # pools numerators and denominators over the whole set before dividing. See
        # phaselock.metrics.motion_mask.physics_iq_score.
        variance = None
        if sample.meta.get("pair_path"):
            take_two = load_video(sample.meta["pair_path"], num_frames=score_frames,
                                  height=height, width=width)
            variance = motion_mask_scores(take_two, reference)
        image = load_image(sample.image_path)
        video_name = sample.meta["output_name"]

        for name in guidances:
            path = output / "videos" / settings[name] / video_name
            diagnostic_enabled = (config.diagnostics.enabled
                                  and config.phaselock.prior_mode == "running_momentum")
            if path.exists() and not args.overwrite:
                if name == "baseline" or not diagnostic_enabled:
                    continue

            diagnostic_trace = None
            diagnostic_dir = (output / config.diagnostics.output_subdir / Path(video_name).stem / settings[name])
            diagnostic_path = diagnostic_dir / "dashboard.mp4"
            if (config.diagnostics.enabled and name != "baseline"
                    and config.phaselock.prior_mode == "running_momentum"):
                diagnostic_trace = MomentumTrace(name, save_raw=config.diagnostics.save_raw_tensors,
                                             record_steps=(set(config.diagnostics.record_steps)
                                                           if config.diagnostics.record_steps else None))
            frames, prior = generate(
                pipeline, name, args.save_prior, config.generation.seed,
                diagnostic_recorder=diagnostic_trace,
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
            if not path.exists() or args.overwrite or diagnostic_trace is not None:
                # Keep the evaluator's exact filename and pixels; write a sibling for viewing.
                prompted_video(frames_to_tensor(frames),
                               path.with_name(f"{path.stem}_prompted{path.suffix}"),
                               sample.prompt, fps=spec.default_fps)
            if diagnostic_trace is not None and diagnostic_trace.steps:
                diagnostic_provenance = {
                    "sample_id": sample.sample_id, "guidance": name,
                    "backend": config.backend.name, "seed": config.generation.seed,
                    "running_momentum_source": source,
                    "source_state": ("post_scheduler_step_callback_latents" if source == "latent"
                                     else "model_prediction_x0_hat_from_pre_step_latents_at_t" if source == "x0_hat"
                                     else "smooth_blend_of_post_step_latents_and_current_step_x0_hat"),
                    "correction_target": "post_scheduler_step_latents",
                }
                diagnostic_trace.save(diagnostic_dir, provenance=diagnostic_provenance)
                render_dashboard(
                    diagnostic_trace, diagnostic_path,
                    decode=lambda latent: backend.decode(latent.to(backend.device)),
                    fps=config.diagnostics.fps,
                    preview_height=config.diagnostics.preview_height,
                    temporal_ratio=spec.temporal_ratio,
                    provenance=diagnostic_provenance,
                )
            # The few-step generation is identical for every setting on a clip -- same
            # seed, same two steps, no guidance -- so it is written once, and to its own
            # directory. Alongside the others it would both duplicate 12 times and break
            # the official evaluator, which counts the .mp4 files in a setting's directory
            # and asserts exactly 198.
            if prior is not None:
                few_path = output / "videos" / "few_step" / video_name
                if not few_path.exists() or args.overwrite:
                    few_path.parent.mkdir(parents=True, exist_ok=True)
                    prompted_video(frames_to_tensor(prior),
                                   few_path, sample.prompt, fps=spec.default_fps)

            # The full clip goes to disk -- the official evaluator does its own trim --
            # but only the benchmark window is scored here.
            generated = frames_to_tensor(frames)[:score_frames]
            scores = motion_mask_scores(generated, reference)
            rows.append({
                "sample_id": sample.sample_id,
                "output_name": video_name,
                "category": sample.category,
                "perspective": sample.meta["perspective"],
                "guidance": name,
                "setting": settings[name],
                "guidance_strength": (0.0 if name == "baseline"
                                      else config.phaselock.guidance_strength),
                "few_step_prior_type": "" if name == "baseline" else name,
                "few_step_prior_source": (pipeline.source if name != "baseline" and not running else ""),
                "running_momentum_source": (pipeline.source if name != "baseline" and running else ""),
                "momentum_source_timing": (("post_scheduler_step_callback_latents" if pipeline.source == "latent"
                                            else "model_prediction_x0_hat_from_pre_step_latents_at_t" if pipeline.source == "x0_hat"
                                            else "smooth_blend_of_post_step_latents_and_current_step_x0_hat")
                                           if name != "baseline" and running else ""),
                "prior_rms": (float("nan") if name == "baseline"
                              else pipeline.last_prior_rms),
                "spatial_iou": scores.spatial_iou,
                "spatiotemporal_iou": scores.spatiotemporal_iou,
                "weighted_spatial_iou": scores.weighted_spatial_iou,
                "mse": scores.mse,
                "raw_score": scores.raw_score,
                # The real-vs-real value of each metric on this scene. The reportable
                # Physics-IQ score is built from these across all clips, not per clip.
                "variance_spatial_iou": _variance_of(variance, "spatial_iou"),
                "variance_spatiotemporal_iou": _variance_of(variance,
                                                            "spatiotemporal_iou"),
                "variance_weighted_spatial_iou": _variance_of(variance,
                                                              "weighted_spatial_iou"),
                "variance_mse": _variance_of(variance, "mse"),
                "motion": motion_magnitude(generated),
                "reference_motion": motion_magnitude(reference),
            })

        torch.cuda.empty_cache()
        gc.collect()

    if not rows:
        logger.info("every video already on disk; nothing regenerated. "
                    "Pass --overwrite to redo them.")
        return

    # One file per setting. A shared physics_iq.csv opened with "w" meant each invocation
    # erased the last one's rows, so a five-setting ablation ended with only the fifth --
    # and nothing to compare it against. report_guidance.py globs these.
    path = output / f"physics_iq_{settings[guidances[-1]]}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", path)

    report(rows, [g for g in guidances if any(r["guidance"] == g for r in rows)])
    print(
        f"\nVideos are under {output / 'videos'}/<guidance>/, named as the official evaluator\n"
        "expects, so a directory can be handed to the Physics-IQ repo unchanged. Its score\n"
        "and the one above are two independent implementations over the same files."
    )


VARIANCE_FIELDS = {
    "spatial_iou": "variance_spatial_iou",
    "spatiotemporal_iou": "variance_spatiotemporal_iou",
    "weighted_spatial_iou": "variance_weighted_spatial_iou",
    "mse": "variance_mse",
}


def score_of(rows: list[dict]) -> float:
    """The Physics-IQ score for a set of rows, pooled the way the benchmark pools it.

    Rows whose scenario has no second real take carry NaN variances and are dropped: they
    have no reference to normalise against, and one NaN would poison the pooled mean.
    """
    usable = [r for r in rows
              if all(r[v] == r[v] for v in VARIANCE_FIELDS.values())]
    if not usable:
        return float("nan")
    scores = [MotionMaskScores(**{k: float(r[k]) for k in VARIANCE_FIELDS})
              for r in usable]
    variances = [MotionMaskScores(**{k: float(r[v]) for k, v in VARIANCE_FIELDS.items()})
                 for r in usable]
    return physics_iq_score(scores, variances)


def report(rows: list[dict], guidances: list[str]) -> None:
    """Per-guidance means, then each one's paired delta against the unguided baseline.

    Paired, not just a difference of means: every setting generated the same clips from
    the same seed, so a per-sample delta removes the clip-to-clip variance that otherwise
    swamps a few points of guidance effect.
    """
    import statistics as stats_module

    by_guidance = {g: [r for r in rows if r["guidance"] == g] for g in guidances}
    metrics = [("raw_score", True), ("spatial_iou", True), ("spatiotemporal_iou", True),
               ("weighted_spatial_iou", True), ("mse", False), ("motion", None)]

    print(f"\n{len(by_guidance[guidances[0]])} clips\n")
    # THE number, and the only one that is comparable to a published Physics-IQ result.
    # It is a property of the whole set of clips, not of any one of them, so it is printed
    # as a single row here rather than averaged out of a column.
    print(f"{'PHYSICS-IQ SCORE %':<24}"
          + "".join(f"{score_of(by_guidance[g]):>14.2f}" for g in guidances))
    print()
    print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in guidances))
    print("-" * (24 + 14 * len(guidances)))
    for metric, higher_is_better in metrics:
        line = f"{metric:<24}"
        for g in guidances:
            line += f"{stats_module.mean(r[metric] for r in by_guidance[g]):>14.4f}"
        if higher_is_better is None:
            line += "   (control, not a score)"
        print(line)

    priors = {g: [r["prior_rms"] for r in by_guidance[g]] for g in guidances if g != "baseline"}
    if priors:
        print("\nprior magnitude (RMS of the target each setting matches)")
        for setting, values in priors.items():
            print(f"  {setting:<10}{stats_module.mean(values):>12.5f}")
        print("  lambda is shared, so magnitudes differing by orders of "
              "magnitude\n  mean the same lambda is a different intervention -- see the "
              "run's notes.")

    guided = [g for g in guidances if g != "baseline"]
    if "baseline" in by_guidance and guided:
        print(f"\npaired delta against baseline (+ is better, except mse)\n")
        print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in guided))
        print("-" * (24 + 14 * len(guided)))
        control = {
            metric: {r["sample_id"]: r[metric] for r in by_guidance["baseline"]}
            for metric, _ in metrics
        }
        for metric, _ in metrics:
            line = f"{metric:<24}"
            for g in guided:
                deltas = [r[metric] - control[metric][r["sample_id"]] for r in by_guidance[g]
                          if r["sample_id"] in control[metric]]
                line += (f"{stats_module.mean(deltas):>+14.4f}" if deltas
                         else f"{'-':>14}")
            print(line)

    reference = stats_module.mean(r["reference_motion"] for r in by_guidance[guidances[0]])
    print(f"\nreal continuations move {reference:.4f} per frame. A setting that scores better "
          "while\nmoving markedly less has found the static optimum, not physics.")


if __name__ == "__main__":
    main()
