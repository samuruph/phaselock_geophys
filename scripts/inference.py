#!/usr/bin/env python
"""Single-clip generation, with or without PhaseLock guidance.

    # Wan2.1 smoke test on local weights
    python scripts/inference.py --backend wan21_t2v_1_3b \
        --prompt "a ball bouncing on a table" --output /tmp/wan.mp4

    # CogVideoX image-to-video with guidance
    python scripts/inference.py --backend cogvideox_5b_i2v \
        --prompt "water pouring into a glass" --image glass.jpg --output out.mp4

    # baseline, no guidance
    python scripts/inference.py --backend cogvideox_5b_i2v --prompt "..." \
        --image glass.jpg --output base.mp4 --no-phaselock

Frame count, resolution and frame rate default to the backend's native settings, which
is what the PhaseLock paper used per model: 49 frames at 8 fps for CogVideoX, 81 at
16 fps for Wan2.1.
"""

from __future__ import annotations

import argparse
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

from phaselock import set_seed
from phaselock.backends import BACKENDS, get_entry, get_spec, load_backend
from phaselock.pipelines.phaselock import PhaseLockPipeline
from phaselock.utils import resolve_dtype

logger = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="cogvideox_5b_i2v", choices=sorted(BACKENDS))
    parser.add_argument("--model-id", default=None, help="override the registry default")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--image", default=None, help="conditioning image, required for i2v backends")
    parser.add_argument("--output", default="output.mp4")

    parser.add_argument("--num-frames", type=int, default=None, help="default: the backend's native count")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")

    parser.add_argument("--few-steps", type=int, default=2)
    parser.add_argument("--full-steps", type=int, default=50)
    parser.add_argument("--guidance-strength", type=float, default=0.05)
    parser.add_argument("--guide-start", type=int, default=0)
    parser.add_argument("--guide-end", type=int, default=None, help="default: full_steps // 2")
    parser.add_argument("--no-phaselock", action="store_true", help="plain baseline sampling")
    parser.add_argument("--save-few", default=None, help="also write the few-step result here")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    set_seed(args.seed)

    entry = get_entry(args.backend)
    spec = get_spec(args.backend)
    if entry.mode == "i2v" and not args.image:
        raise SystemExit(f"--image is required for the image-to-video backend {args.backend!r}")
    if not entry.validated:
        logger.warning(
            "%s is registered but has not been validated end to end (it does not fit the "
            "development GPU); layout and normalisation are implemented, results are not verified",
            args.backend,
        )

    fps = args.fps or spec.default_fps
    logger.info(
        "%s: %d frames at %dx%d, %d fps",
        args.backend,
        args.num_frames or spec.default_num_frames,
        args.height or spec.default_height,
        args.width or spec.default_width,
        fps,
    )

    backend = load_backend(
        args.backend,
        model_id=args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        enable_offload=True,
    )
    image = load_image(args.image) if args.image else None

    call = dict(
        prompt=args.prompt,
        image=image,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    )

    if args.no_phaselock:
        logger.info("baseline: %d steps, no guidance", args.full_steps)
        frames = backend.pipe(
            **backend.generation_kwargs(**call),
            num_inference_steps=args.full_steps,
            generator=torch.Generator(device=backend.device).manual_seed(args.seed),
        ).frames[0]
        few_frames = None
    else:
        logger.info(
            "phaselock: %d-step prior -> %d steps, lambda_0=%.3f over [%d, %s)",
            args.few_steps, args.full_steps, args.guidance_strength,
            args.guide_start, args.guide_end or args.full_steps // 2,
        )
        pipeline = PhaseLockPipeline(
            backend,
            few_steps=args.few_steps,
            full_steps=args.full_steps,
            guidance_strength=args.guidance_strength,
            guide_start=args.guide_start,
            guide_end=args.guide_end,
        )
        frames, few_frames = pipeline(**call, seed=args.seed, return_few_result=True)

    export_to_video(frames, args.output, fps=fps)
    logger.info("wrote %s", args.output)
    if args.save_few and few_frames is not None:
        export_to_video(few_frames, args.save_few, fps=fps)
        logger.info("wrote %s", args.save_few)


if __name__ == "__main__":
    main()
