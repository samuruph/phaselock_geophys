"""Stage 6: baseline I2V continuation of LikePhys valid clips.

Ground truth stays known throughout. Conditioning on a *valid* clip's first frame means
the real continuation is a physically plausible reference, so a generation can be scored
two independent ways:

1. **Fidelity** to that real continuation, via the Physics-IQ motion-mask family. LikePhys
   is rendered with a static camera, so the protocol's assumptions hold directly.
2. **The Stage 5 detector** applied to the generation itself, whose accuracy is already
   established on labelled pairs.

Having both matters: (1) needs no assumption that geometry measures physics, so it
arbitrates when the two disagree.

Sampling is plain baseline throughout. PhaseLock guidance is never applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from ..backends.base import VideoBackend
from ..config import Config
from ..datasets import VideoSample, load_video, save_video
from ..datasets.likephys import LikePhys
from ..metrics.motion_mask import MotionMaskScores, motion_mask_scores
from ..analysis import video
from ..pipelines.generation import generate_with_probes
from ..probes.trajectory import ProbeRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One generated continuation and how it scored."""

    sample_id: str
    scenario: str
    seed: int
    num_steps: int
    blur_sigma: float
    scores: MotionMaskScores
    record: Optional[ProbeRecord] = None
    path: Optional[str] = None

    def flatten(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scenario": self.scenario,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "blur_sigma": self.blur_sigma,
            "spatial_iou": self.scores.spatial_iou,
            "spatiotemporal_iou": self.scores.spatiotemporal_iou,
            "weighted_spatial_iou": self.scores.weighted_spatial_iou,
            "mse": self.scores.mse,
            "raw_score": self.scores.raw_score,
            "path": self.path or "",
        }


def frames_to_tensor(frames: Any) -> torch.Tensor:
    """Pipeline output (PIL list or array) -> ``(F, 3, H, W)`` in [0, 1]."""
    if isinstance(frames, torch.Tensor):
        return frames
    import numpy as np
    import torchvision.transforms.functional as TF

    if isinstance(frames, np.ndarray):
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float().clamp(0, 1)
    return torch.stack([TF.to_tensor(frame) for frame in frames])


def reference_continuation(
    sample: VideoSample, backend: VideoBackend, blur_sigma: float = 0.0
) -> torch.Tensor:
    """The real clip, resampled to the geometry the generation will have.

    Both arms must pass through identical preprocessing, otherwise the motion-mask
    comparison measures the resampling as much as the physics.
    """
    spec = backend.spec
    return load_video(
        sample.path,
        num_frames=spec.default_num_frames,
        height=spec.default_height,
        width=spec.default_width,
        blur_sigma=blur_sigma,
    )


def generate_candidate(
    backend: VideoBackend,
    sample: VideoSample,
    dataset: LikePhys,
    config: Config,
    seed: int,
    num_steps: Optional[int] = None,
    blur_sigma: float = 0.0,
    video_dir: Optional[Path] = None,
) -> Candidate:
    """Condition on the clip's first frame, generate, and score against the real rest."""
    from PIL import Image

    scenario = sample.meta["scenario"]
    reference = reference_continuation(sample, backend, blur_sigma=0.0)
    first_frame = Image.fromarray(
        (reference[0].permute(1, 2, 0) * 255).byte().cpu().numpy()
    )

    steps = num_steps if num_steps is not None else config.generation.num_steps
    result = generate_with_probes(
        backend,
        prompt=dataset.prompt_for(scenario),
        image=first_frame,
        num_steps=steps,
        record_steps=config.probe.record_steps,
        sources=config.probe.sources,
        blocks=config.probe.blocks,
        block_stride=config.probe.block_stride,
        pooling=config.probe.pooling,
        guidance_scale=config.generation.guidance_scale,
        negative_prompt=config.generation.negative_prompt,
        seed=seed,
        provenance={
            "sample_id": sample.sample_id,
            "scenario": scenario,
            "dataset": dataset.name,
            "blur_sigma": blur_sigma,
        },
    )

    generated = frames_to_tensor(result.frames)
    # Blur is applied to both arms identically, per PhaseLock's control.
    if blur_sigma > 0:
        from ..datasets.video_io import gaussian_blur

        generated = gaussian_blur(generated, blur_sigma)
        reference = gaussian_blur(reference, blur_sigma)

    path = None
    if video_dir is not None:
        # Named `_generation.mp4` to sit alongside detection's `_pair` / `_inversion` /
        # `_roundtrip`, so a videos/ directory reads the same whichever stage produced it,
        # with the run's model and dataset already carried by the parent path.
        stem = f"{sample.sample_id.replace('/', '_')}_k{steps}_s{seed}"
        # Side by side against the real continuation rather than alone. A generated clip
        # on its own cannot be judged -- the question is always whether its *motion*
        # matches what actually happened from the same first frame, and both arms start
        # from that frame, so column 0 should agree and any later divergence is the
        # model's own dynamics.
        path = str(video.write_grid(
            {"real (reference)": reference, f"generated  k={steps}": generated},
            Path(video_dir) / f"{stem}_generation.mp4",
            fps=backend.spec.default_fps, columns=2,
        ))
        save_video(generated, str(Path(video_dir) / f"{stem}_raw.mp4"),
                   fps=backend.spec.default_fps)

    return Candidate(
        sample_id=sample.sample_id,
        scenario=scenario,
        seed=seed,
        num_steps=steps,
        blur_sigma=blur_sigma,
        scores=motion_mask_scores(generated, reference),
        record=result.record,
        path=path,
    )


def best_of_n(
    candidates: Sequence[Candidate], scores: Sequence[float]
) -> tuple[Candidate, Candidate, Candidate]:
    """Select by a verifier score and report what selection was worth.

    Args:
        candidates: Generations for one conditioning setup.
        scores: The verifier's plausibility score per candidate, lower meaning more
            plausible (every GeoPhys statistic is oriented that way).

    Returns:
        ``(selected, baseline, oracle)`` -- the verifier's pick, the first candidate as
        the no-verifier control, and the best achievable pick. The gap between baseline
        and oracle is the headroom; the gap between selected and baseline is the gain.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    if len(scores) != len(candidates):
        raise ValueError(f"got {len(candidates)} candidates but {len(scores)} scores")

    selected = candidates[min(range(len(scores)), key=scores.__getitem__)]
    oracle = max(candidates, key=lambda candidate: candidate.scores.raw_score)
    return selected, candidates[0], oracle
