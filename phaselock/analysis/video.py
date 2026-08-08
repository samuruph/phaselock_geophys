"""Side-by-side mp4s, for looking at motion rather than at sampled stills.

Contact sheets answer "is the content right?" but they cannot answer "is the *motion*
right?", and motion is the entire subject of this project. Eight stills from an 81-frame
clip step straight over a jitter violation, and they show a denoising trajectory as a row
of nearly identical thumbnails. Played as video the same data reads immediately.

Every writer here tiles clips into one frame and writes a single mp4, so the panels are
locked to the same clock: a difference between panels is a real difference at that
instant, not an artifact of scrubbing two players by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

# Drawn in pixel space on the tiled frame, so they stay legible whatever the source size.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_BAND = 26
_GUTTER = 3


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    """``(F, 3, H, W)`` in [0, 1] -> ``(F, H, W, 3)`` uint8 RGB."""
    return (frames.detach().float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def _label(panel: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Put a caption band above a panel rather than over the image.

    Text burned onto the video would sit on top of exactly the pixels being judged.
    """
    height, width = panel.shape[:2]
    band = np.full((_LABEL_BAND, width, 3), 22, dtype=np.uint8)
    scale = max(0.34, min(0.52, width / 620))
    cv2.putText(band, text, (6, 17), _FONT, scale, (245, 245, 245), 1, cv2.LINE_AA)
    if sub:
        size = cv2.getTextSize(sub, _FONT, scale * 0.82, 1)[0]
        cv2.putText(band, sub, (width - size[0] - 6, 17), _FONT, scale * 0.82,
                    (150, 150, 150), 1, cv2.LINE_AA)
    return np.concatenate([band, panel], axis=0)


def _tile(panels: Sequence[np.ndarray], columns: int) -> np.ndarray:
    """Lay panels out row-major, padding a short last row with black."""
    rows = []
    for start in range(0, len(panels), columns):
        row = list(panels[start : start + columns])
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        gutter = np.zeros((row[0].shape[0], _GUTTER, 3), dtype=np.uint8)
        stitched = row[0]
        for panel in row[1:]:
            stitched = np.concatenate([stitched, gutter, panel], axis=1)
        rows.append(stitched)
    gutter = np.zeros((_GUTTER, rows[0].shape[1], 3), dtype=np.uint8)
    out = rows[0]
    for row in rows[1:]:
        out = np.concatenate([out, gutter, row], axis=0)

    # mp4v silently crops odd dimensions, which would shave a column or row off the last
    # panel. Pad instead, so nothing that was drawn is lost.
    pad_h, pad_w = out.shape[0] % 2, out.shape[1] % 2
    if pad_h or pad_w:
        out = np.pad(out, ((0, pad_h), (0, pad_w), (0, 0)))
    return out


def write_grid(
    clips: Mapping[str, torch.Tensor],
    path: Path,
    fps: int = 12,
    columns: Optional[int] = None,
    subtitles: Optional[Mapping[str, str]] = None,
) -> Path:
    """Tile named clips into one mp4, all playing on the same clock.

    Clips shorter than the longest hold their final frame rather than going black, so a
    panel never disappears mid-playback and the comparison stays readable to the end.
    """
    if not clips:
        raise ValueError("no clips to write")

    arrays = {name: _to_uint8(frames) for name, frames in clips.items()}
    shapes = {a.shape[1:3] for a in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"all clips must share a frame size, got {shapes}")

    length = max(a.shape[0] for a in arrays.values())
    columns = columns or len(arrays)
    subtitles = subtitles or {}

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for index in range(length):
            panels = [
                _label(array[min(index, array.shape[0] - 1)], name, subtitles.get(name, ""))
                for name, array in arrays.items()
            ]
            frame = _tile(panels, columns)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open a writer for {path}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    return path


def pair_video(
    plausible: torch.Tensor,
    violated: torch.Tensor,
    path: Path,
    scenario: str = "",
    violation: str = "",
    fps: int = 12,
) -> Path:
    """The matched pair side by side, on one clock.

    This is the check no still can make: whether the violation is actually present in the
    preprocessed tensor the model sees. Several LikePhys violations are purely temporal --
    a freeze, a jitter, a shuffled segment -- and are invisible in sampled frames.
    """
    return write_grid(
        {
            "plausible": plausible,
            f"violated - {violation}" if violation else "violated": violated,
        },
        path, fps=fps, columns=2,
        subtitles={"plausible": scenario} if scenario else None,
    )


def inversion_video(
    steps: Sequence[tuple[float, torch.Tensor]],
    path: Path,
    original: Optional[torch.Tensor] = None,
    kind: str = "x0_hat",
    fps: int = 12,
    columns: Optional[int] = None,
) -> Path:
    """One panel per recorded inversion step, all playing together.

    ``kind="x0_hat"`` is the informative one. Decoding the noisy latent itself mostly
    shows noise, and shows it at every step; the model's clean estimate answers the
    question that matters for this project -- *what does the model still know about this
    video at this noise level* -- and degrades visibly as the trajectory walks toward
    noise. Where it stops tracking the real motion is where the physical content of the
    trajectory has been destroyed.

    ``steps`` is ``(tau, frames)`` per recorded point, tau = 0 clean and 1 pure noise.
    """
    clips: dict[str, torch.Tensor] = {}
    if original is not None:
        clips["original"] = original
    for tau, frames in steps:
        clips[f"t={round((1.0 - tau) * 1000)}"] = frames
    subtitles = {f"t={round((1.0 - tau) * 1000)}": f"{kind}  s={tau:.2f}" for tau, _ in steps}
    return write_grid(clips, path, fps=fps, columns=columns, subtitles=subtitles)
