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
_GUTTER = 4

_MIN_SCALE = 0.42
"""Hard floor. Below this the caption is unreadable once the tiled frame is shrunk to
fit a window, which is how these are actually viewed."""

_MAX_SCALE = 0.95
"""Hard ceiling, independent of panel size. Captions are labels, not headings."""


def _band_height(width: int) -> int:
    return max(30, min(46, int(round(width * 0.062))))


def _fit_scale(texts: Sequence[str], width: int, thickness: int = 1) -> float:
    """Largest scale at which every caption still fits the panel width.

    Sizing from the panel width alone is not enough: a violation name like
    `invalid_momentum_amplification` is four times longer than `plausible`, so a scale
    that fits one runs off the edge for the other. Measure the actual strings and shrink
    to whichever needs it most, so a caption is never clipped whatever it says.
    """
    usable = width - 2 * _pad(width)
    scale = _MAX_SCALE
    for text in texts:
        if not text:
            continue
        measured = cv2.getTextSize(text, _FONT, _MAX_SCALE, thickness)[0][0]
        if measured > usable:
            scale = min(scale, _MAX_SCALE * usable / measured)
    return max(_MIN_SCALE, scale)


def _pad(width: int) -> int:
    return max(6, width // 70)


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    """``(F, 3, H, W)`` in [0, 1] -> ``(F, H, W, 3)`` uint8 RGB."""
    return (frames.detach().float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def _label(panel: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Put a caption band above a panel rather than over the image.

    Text burned onto the video would sit on top of exactly the pixels being judged.
    """
    width = panel.shape[1]
    height = _band_height(width)
    # Title and subtitle share one scale, sized so the longer of them fits.
    scale = _fit_scale([text, sub], width)
    band = np.full((height, width, 3), 22, dtype=np.uint8)

    pad = _pad(width)
    baseline = int(height * 0.7)
    cv2.putText(band, text, (pad, baseline), _FONT, scale, (245, 245, 245), 1, cv2.LINE_AA)
    if sub:
        title_end = cv2.getTextSize(text, _FONT, scale, 1)[0][0] + pad
        size = cv2.getTextSize(sub, _FONT, scale, 1)[0]
        # Drop the subtitle rather than let it overlap the title.
        if title_end + size[0] + 2 * pad <= width:
            cv2.putText(band, sub, (width - size[0] - pad, baseline), _FONT, scale,
                        (155, 155, 155), 1, cv2.LINE_AA)
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


def _prompt_bar(prompt: str, width: int) -> np.ndarray:
    """Render the full conditioning prompt in a clear, wrapped header bar."""
    text = prompt.strip() or "(empty prompt)"
    label = "PROMPT"
    font_scale = 0.48
    thickness = 1
    pad = max(10, width // 80)
    line_height = 21
    available = max(1, width - 2 * pad)

    def wrap(value: str) -> list[str]:
        words, lines, current = value.split(), [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and cv2.getTextSize(candidate, _FONT, font_scale, thickness)[0][0] > available:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        # Split pathological long tokens so even URLs or unbroken identifiers remain visible.
        expanded = []
        for line in lines:
            while cv2.getTextSize(line, _FONT, font_scale, thickness)[0][0] > available:
                cut = max(1, int(len(line) * available / cv2.getTextSize(line, _FONT, font_scale, thickness)[0][0]))
                expanded.append(line[:cut])
                line = line[cut:]
            expanded.append(line)
        return expanded or [""]

    lines = wrap(text)
    height = pad * 2 + line_height * (len(lines) + 1)
    bar = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(bar, label, (pad, pad + 13), _FONT, 0.38, (120, 205, 255), 1, cv2.LINE_AA)
    for index, line in enumerate(lines):
        y = pad + line_height * (index + 2)
        cv2.putText(bar, line, (pad, y), _FONT, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
    return bar


def write_grid(
    clips: Mapping[str, torch.Tensor],
    path: Path,
    fps: int = 12,
    columns: Optional[int] = None,
    subtitles: Optional[Mapping[str, str]] = None,
    prompt: Optional[str] = None,
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
    frames = (
        (lambda tiled: np.concatenate([_prompt_bar(prompt, tiled.shape[1]), tiled], axis=0)
         if prompt is not None else tiled)(_tile(
            [
                _label(array[min(index, array.shape[0] - 1)], name, subtitles.get(name, ""))
                for name, array in arrays.items()
            ],
            columns,
        ))
        for index in range(length)
    )
    encode_frames(frames, path, fps)
    return path


def prompted_video(frames: torch.Tensor, path: Path, prompt: str, fps: int = 8) -> Path:
    """Save a single clip with its prompt displayed above every frame."""
    array = _to_uint8(frames)
    header = _prompt_bar(prompt, array.shape[2])
    encode_frames((np.concatenate([header, frame], axis=0) for frame in array), path, fps)
    return path


def encode_frames(frames, path: Path, fps: int) -> None:
    """Write H.264, falling back to OpenCV's MPEG-4 only if ffmpeg is missing.

    The codec is not cosmetic. OpenCV's default ``mp4v`` is MPEG-4 Part 2, which
    Chromium cannot decode, so the file plays in VLC but shows up blank in a VS Code
    tab, a browser, or anything else Electron-based -- the exact places these get
    looked at. imageio-ffmpeg ships a static ffmpeg binary, so H.264 needs no system
    package.
    """
    # yuv420p requires even width and height.  Prompt bars and annotation strips can
    # turn an otherwise even model frame into an odd-sized video (for example 480 +
    # 125 = 605), so pad only the bottom/right edge when necessary.  Edge padding does
    # not alter the rendered content and keeps every caller safe, including generators
    # that produce frames lazily.
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"no frames to write to {path}") from None

    def even_frame(frame):
        height, width = frame.shape[:2]
        pad_height, pad_width = height % 2, width % 2
        if not pad_height and not pad_width:
            return frame
        return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")

    first = even_frame(first)

    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover - exercised only where ffmpeg is absent
        imageio = None

    if imageio is not None:
        writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=8,
            pixelformat="yuv420p",  # required for browser playback
            macro_block_size=1,  # dimensions are already even; do not pad them again
        )
        try:
            writer.append_data(first)
            for frame in iterator:
                writer.append_data(even_frame(frame))
        finally:
            writer.close()
        return

    writer = None  # pragma: no cover - fallback path
    try:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (first.shape[1], first.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open a writer for {path}")
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for frame in iterator:
            writer.write(cv2.cvtColor(even_frame(frame), cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()


def pair_video(
    plausible: torch.Tensor,
    violated: torch.Tensor,
    path: Path,
    scenario: str = "",
    violation: str = "",
    fps: int = 12,
    prompt: Optional[str] = None,
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
        prompt=prompt,
    )


def inversion_video(
    steps: Sequence[tuple[float, torch.Tensor]],
    path: Path,
    original: Optional[torch.Tensor] = None,
    kind: str = "x0_hat",
    fps: int = 12,
    columns: Optional[int] = None,
    prompt: Optional[str] = None,
) -> Path:
    """One panel per recorded inversion step, all playing together.

    ``kind="x0_hat"`` is the informative one. Decoding the noisy latent itself mostly
    shows noise, and shows it at every step; the model's clean estimate answers the
    question that matters for this project -- *what does the model still know about this
    video at this noise level* -- and degrades visibly as the trajectory walks toward
    noise. Where it stops tracking the real motion is where the physical content of the
    trajectory has been destroyed.

    ``steps`` is ``(tau, frames)`` per recorded point, on the pipeline's shared
    coordinate ``tau = 1 - timestep/1000``: tau = 1 is clean data, tau = 0 is pure noise.
    Panels are captioned with the diffusion timestep itself rather than a step counter,
    so a panel can be matched against a row of ``statistics.csv``.
    """
    clips: dict[str, torch.Tensor] = {}
    if original is not None:
        clips["original"] = original
    label = lambda tau: f"t={round((1.0 - tau) * 1000)}"
    for tau, frames in steps:
        clips[label(tau)] = frames
    return write_grid(
        clips, path, fps=fps, columns=columns,
        subtitles={label(tau): kind for tau, _ in steps},
        prompt=prompt,
    )
