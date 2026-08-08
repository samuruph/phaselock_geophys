"""Video decoding and the preprocessing every experiment shares.

Source clips and model input rarely agree. LikePhys is 60 frames of 512x512 at 30 fps;
IntPhys2 is 636 frames of 512x512 at 60 fps; CogVideoX wants 49 frames at 480x720.
Everything here is deterministic, so two runs over the same clip produce identical input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def decode_video(path: str, max_frames: Optional[int] = None) -> torch.Tensor:
    """Decode a video to ``(F, 3, H, W)`` float32 in [0, 1], RGB."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    frames = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"decoded zero frames from {path}")
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def temporal_window(frames: torch.Tensor, window: Optional[float]) -> torch.Tensor:
    """Keep a centred fraction of the clip, or all of it when ``window`` is None.

    IntPhys2 clips run 10.6 s, so resampling the whole thing down to 49 frames is a 13x
    decimation that can step straight over a brief violation. Narrowing the window
    trades coverage for temporal resolution.
    """
    if window is None or window >= 1.0:
        return frames
    if not 0.0 < window < 1.0:
        raise ValueError(f"window must be in (0, 1] or None, got {window}")
    count = frames.shape[0]
    keep = max(2, int(round(count * window)))
    start = (count - keep) // 2
    return frames[start : start + keep]


def resample_frames(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Uniformly resample along time to exactly ``num_frames`` by nearest index.

    Nearest-index rather than interpolation: blending neighbouring frames would
    manufacture motion blur that the geometric statistics would read as smoother
    dynamics, biasing exactly the quantity under study.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    count = frames.shape[0]
    if count == num_frames:
        return frames
    index = torch.linspace(0, count - 1, num_frames).round().long().clamp(0, count - 1)
    return frames[index]


def letterbox(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize preserving aspect ratio, then pad to exactly ``(height, width)``.

    Stretching a 512x512 source into 480x720 would change every apparent velocity in the
    scene by a different factor along each axis. Padding is identical for both members
    of a matched pair, so it cannot affect a within-pair comparison.
    """
    _, _, source_height, source_width = frames.shape
    scale = min(height / source_height, width / source_width)
    new_height = max(1, int(round(source_height * scale)))
    new_width = max(1, int(round(source_width * scale)))

    resized = F.interpolate(
        frames, size=(new_height, new_width), mode="bilinear", align_corners=False, antialias=True
    )
    pad_top = (height - new_height) // 2
    pad_left = (width - new_width) // 2
    return F.pad(
        resized,
        (pad_left, width - new_width - pad_left, pad_top, height - new_height - pad_top),
        value=0.0,
    )


def gaussian_blur(frames: torch.Tensor, sigma: float) -> torch.Tensor:
    """Isotropic Gaussian blur, applied per frame.

    This is PhaseLock's blur control (sigma in {0, 8, 16}). It must be applied to *every*
    arm of a comparison including the reference: blurring only the sharper arm tests a
    different hypothesis. The kernel is truncated at 3 sigma, the usual convention.
    """
    if sigma <= 0:
        return frames
    radius = max(1, int(round(3.0 * sigma)))
    size = 2 * radius + 1

    offsets = torch.arange(size, dtype=frames.dtype, device=frames.device) - radius
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()

    channels = frames.shape[1]
    blurred = F.conv2d(
        F.pad(frames, (radius, radius, 0, 0), mode="reflect"),
        kernel.view(1, 1, 1, size).expand(channels, 1, 1, size),
        groups=channels,
    )
    return F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"),
        kernel.view(1, 1, size, 1).expand(channels, 1, size, 1),
        groups=channels,
    )


def load_video(
    path: str,
    num_frames: int,
    height: int,
    width: int,
    window: Optional[float] = None,
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """Decode and preprocess one clip into model-ready ``(F, 3, H, W)`` in [0, 1].

    Order matters: window and resample before blurring, so that ``blur_sigma`` means the
    same thing in output pixels regardless of the source resolution.
    """
    frames = decode_video(path)
    frames = temporal_window(frames, window)
    frames = resample_frames(frames, num_frames)
    frames = letterbox(frames, height, width)
    return gaussian_blur(frames, blur_sigma)


def save_video(frames: torch.Tensor, path: str, fps: int = 8) -> None:
    """Write ``(F, 3, H, W)`` in [0, 1] to an H.264 mp4.

    H.264 rather than OpenCV's default mp4v, which is MPEG-4 Part 2 and cannot be decoded
    by Chromium -- so those files play in VLC but render as green mush in a VS Code tab or
    a browser, which is where they actually get looked at. Shares one encoder with the
    analysis videos so there is a single place this can be got wrong.
    """
    from ..analysis.video import encode_frames

    array = (frames.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    encode_frames(array, Path(path), fps)
