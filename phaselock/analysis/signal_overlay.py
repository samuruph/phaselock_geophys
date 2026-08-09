"""Draw the computed signals underneath the videos they were computed from.

A number in `statistics.csv` and a clip in `videos/` describe the same thing and are
impossible to hold side by side. This composites one onto the other: the video on top,
and beneath it the five statistics for that exact clip, plausible against violated.

**Everything here is post-processing.** It reads `statistics.csv` and the mp4s that a run
already produced and re-encodes them, so it costs seconds of CPU and never touches the
GPU. Nothing is recomputed.

That constraint decides what can be drawn, and it is worth being explicit about the two
things that cannot:

*No spatial map.* Every trajectory is spatially mean-pooled inside the forward hook -- a
`(T, tokens, D)` activation becomes `(T, D)` before any statistic exists. Where in the
frame a violation happened is destroyed at that point, by design, because GeoPhys is
defined on a pooled per-frame trajectory. Recovering it would need `pooling="flatten"` and
a re-run.

*No per-video-frame curve.* `statistics.csv` stores the temporal *summaries* -- a mean or
a standard deviation over the clip -- not the per-frame intermediates they summarise. So
the x-axis here is the **denoising step**, not the video frame. Drawing `s_t` against
video time would need `probe.save_trajectories = true` and a re-run.

What remains is still the useful comparison: how each statistic evolves as the trajectory
is walked from clean toward noise, and where the violated clip separates from the
plausible one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from . import palette
from ..metrics.geophys import STATISTICS

STRIP_HEIGHT = 2.05
"""Inches. Tall enough for five readable panels, short enough not to dwarf the video."""


def per_step(
    rows: Sequence[Mapping[str, str]],
    sample_id: str,
    source: str,
    block: int,
) -> dict[str, list[tuple[int, float]]]:
    """`{statistic: [(step, value), ...]}` for one clip at one probe location.

    Reads the rows of `statistics.csv` directly, so this stays usable for any run without
    re-deriving anything.
    """
    out: dict[str, list[tuple[int, float]]] = {name: [] for name in STATISTICS}
    for row in rows:
        if row["sample_id"] != sample_id or row["source"] != source:
            continue
        if int(row["block"]) != block:
            continue
        for name in STATISTICS:
            value = row.get(f"phi_{name}")
            if value not in (None, ""):
                out[name].append((int(row["step"]), float(value)))
    return {name: sorted(points) for name, points in out.items() if points}


def choose_location(rows: Sequence[Mapping[str, str]], preferred: Optional[str] = None) -> tuple[str, int]:
    """`(source, block)` to draw, defaulting to the deepest-covered hidden-state block.

    ``preferred`` accepts ``"source"`` or ``"source/b12"``. Without it, the middle hidden
    state block is used -- a single arbitrary-but-stable choice, since a strip can only
    show one location and the alternative is picking silently per clip.
    """
    if preferred:
        source, _, block = preferred.partition("/b")
        if block:
            return source, int(block)
        available = sorted({int(r["block"]) for r in rows if r["source"] == source})
        return source, (available[len(available) // 2] if available else -1)

    blocks = sorted({int(r["block"]) for r in rows if r["source"] == "hidden_states"})
    if blocks:
        return "hidden_states", blocks[len(blocks) // 2]
    return (rows[0]["source"], int(rows[0]["block"])) if rows else ("latent", -1)


def signal_strip(
    plausible: Mapping[str, list[tuple[int, float]]],
    violated: Mapping[str, list[tuple[int, float]]],
    width_px: int,
    taus: Optional[Mapping[int, float]] = None,
    caption: str = "",
    dpi: int = 100,
) -> np.ndarray:
    """Five side-by-side panels, one per statistic, plausible vs violated.

    Each panel is independently scaled: the statistics differ by orders of magnitude
    (`phi_accel` is a squared norm, `phi_curv` an angle in radians), and a shared axis
    would flatten four of them into a line. What matters is the *gap* within a panel, not
    the height across panels.
    """
    import matplotlib.pyplot as plt

    palette.apply_style()
    names = [n for n in STATISTICS if n in plausible or n in violated]
    if not names:
        raise ValueError("no statistics to draw")

    figure, axes = plt.subplots(
        1, len(names), figsize=(width_px / dpi, STRIP_HEIGHT), dpi=dpi, squeeze=False,
    )
    for axis, name in zip(axes[0], names):
        drawn = {}
        for key, series, colour, label in (
            ("plausible", plausible.get(name), palette.CATEGORICAL[0], "plausible"),
            ("violated", violated.get(name), palette.CATEGORICAL[7], "violated"),
        ):
            if not series:
                continue
            steps = [s for s, _ in series]
            values = [v for _, v in series]
            ticks = [round((1.0 - taus[s]) * 1000) for s in steps] if taus else steps
            drawn[key] = (ticks, values)
            axis.plot(ticks, values, color=colour, linewidth=1.6, marker="o",
                      markersize=2.6, label=label, zorder=3)

        # Shade the gap, and shade it by whether the gap points the right way. Every
        # statistic is oriented so larger means less regular, so violated *above*
        # plausible is the signal calling this pair correctly at that step. Two curves
        # that cross are the interesting case and are invisible without this.
        if len(drawn) == 2 and drawn["plausible"][0] == drawn["violated"][0]:
            ticks = drawn["plausible"][0]
            low, high = np.array(drawn["plausible"][1]), np.array(drawn["violated"][1])
            axis.fill_between(ticks, low, high, where=high >= low, interpolate=True,
                              color=palette.DIVERGING_HIGH, alpha=0.16, zorder=1)
            axis.fill_between(ticks, low, high, where=high < low, interpolate=True,
                              color=palette.DIVERGING_LOW, alpha=0.16, zorder=1)
            correct = int((high > low).sum())
            axis.set_xlabel(f"correct at {correct}/{len(ticks)} steps",
                            fontsize=6.5, labelpad=1)
        else:
            axis.set_xlabel("denoising timestep", fontsize=6.5, labelpad=1)

        axis.set_title(palette.statistic_label(name), fontsize=7.5, pad=3)
        axis.tick_params(labelsize=6)

    axes[0][0].legend(fontsize=6.5, loc="best")
    if caption:
        palette.caption(figure, caption)
    figure.tight_layout(rect=(0, 0.10 if caption else 0, 1, 1))

    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return image


def attach(source: Path, destination: Path, strip: np.ndarray, fps: int = 12) -> Path:
    """Re-encode ``source`` with ``strip`` fixed beneath every frame."""
    import cv2

    from .video import encode_frames

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open {source}")
    try:
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"decoded zero frames from {source}")

    width = frames[0].shape[1]
    if strip.shape[1] != width:
        scale = width / strip.shape[1]
        strip = cv2.resize(strip, (width, max(1, int(round(strip.shape[0] * scale)))))
    # The composite must stay even for H.264.
    pad = (frames[0].shape[0] + strip.shape[0]) % 2
    if pad:
        strip = np.pad(strip, ((0, pad), (0, 0), (0, 0)), constant_values=255)

    destination.parent.mkdir(parents=True, exist_ok=True)
    encode_frames((np.concatenate([f, strip], axis=0) for f in frames), destination, fps)
    return destination
