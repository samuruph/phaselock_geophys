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


def across_depth(
    rows: Sequence[Mapping[str, str]],
    sample_id: str,
    source: str = "hidden_states",
) -> dict[str, list[tuple[int, float, float]]]:
    """`{statistic: [(step, mean, std), ...]}`, aggregated over every recorded block.

    One clip gives 30 or 42 values per statistic per step, one per DiT block, and picking
    a single block to plot throws all but one away and invites cherry-picking. Summarising
    depth as a mean and a spread keeps the whole probe grid in one panel: the line is what
    the network says on average at that step, the band is how much depth disagrees.

    A wide band is informative in its own right -- it means the statistic reads
    differently at different depths, which is exactly the case for ``accel``, where the
    n=96 run found a coherent band across blocks 3-22 and noise outside it.
    """
    buckets: dict[str, dict[int, list[float]]] = {name: {} for name in STATISTICS}
    for row in rows:
        if row["sample_id"] != sample_id or row["source"] != source:
            continue
        step = int(row["step"])
        for name in STATISTICS:
            value = row.get(f"phi_{name}")
            if value not in (None, ""):
                buckets[name].setdefault(step, []).append(float(value))

    out: dict[str, list[tuple[int, float, float]]] = {}
    for name, by_step in buckets.items():
        if not by_step:
            continue
        out[name] = [
            (step, float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0)
            for step, v in sorted(by_step.items())
        ]
    return out


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


def animated_strips(
    plausible: Mapping[str, list[tuple[int, float, float]]],
    violated: Mapping[str, list[tuple[int, float, float]]],
    width_px: int,
    frames: int,
    taus: Optional[Mapping[int, float]] = None,
    caption: str = "",
    dpi: int = 100,
) -> list[np.ndarray]:
    """One strip per video frame, the curves drawing in as the clip plays.

    Each statistic is a line for the mean over DiT blocks and a band for +/-1 s.d. across
    them, so the whole probe grid is in one panel rather than one arbitrary block.

    The x-axis is the denoising step, which does *not* advance with video time -- there is
    no per-video-frame value on disk to plot. Progressive reveal is therefore a reading
    aid, not a claim of correspondence: it walks the trajectory from clean to noise over
    the clip's duration so the shape can be followed, and the axes are fixed throughout so
    nothing appears to move that is not moving.
    """
    import matplotlib.pyplot as plt

    palette.apply_style()
    names = [n for n in STATISTICS if n in plausible or n in violated]
    if not names:
        raise ValueError("no statistics to draw")
    if frames < 1:
        raise ValueError(f"frames must be positive, got {frames}")

    steps = sorted({s for series in plausible.values() for s, _, _ in series})
    ticks = {s: (round((1.0 - taus[s]) * 1000) if taus else s) for s in steps}

    # Fixed limits, computed once over everything that will ever be drawn. Autoscaling
    # per frame would make a static curve appear to writhe.
    limits = {}
    for name in names:
        values = [
            v + sign * d
            for series in (plausible.get(name, []), violated.get(name, []))
            for _, v, d in series
            for sign in (-1, 1)
        ]
        low, high = min(values), max(values)
        margin = 0.08 * (high - low) or max(abs(high), 1.0) * 0.08
        limits[name] = (low - margin, high + margin)

    figure, axes = plt.subplots(
        1, len(names), figsize=(width_px / dpi, STRIP_HEIGHT), dpi=dpi, squeeze=False,
    )
    # The reveal has at most one distinct state per denoising step -- ten, against
    # eighty-one video frames -- so render each once and hand back references. Drawing
    # per frame would be eight times the matplotlib work for identical pixels.
    rendered: dict[int, np.ndarray] = {}
    out: list[np.ndarray] = []
    for index in range(frames):
        # How much of the trajectory to reveal, always ending on the complete curve.
        shown = max(2, int(round((index + 1) / frames * len(steps))))
        if shown in rendered:
            out.append(rendered[shown])
            continue
        for axis, name in zip(axes[0], names):
            axis.clear()
            for series, colour, label in (
                (plausible.get(name), palette.CATEGORICAL[0], "plausible"),
                (violated.get(name), palette.CATEGORICAL[7], "violated"),
            ):
                if not series:
                    continue
                head = series[:shown]
                xs = [ticks[s] for s, _, _ in head]
                means = np.array([m for _, m, _ in head])
                spread = np.array([d for _, _, d in head])
                axis.fill_between(xs, means - spread, means + spread, color=colour,
                                  alpha=0.18, linewidth=0, zorder=1)
                axis.plot(xs, means, color=colour, linewidth=1.7, zorder=3, label=label)
                axis.plot(xs[-1:], means[-1:], marker="o", markersize=4.5, color=colour,
                          zorder=4)

            axis.set_xlim(min(ticks.values()), max(ticks.values()))
            axis.set_ylim(*limits[name])
            axis.set_title(palette.statistic_label(name), fontsize=7.5, pad=3)
            axis.set_xlabel("denoising timestep", fontsize=6.5, labelpad=1)
            axis.tick_params(labelsize=6)
        axes[0][0].legend(fontsize=6.5, loc="best")
        if caption:
            palette.caption(figure, caption)
        figure.tight_layout(rect=(0, 0.10 if caption else 0, 1, 1))
        figure.canvas.draw()
        rendered[shown] = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        out.append(rendered[shown])

    plt.close(figure)
    return out


def attach_animated(
    source: Path, destination: Path, strips: Sequence[np.ndarray], fps: int = 12
) -> Path:
    """Re-encode ``source`` with a per-frame strip beneath each frame."""
    import cv2

    from .video import encode_frames

    frames = _read(source)
    width = frames[0].shape[1]

    def composite():
        for index, frame in enumerate(frames):
            strip = strips[min(index, len(strips) - 1)]
            if strip.shape[1] != width:
                scale = width / strip.shape[1]
                strip = cv2.resize(strip, (width, max(1, int(round(strip.shape[0] * scale)))))
            pad = (frame.shape[0] + strip.shape[0]) % 2
            if pad:
                strip = np.pad(strip, ((0, pad), (0, 0), (0, 0)), constant_values=255)
            yield np.concatenate([frame, strip], axis=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    encode_frames(composite(), destination, fps)
    return destination


def _read(source: Path) -> list[np.ndarray]:
    """Decode an mp4 to a list of RGB frames."""
    import cv2

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
    return frames


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
