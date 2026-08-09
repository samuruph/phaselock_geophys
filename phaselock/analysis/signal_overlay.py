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

*Video time needs the trajectories.* `statistics.csv` stores the temporal *summaries* --
a mean or a standard deviation over the clip -- and a mean cannot be inverted back into
the values it came from. So from the CSV alone the x-axis can only be the **denoising
step**. With `probe.save_trajectories = true` the full `(T, D)` arrays are on disk and
:func:`per_video_frame` recovers the per-frame intermediates, which is what
:func:`timeline_strips` plots against actual video time.

**The denoising axis runs `t = 0` (clean video) to `t = 1000` (pure noise).** Inversion
starts at the clean latent and walks up, so `t = 0` is where it begins. The recorded
points are not evenly spaced -- the sampler's own schedule is denser near the noise end --
which is why the plotted ticks read 0, 251, 459, 586, ... rather than 0, 100, 200.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from . import palette
from ..metrics.geophys import STATISTICS

STRIP_HEIGHT = 2.05
"""Inches. Tall enough for five readable panels, short enough not to dwarf the video."""


PER_FRAME = ("speed", "curv", "accel", "perr")
"""The four intermediates that exist per frame. `ang` has no per-frame form -- it is a
standard deviation over the whole clip, so there is nothing to plot against time."""


def per_video_frame(
    record,
    spec,
    source: str = "hidden_states",
    step: Optional[int] = None,
    order: int = 3,
    fit: str = "span",
) -> dict[str, np.ndarray]:
    """Per-*video-frame* intermediates, averaged over depth, from a saved trajectory.

    This is the only way to put video time on the x-axis. `statistics.csv` stores each
    clip's temporal summary -- a mean or standard deviation over its frames -- and a mean
    cannot be inverted back into the values it came from. The full `(T, D)` trajectories
    can, which is what `probe.save_trajectories` is for.

    Two expansions happen here:

    * **depth** -- every recorded block gives its own intermediate series, averaged into
      one, the same summary the animated strip uses.
    * **time** -- the intermediates are per *latent* frame, and the causal VAE folds four
      video frames into each latent after the first, so each value is held across the
      video frames it was computed from. The result is a step function at latent
      resolution, which is the honest rendering: it is 21 measurements stretched over 81
      frames, not 81 measurements.

    Differencing also shortens the series -- speed loses one frame, curvature and
    acceleration two, the residual `order` -- so each is right-aligned to the frames it
    actually describes and left-padded with NaN.
    """
    from ..backends.base import num_latent_frames
    from ..metrics.geophys import geophys_signals
    from .latent_motion import frames_for_latent

    blocks = [k.block for k in record.trajectories if k.source == source
              and (step is None or k.step == step)]
    if not blocks:
        return {}
    chosen_step = step if step is not None else sorted(record.steps)[0]

    collected: dict[str, list[np.ndarray]] = {name: [] for name in PER_FRAME}
    for block in sorted(set(blocks)):
        try:
            trajectory = record.get(source, chosen_step, block)
        except KeyError:
            continue
        signals = geophys_signals(trajectory, order=order, fit=fit)
        for name, series in (
            ("speed", signals.speed),
            ("curv", signals.turning_angle),
            ("accel", signals.acceleration),
            ("perr", signals.residual),
        ):
            collected[name].append(series.detach().float().cpu().numpy())

    latents = num_latent_frames(spec.default_num_frames, spec)
    out: dict[str, np.ndarray] = {}
    for name, stack in collected.items():
        if not stack:
            continue
        mean = np.mean(np.stack(stack), axis=0)
        expanded = _spread_over_span(mean, name, latents, spec, order)
        if expanded is not None:
            out[name] = expanded
    return out


def _latent_span(index: int, name: str, order: int) -> range:
    """Which latent frames an intermediate at ``index`` actually describes.

    Getting this wrong puts the curve in the wrong place, which matters most for exactly
    the question the timeline exists to answer -- whether the signal fires when the
    violation happens.

    Right-aligning them all, which is the obvious thing to do since differencing shortens
    the series, is wrong: it lags curvature and acceleration by a whole latent frame, four
    video frames, because both are second differences centred on the middle of the three
    latents they touch, not the last.

    * ``speed``   ``v_i = z_{i+1} - z_i``            spans latents ``i .. i+1``
    * ``curv``    angle between ``v_i`` and ``v_{i+1}``  spans ``i .. i+2``
    * ``accel``   ``v_{i+1} - v_i``                   spans ``i .. i+2``
    * ``perr``    predicts ``z_{i+order}`` from the ``order`` before it, so it is *about*
      that one frame
    """
    if name == "speed":
        return range(index, index + 2)
    if name in ("curv", "accel"):
        return range(index, index + 3)
    return range(index + order, index + order + 1)


def _spread_over_span(values, name, latents, spec, order):
    """Expand per-intermediate values onto video frames, averaging where spans overlap.

    Overlap is real -- consecutive accelerations share two of their three latents -- so
    averaging is the honest reduction rather than letting the last writer win.
    """
    from .latent_motion import frames_for_latent

    total = np.zeros(spec.default_num_frames, dtype=np.float64)
    count = np.zeros(spec.default_num_frames, dtype=np.float64)
    for index, value in enumerate(values):
        for latent in _latent_span(index, name, order):
            if latent >= latents:
                continue
            for frame in frames_for_latent(latent, spec):
                if frame < total.size:
                    total[frame] += float(value)
                    count[frame] += 1.0
    if not count.any():
        return None
    out = np.full(spec.default_num_frames, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


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
            axis.set_xlabel("t:  0 = clean video  ->  1000 = noise", fontsize=6.5, labelpad=1)

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
            axis.set_xlabel("t:  0 = clean video  ->  1000 = noise", fontsize=6.5, labelpad=1)
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


PLAUSIBLE_COLOUR = "#1b9e5a"
VIOLATED_COLOUR = "#d1495b"
DELTA_COLOUR = "#8d6bc8"
"""GeoPhys Figure 4's scheme: green valid, red invalid, and the difference in violet."""


def _draw_timeline(
    axes,
    names: Sequence[str],
    plausible: Mapping[str, np.ndarray],
    violated: Mapping[str, np.ndarray],
    limits: Mapping[str, tuple[float, float]],
    playhead: Optional[int] = None,
    series_label: str = "plausible",
) -> None:
    """One panel per signal: both clips, plus the signed gap filled against zero.

    The delta is the part that earns the figure. Two curves that nearly overlap look
    identical, and the question is not whether they differ overall but *when* -- whether
    the gap opens at the moment the violation happens. Filling it against zero on the same
    axis makes that readable at a glance, which is exactly what GeoPhys Figure 4 does.

    Sign convention here is the repo's, not the paper's: **violated minus plausible**, so
    a positive fill means the detector is calling that frame correctly. GeoPhys plots
    valid minus invalid, the other way round.
    """
    for axis, name in zip(axes, names):
        axis.clear()
        low, high = plausible.get(name), violated.get(name)

        # A generation has no counterpart to contrast against, so a single series is a
        # legitimate input: draw the signal, skip the delta, and do not pretend there is
        # a comparison.
        if low is not None and high is not None and len(low) == len(high):
            frames = np.arange(len(low))
            delta = high - low
            baseline = limits[name][0]
            # Anchored to the panel floor rather than to zero: these statistics are far
            # from zero, so a fill at true zero would be off-screen.
            axis.fill_between(frames, baseline, baseline + np.nan_to_num(delta),
                              where=~np.isnan(delta), color=DELTA_COLOUR, alpha=0.45,
                              linewidth=0, zorder=2,
                              label="violated - plausible")
            axis.axhline(baseline, color=palette.TEXT_MUTED, linewidth=0.8,
                         linestyle=(0, (3, 3)), zorder=1)

        for series, colour, label in (
            (low, PLAUSIBLE_COLOUR, series_label),
            (high, VIOLATED_COLOUR, "violated"),
        ):
            if series is None:
                continue
            axis.plot(np.arange(len(series)), series, color=colour, linewidth=1.6,
                      label=label, zorder=3)
            if playhead is not None and playhead < len(series) and not np.isnan(series[playhead]):
                axis.plot([playhead], [series[playhead]], marker="o", markersize=5,
                          color=colour, zorder=5)

        if playhead is not None:
            axis.axvline(playhead, color=palette.TEXT_PRIMARY, linewidth=1.2, zorder=4)

        axis.set_xlim(0, max(len(s) for s in (low, high) if s is not None) - 1)
        axis.set_ylim(*limits[name])
        axis.set_title(palette.statistic_label(name), fontsize=8, pad=3)
        axis.set_xlabel("video frame", fontsize=7, labelpad=1)
        axis.tick_params(labelsize=6.5)


def _timeline_limits(
    names: Sequence[str],
    plausible: Mapping[str, np.ndarray],
    violated: Mapping[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Fixed y-limits with headroom below for the delta fill."""
    limits = {}
    for name in names:
        stack = [s for s in (plausible.get(name), violated.get(name)) if s is not None]
        values = np.concatenate([s[~np.isnan(s)] for s in stack])
        low, high = float(values.min()), float(values.max())
        span = (high - low) or max(abs(high), 1.0)
        limits[name] = (low - 0.35 * span, high + 0.10 * span)
    return limits


def timeline_figure(
    plausible: Mapping[str, np.ndarray],
    violated: Optional[Mapping[str, np.ndarray]] = None,
    path: Path = None,
    title: str = "",
    caption: str = "",
    series_label: str = "plausible",
) -> Optional[Path]:
    """Static version: every per-frame signal, one panel each.

    ``violated`` is optional. With it, both curves are drawn and the signed gap filled --
    the paired case. Without it, a single trajectory is drawn on its own, which is what a
    *generated* clip needs: there is nothing to contrast it against, and inventing a
    comparison would be worse than showing one line.
    """
    violated = violated or {}
    import matplotlib.pyplot as plt

    names = [n for n in PER_FRAME if n in plausible or n in violated]
    if not names:
        return None

    palette.apply_style()
    figure, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.6), squeeze=False)
    _draw_timeline(axes[0], names, plausible, violated,
                   _timeline_limits(names, plausible, violated),
                   series_label=series_label)
    axes[0][0].legend(fontsize=7.5, loc="upper left")
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="bold")
    default = (
        "Per-frame signal for both clips, with the signed gap filled against the panel "
        "floor. Positive fill = the violated clip scores higher, which is the detector "
        "calling that frame correctly. "
    ) if violated else (
        "Per-frame signal for the generated clip. No counterpart exists to compare it "
        "against, so there is no gap to fill. "
    )
    palette.caption(figure, caption or default + (
        "Values are per latent frame, held across the four video frames each latent "
        "encodes, so the curves are step functions."
    ))
    figure.tight_layout(rect=(0, 0.06, 1, 0.94 if title else 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def timeline_strips(
    plausible: Mapping[str, np.ndarray],
    violated: Mapping[str, np.ndarray],
    width_px: int,
    caption: str = "",
    dpi: int = 100,
    series_label: str = "plausible",
) -> list[np.ndarray]:
    """One strip per video frame, x = **video frame**, with a playhead that tracks it.

    The animated twin of :func:`timeline_figure`. The axis means what it looks like it
    means here: the vertical line is the frame on screen above, so a spike in the delta
    lines up with the moment in the clip that caused it.

    Curves are drawn in full from the start rather than revealed -- the point is to see
    the spike coming and then watch the video reach it.
    """
    import matplotlib.pyplot as plt

    names = [n for n in PER_FRAME if n in plausible or n in violated]
    if not names:
        raise ValueError("no per-frame signals to draw")
    frames = max(len(s) for s in list(plausible.values()) + list(violated.values()))
    limits = _timeline_limits(names, plausible, violated)

    palette.apply_style()
    figure, axes = plt.subplots(
        1, len(names), figsize=(width_px / dpi, STRIP_HEIGHT), dpi=dpi, squeeze=False,
    )
    out: list[np.ndarray] = []
    for frame in range(frames):
        _draw_timeline(axes[0], names, plausible, violated, limits, playhead=frame,
                       series_label=series_label)
        axes[0][0].legend(fontsize=6, loc="upper left")
        if caption:
            palette.caption(figure, caption)
        figure.tight_layout(rect=(0, 0.10 if caption else 0, 1, 1))
        figure.canvas.draw()
        out.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())

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
