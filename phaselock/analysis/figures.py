"""Figures for the detection and generation results.

Each figure answers exactly one question, named in its title, with the takeaway spelled
out in a caption underneath. Accuracy plots always show the chance line, because "no
signal" is the null hypothesis and it should be visible rather than inferred.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import palette

CHANCE = 50.0


def _pct(value: Any) -> float:
    return 100.0 * float(value)


def _by_source(rows: Sequence[dict], kind: str = "phi") -> dict[str, dict]:
    """Best-scoring row per source."""
    best: dict[str, dict] = {}
    for row in rows:
        if row.get("kind") != kind:
            continue
        current = best.get(row["source"])
        if current is None or float(row["accuracy"]) > float(current["accuracy"]):
            best[row["source"]] = row
    return best


# -- F1: which representation carries the geometry --------------------------


def source_comparison(
    rows: Sequence[dict],
    path: Path,
    external: Optional[dict] = None,
    kind: str = "phi",
    null=None,
) -> Optional[Path]:
    """Best pairwise accuracy per representation, against the external baseline.

    The headline figure: it is the whole point of the project stated as one chart.
    Sources are nominal categories, so they share one hue rather than a value ramp --
    the bar length already encodes the magnitude. The external baseline is a different
    kind of thing, so it gets its own hue.
    """
    import matplotlib.pyplot as plt

    best = _by_source(rows, kind)
    if not best:
        return None

    # Internal sources form one comparable group; the external baseline is a different
    # kind of thing, measured on a different sample with a different selection, so it is
    # separated rather than interleaved into the ranking.
    entries = sorted(best.items(), key=lambda item: float(item[1]["accuracy"]))
    if external:
        entries = [("dinov2", external)] + entries

    labels = [palette.source_label(name) for name, _ in entries]
    values = [_pct(row["accuracy"]) for _, row in entries]
    lows = [max(0.0, value - _pct(row["ci_low"])) for value, (_, row) in zip(values, entries)]
    highs = [max(0.0, _pct(row["ci_high"]) - value) for value, (_, row) in zip(values, entries)]
    colours = [
        palette.REFERENCE if name == "dinov2" else palette.PRIMARY for name, _ in entries
    ]

    figure, axis = plt.subplots(figsize=(9.8, 0.66 * len(entries) + 2.9))
    if null is not None:
        # Selecting the best of thousands of signals is biased upward; this is how high
        # that selection climbs on label-shuffled data, where there is nothing to find.
        #
        # The band covers only the internal rows. The external baseline was selected from
        # far fewer signals on far more pairs, so its noise floor is much lower and
        # sweeping one band across both would understate it.
        internal = [i for i, (name, _) in enumerate(entries) if name != "dinov2"]
        if internal:
            from matplotlib.patches import Rectangle

            low, high = min(internal) - 0.45, max(internal) + 0.45
            axis.add_patch(Rectangle(
                (CHANCE, low), 100 * null.null_p95 - CHANCE, high - low,
                facecolor=palette.GRID, edgecolor="none", alpha=0.85, zorder=0,
            ))
            axis.text(
                100 * null.null_p95 - 0.8, low + 0.05,
                f"best of {null.n_signals} internal signals reaches here by chance  ",
                va="bottom", ha="right", fontsize=7.5, color=palette.TEXT_MUTED, zorder=5,
            )
            # A rule between the external baseline and the internal group.
            axis.axhline(min(internal) - 0.5, color=palette.GRID, linewidth=1.4, zorder=2)
    bars = axis.barh(labels, values, height=0.62, color=colours, zorder=3)
    axis.errorbar(
        values, range(len(values)), xerr=[lows, highs], fmt="none",
        ecolor=palette.TEXT_MUTED, elinewidth=1.2, capsize=3, zorder=4,
    )
    palette.annotate_chance(axis, CHANCE, horizontal=True)

    # Labels sit past the *whisker*, not past the bar, or they collide with the CI.
    label_x = [max(value, _pct(row["ci_high"])) + 1.4 for value, (_, row) in zip(values, entries)]
    for bar, value, x, (name, row) in zip(bars, values, label_x, entries):
        detail = "" if name == "dinov2" else f"  {_signal_location(row)}"
        axis.text(
            x, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%  (n={int(row['n_pairs'])}){detail}", va="center", ha="left",
            fontsize=8.5, color=palette.TEXT_SECONDARY,
        )

    axis.set_xlim(40, max(102, max(label_x) + 17))
    axis.set_xlabel("pairwise detection accuracy (%)   —  50% is chance")
    axis.set_title("Which representation separates plausible from violated physics?")
    axis.grid(axis="y", visible=False)
    note = (
        "Best probe location per source (95% bootstrap CI, grouped by scenario). "
        f"{palette.NOTATION_KEY}"
    )
    if null is not None:
        note += (
            f"\nGrey band = selection noise floor for the internal sources: the best of "
            f"{null.n_signals} signals reaches {100 * null.null_p95:.0f}% on shuffled labels. "
            "Only bars clear of it are real. It does not apply to DINOv2, selected from far "
            "fewer signals on more pairs."
        )
    palette.caption(figure, note)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _signal_location(row: dict) -> str:
    """Where the signal was read, in words. "b7/s4" tells a reader nothing."""
    block = int(row["block"])
    where = "" if block < 0 else f"block {block}, "
    return f"{where}step {int(row['step'])} — {palette.statistic_plain(row['statistic'])}"


# -- F2: where in depth does the signal live --------------------------------


def depth_profile(
    rows: Sequence[dict], path: Path, source: str = "hidden_states", kind: str = "phi"
) -> Optional[Path]:
    """Accuracy against block index, one line per statistic.

    The geometric analogue of the Invisible Hand's Figure 4, which found linear-probe
    accuracy peaking in the middle third of the network. A line plot rather than a
    heatmap because the question is *where the peak is*, which a line answers directly.
    """
    import matplotlib.pyplot as plt

    per_statistic: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row["source"] != source or row.get("kind") != kind:
            continue
        block = int(row["block"])
        if block < 0:
            continue
        # Best over denoising steps, so the line is "what this depth can do".
        accuracy = _pct(row["accuracy"])
        per_statistic[row["statistic"]][block] = max(
            per_statistic[row["statistic"]].get(block, 0.0), accuracy
        )
    if not per_statistic:
        return None

    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    for name, series in sorted(per_statistic.items()):
        blocks = sorted(series)
        axis.plot(
            blocks, [series[b] for b in blocks],
            color=palette.STATISTIC_COLOURS.get(name, palette.PRIMARY),
            label=palette.statistic_label(name), marker="o", markersize=4,
        )
    palette.annotate_chance(axis, CHANCE)

    depth = max(max(s) for s in per_statistic.values())
    axis.axvspan(depth / 3, 2 * depth / 3, color=palette.GRID, alpha=0.55, zorder=0)
    axis.text(
        depth / 2, axis.get_ylim()[1], "middle third", ha="center", va="top",
        fontsize=8, color=palette.TEXT_MUTED,
    )

    axis.set_xlabel(f"{palette.source_label(source)} — block index (0 = input side)")
    axis.set_ylabel("pairwise detection accuracy (%)")
    axis.set_title("Where in depth is physical plausibility most readable?")
    axis.legend(loc="best", ncol=2)
    palette.caption(
        figure,
        "Best over recorded denoising steps at each depth. The shaded band is the middle "
        "third, where 'The Invisible Hand of Physics' reports linear probes peaking.",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F3: depth x denoising time ---------------------------------------------


def depth_time_heatmaps(
    rows: Sequence[dict], path: Path, statistic: str = "curv", kind: str = "phi"
) -> Optional[Path]:
    """Block x denoising-step accuracy, one panel per source, on a shared scale.

    Diverging about chance: 50% is the neutral midpoint, so above and below read as
    opposite rather than as two shades of the same thing.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    grids: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    for row in rows:
        if row["statistic"] != statistic or row.get("kind") != kind:
            continue
        grids[row["source"]][(int(row["block"]), int(row["step"]))] = _pct(row["accuracy"])
    # This figure is about depth, so a source with a single block has nothing to show
    # here and would be stretched into a meaningless one-row panel.
    grids = {k: v for k, v in grids.items() if len({b for b, _ in v}) > 1}
    if not grids:
        return None

    order = sorted(grids, key=lambda s: -max(grids[s].values()))
    # constrained_layout rather than tight_layout: the shared colourbar is attached to
    # the whole axes row, which tight_layout cannot solve.
    figure, axes = plt.subplots(
        1, len(order), figsize=(3.1 * len(order) + 1.4, 4.4), squeeze=False,
        layout="constrained",
    )
    extreme = max(abs(v - CHANCE) for grid in grids.values() for v in grid.values())
    span = max(8.0, extreme)
    image = None

    for axis, source in zip(axes[0], order):
        grid = grids[source]
        blocks = sorted({b for b, _ in grid})
        steps = sorted({s for _, s in grid})
        array = np.full((len(blocks), len(steps)), np.nan)
        for (block, step), value in grid.items():
            array[blocks.index(block), steps.index(step)] = value

        image = axis.imshow(
            array, aspect="auto", cmap=palette.diverging_cmap(),
            vmin=CHANCE - span, vmax=CHANCE + span, interpolation="nearest",
        )
        axis.set_xticks(range(len(steps)), steps, fontsize=7.5)
        show = blocks if len(blocks) <= 12 else blocks[:: max(1, len(blocks) // 10)]
        axis.set_yticks([blocks.index(b) for b in show],
                        ["-" if b < 0 else b for b in show], fontsize=7.5)
        axis.set_title(f"{palette.source_label(source)}\nbest {max(grid.values()):.1f}%", fontsize=9)
        axis.set_xlabel("recorded step")
        axis.grid(False)
    axes[0][0].set_ylabel("block index")

    bar = figure.colorbar(image, ax=axes[0], fraction=0.025, pad=0.02)
    bar.set_label("pairwise accuracy (%)", fontsize=8.5, color=palette.TEXT_SECONDARY)
    bar.ax.axhline(CHANCE, color=palette.TEXT_PRIMARY, linewidth=1.0)

    figure.suptitle(
        f"Detection accuracy across depth and denoising time — {palette.statistic_label(statistic)}",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=palette.TEXT_PRIMARY,
    )
    # constrained_layout does not know about figure.text, so reserve the strip itself.
    figure.get_layout_engine().set(rect=(0, 0.07, 1, 0.90))
    palette.caption(
        figure,
        "Step 0 is the clean end of the trajectory, the last is the noisy end. Shared diverging "
        "scale centred on chance: white is no signal, red above, blue below.",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F4: which statistic, and do the ensembles help -------------------------


def statistic_comparison(
    rows: Sequence[dict], path: Path, source: Optional[str] = None
) -> Optional[Path]:
    """Accuracy per geometric statistic, with the OR and Majority ensembles alongside.

    GeoPhys's headline claim is that the ensemble beats any single statistic, because
    different signals catch different violations. This is where that either shows up or
    does not.
    """
    import matplotlib.pyplot as plt

    if source is None:
        best = _by_source(rows, "phi")
        if not best:
            return None
        source = max(best, key=lambda name: float(best[name]["accuracy"]))

    scores: dict[str, float] = {}
    for row in rows:
        if row["source"] != source:
            continue
        name = row["statistic"]
        scores[name] = max(scores.get(name, 0.0), _pct(row["accuracy"]))
    if not scores:
        return None

    singles = [n for n in ("speed", "curv", "ang", "accel", "perr") if n in scores]
    ensembles = [n for n in ("or", "majority") if n in scores]
    names = singles + ensembles
    values = [scores[n] for n in names]

    figure, axis = plt.subplots(figsize=(7.4, 4.0))
    bars = axis.bar(
        range(len(names)), values, width=0.62,
        color=[palette.STATISTIC_COLOURS.get(n, palette.PRIMARY) for n in names], zorder=3,
    )
    palette.annotate_chance(axis, CHANCE)
    if ensembles:
        axis.axvline(len(singles) - 0.5, color=palette.GRID, linewidth=1.4)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}",
            ha="center", va="bottom", fontsize=8.5, color=palette.TEXT_SECONDARY,
        )

    axis.set_xticks(range(len(names)),
                    [n if n not in ("or", "majority") else n.upper() for n in names])
    axis.set_ylim(40, max(100, max(values) + 8))
    axis.set_ylabel("pairwise detection accuracy (%)")
    axis.set_title(f"Which geometric statistic carries the signal? — {palette.source_label(source)}")
    axis.grid(axis="x", visible=False)
    palette.caption(
        figure,
        "Left of the divider: the five statistics individually. Right: combined across all "
        "five. GeoPhys reports the OR combination well above any single signal.",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F5: statistic vs its rate of change under the flow ---------------------


def drift_comparison(rows: Sequence[dict], path: Path) -> Optional[Path]:
    """The statistic against its geometric drift, at the same probe location.

    Asks whether a violated clip is better identified by *what* the geometry is, or by
    how hard the model's own flow is pushing against it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        kind = row.get("kind")
        if kind not in ("phi", "drift"):
            continue
        paired[row["source"]][kind] = max(
            paired[row["source"]].get(kind, 0.0), _pct(row["accuracy"])
        )
    complete = {s: v for s, v in paired.items() if "phi" in v and "drift" in v}
    if not complete:
        return None

    order = sorted(complete, key=lambda s: -complete[s]["phi"])
    positions = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(7.4, 4.0))
    axis.bar(positions - 0.19, [complete[s]["phi"] for s in order], width=0.36,
             color=palette.CATEGORICAL[0], label=r"$\varphi_\sigma$  (the statistic)", zorder=3)
    axis.bar(positions + 0.19, [complete[s]["drift"] for s in order], width=0.36,
             color=palette.CATEGORICAL[2], label=r"$\dot{g}_\sigma$  (its drift under the flow)", zorder=3)
    palette.annotate_chance(axis, CHANCE)

    axis.set_xticks(positions, [palette.source_label(s) for s in order], fontsize=8)
    axis.set_ylim(40, 100)
    axis.set_ylabel("pairwise detection accuracy (%)")
    axis.set_title("Is the geometry more telling than its rate of change?")
    axis.legend(loc="upper right")
    axis.grid(axis="x", visible=False)
    palette.caption(
        figure,
        "Drift is exact only for the VAE latent, which is the ODE state; every other source "
        "uses a finite difference across recorded steps and is resolution-limited.",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F6: the step sweep with PhaseLock's blur control -----------------------


def step_sweep(cells: Sequence[dict], path: Path, metric: str = "phi_curv") -> Optional[Path]:
    """A metric against denoising steps, one line per blur level.

    PhaseLock's claim is that a 2-step output is more physically consistent than a
    50-step one. The confound is that it is also blurrier, so the same blur is applied
    to every arm and the question becomes whether the ordering survives.
    """
    import matplotlib.pyplot as plt

    series: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        value = cell.get(metric)
        if value in (None, ""):
            continue
        series[float(cell["blur_sigma"])][int(cell["num_steps"])].append(float(value))
    if not series:
        return None

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for index, sigma in enumerate(sorted(series)):
        by_step = series[sigma]
        steps = sorted(by_step)
        means = [sum(by_step[s]) / len(by_step[s]) for s in steps]
        axis.plot(
            steps, means, marker="o",
            color=palette.CATEGORICAL[index % len(palette.CATEGORICAL)],
            label=f"blur $\\sigma$ = {sigma:g}",
        )

    axis.set_xlabel("denoising steps  K")
    axis.set_ylabel(palette.statistic_label(metric.replace("phi_", "")))
    axis.set_title("Does the few-step effect survive the blur control?")
    axis.legend(loc="best")
    palette.caption(
        figure,
        "Blur is applied to the generation and the real reference alike. If the K=2 advantage "
        "only exists at sigma=0, it was measuring sharpness rather than physics.",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F7: what lives inside a single latent ----------------------------------


def latent_motion_profile(
    profiles: dict[str, Any], path: Path
) -> Optional[Path]:
    """Per-latent-frame energy for plausible against violated clips.

    A causal VAE bundles four video frames into one latent, so motion *within* a latent
    is invisible to a frame difference along the latent axis: a latent containing fast
    motion looks identical to a static one if its endpoints match. This asks whether the
    energy inside each latent is itself a usable prior.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not profiles or "plausible" not in profiles or "violated" not in profiles:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for axis, key, title in (
        (axes[0], "per_frame", "Energy per latent frame"),
        (axes[1], "per_tau", "Energy along the denoising trajectory"),
    ):
        for label, colour in (("plausible", palette.CATEGORICAL[2]), ("violated", palette.CATEGORICAL[7])):
            data = profiles[label].get(key)
            if data is None:
                continue
            x, mean, spread = data
            axis.plot(x, mean, color=colour, marker="o", markersize=4, label=label)
            axis.fill_between(
                x, np.asarray(mean) - np.asarray(spread), np.asarray(mean) + np.asarray(spread),
                color=colour, alpha=0.16, linewidth=0,
            )
        axis.set_title(title, fontsize=9.5)
        axis.legend(loc="best")

    axes[0].set_xlabel("latent frame index  (0 = first video frame, then 4 frames each)")
    axes[0].set_ylabel("channel std of the latent")
    axes[1].set_xlabel(r"denoising time  $\tau$   (0 = noise, 1 = clean)")
    axes[1].set_ylabel("mean channel std")

    figure.suptitle(
        "What does a single latent hold, and how does it evolve?",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=palette.TEXT_PRIMARY,
    )
    palette.caption(
        figure,
        "Shaded band is +/- 1 s.d. across clips. A separation here would be a prior that a "
        "frame difference along the latent axis cannot see, because it lives inside a latent.",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def render_all(
    rows: Sequence[dict],
    directory: Path,
    external: Optional[dict] = None,
    sweep: Optional[Sequence[dict]] = None,
    latent_profiles: Optional[dict] = None,
) -> list[Path]:
    """Render every figure the available data supports."""
    palette.apply_style()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def add(result: Optional[Path]) -> None:
        if result is not None:
            written.append(result)

    add(source_comparison(rows, directory / "01_source_comparison.png", external))
    add(depth_profile(rows, directory / "02_depth_profile.png"))
    best = _by_source(rows, "phi")
    statistic = (
        max(best.values(), key=lambda row: float(row["accuracy"]))["statistic"] if best else "curv"
    )
    add(depth_time_heatmaps(rows, directory / "03_depth_vs_time.png", statistic))
    add(statistic_comparison(rows, directory / "04_statistic_comparison.png"))
    add(drift_comparison(rows, directory / "05_statistic_vs_drift.png"))
    if sweep:
        add(step_sweep(sweep, directory / "06_step_sweep.png"))
    if latent_profiles:
        add(latent_motion_profile(latent_profiles, directory / "07_latent_motion.png"))
    return written
