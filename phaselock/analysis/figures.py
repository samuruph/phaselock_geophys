"""Figures for the detection and generation results.

Each figure answers exactly one question, named in its title, with the takeaway spelled
out in a caption underneath. Accuracy plots always show the chance line, because "no
signal" is the null hypothesis and it should be visible rather than inferred.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import palette
from ..metrics.geophys import STATISTICS

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


def pooling_note(pooling: str, ratio: int = 4) -> str:
    """One line stating how the external path was matched to the latent grid."""
    if pooling == "latent":
        return (
            f"DINOv2 features are averaged over each latent's {ratio} video frames, matching "
            "the causal VAE, so both paths have the same trajectory length and spacing."
        )
    return (
        f"DINOv2 is per-frame ({ratio}x finer in time than the {ratio}-frame latents), so its "
        "statistics are NOT directly comparable to the internal ones. Re-run with "
        "--temporal-pool latent to match."
    )


def source_comparison(
    summaries: Sequence[Any],
    path: Path,
    external: Optional[Sequence[Any]] = None,
    null=None,
    external_pooling: str = "none",
    temporal_ratio: int = 4,
    value_label: str = "pairwise detection accuracy (%)",
    title: str = "Internal representations vs the external DINOv2 baseline",
    formula: str = "",
) -> Optional[Path]:
    """Mean accuracy per (source, statistic) across the whole probe grid.

    The headline figure. Bars are the **mean over every (block, step) cell**, with the
    error bar the spread across cells; the open diamond marks the single best cell.

    Reporting only that best cell -- which is what an earlier version of this figure did
    -- is a selection procedure over hundreds of correlated tests and reads high by
    construction. The mean answers the question actually being asked: does this
    representation carry the signal *wherever you look*, or only at one lucky cell?
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not summaries:
        return None

    # Only the value metrics belong here. phi_* and the two flow-coupling metrics are all
    # "a number read off the trajectory"; drift is a rate of change and gets its own
    # figure. Keying without `kind` silently collapsed the three together.
    summaries = [s for s in summaries if s.kind in ("phi", "coupling")]
    if external:
        summaries = list(summaries) + [s for s in external if s.kind in ("phi", "coupling")]
    if not summaries:
        return None

    internal_names = {s.source for s in summaries if s.source != "dinov2"}
    sources = sorted(internal_names, key=lambda name: -max(
        item.mean for item in summaries if item.source == name
    ))
    # The external baseline goes last, behind a rule: it is the comparison the project
    # exists to make, so it belongs on the same axes, but its cells are readout layers on
    # 800 pairs rather than probe locations on 36, so its noise floor is not the same one.
    has_external = any(s.source == "dinov2" for s in summaries)
    if has_external:
        sources = sources + ["dinov2"]
    statistics = [n for n in tuple(STATISTICS) + ("alignment", "erosion")
                  if any(s.statistic == n for s in summaries)]
    lookup = {(s.source, s.statistic): s for s in summaries}

    figure, axis = plt.subplots(figsize=(2.0 + 1.45 * len(sources), 5.2))
    width = 0.8 / max(len(statistics), 1)
    positions = np.arange(len(sources))

    if null is not None:
        # Two different nulls, because the bars and the diamonds are different statistics.
        # A mean over hundreds of cells cancels noise; a maximum selects for it.
        axis.axhspan(CHANCE, 100 * null.mean_null_p95, color=palette.GRID, alpha=0.9, zorder=0)
        axis.axhline(100 * null.null_p95, color=palette.TEXT_MUTED, linestyle=(0, (2, 3)),
                     linewidth=1.1, zorder=1)
        axis.text(
            len(sources) - 0.45, 100 * null.null_p95,
            f"  a DIAMOND counts only above here ({100 * null.null_p95:.0f}%) ",
            va="bottom", ha="right", fontsize=7.5, color=palette.TEXT_MUTED, zorder=5,
        )
        axis.text(
            -0.45, 100 * null.mean_null_p95,
            f" a BAR counts only above here ({100 * null.mean_null_p95:.0f}%)",
            va="bottom", ha="left", fontsize=7.5, color=palette.TEXT_MUTED, zorder=5,
        )

    for index, name in enumerate(statistics):
        offset = (index - (len(statistics) - 1) / 2) * width
        means = [100 * lookup[(src, name)].mean if (src, name) in lookup else np.nan for src in sources]
        stds = [100 * lookup[(src, name)].std if (src, name) in lookup else 0.0 for src in sources]
        axis.bar(
            positions + offset, means, width=width * 0.9, yerr=stds, capsize=2,
            color=palette.STATISTIC_COLOURS.get(name, palette.PRIMARY),
            error_kw={"elinewidth": 1.0, "ecolor": palette.TEXT_MUTED},
            label=palette.statistic_plain(name), zorder=3,
        )
        bests = [100 * lookup[(src, name)].best if (src, name) in lookup else np.nan for src in sources]
        axis.scatter(
            positions + offset, bests, marker="D", s=13, zorder=5,
            facecolors="none", edgecolors=palette.TEXT_SECONDARY, linewidths=0.9,
        )
        for x, best in zip(positions + offset, bests):
            if np.isnan(best):
                continue
            # Only the max is labelled here. Nine sources x seven statistics leaves no
            # room for two numbers per bar; the mean is already the bar height, and the
            # per-statistic figure carries both.
            axis.text(x, best + 1.4, f"{best:.0f}", ha="center", va="bottom",
                      fontsize=5.6, color=palette.TEXT_MUTED, zorder=6)

    if has_external:
        axis.axvline(len(sources) - 1.5, color=palette.TEXT_MUTED, linewidth=1.2,
                     linestyle=(0, (3, 3)), zorder=2)
        axis.text(len(sources) - 1.45, 97, " external", fontsize=7.5,
                  color=palette.TEXT_MUTED, va="top", ha="left")

    palette.annotate_chance(axis, CHANCE)
    axis.set_xticks(positions, [palette.source_label(s) for s in sources], fontsize=8.5)
    axis.set_ylabel(value_label)
    axis.set_title(title)


    # Two legends. Colour says *which statistic*; the glyphs say *what a mark means*, and
    # without naming them the vertical whisker reads as an error bar or a noise floor when
    # it is neither -- it is the spread of accuracy across probe cells.
    colours = axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=8)
    axis.add_artist(colours)
    glyph_legend = None
    if null is not None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        glyphs = [
            Line2D([0], [0], color=palette.TEXT_MUTED, linewidth=1.0,
                   label="whisker = how much the probe cells disagree (1 s.d.)"),
            Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="none",
                   markeredgecolor=palette.TEXT_SECONDARY,
                   label="diamond = single best cell"),
            Patch(facecolor=palette.GRID,
                  label=f"grey = a mean here is indistinguishable from chance"),
            Line2D([0], [0], color=palette.TEXT_MUTED, linestyle=(0, (2, 3)), linewidth=1.1,
                   label="dashed = a best-of-N pick needs to beat this"),
        ]
        glyph_legend = axis.legend(handles=glyphs, loc="upper center",
                                   bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.5)
    axis.grid(axis="x", visible=False)
    # From zero: a bar encodes magnitude by length, so a truncated baseline exaggerates
    # differences. The chance line and the null bands carry the reference instead.
    axis.set_ylim(0, 105)

    cells = max((s.n_cells for s in summaries if s.source != "dinov2"), default=0)
    external_cells = max((s.n_cells for s in summaries if s.source == "dinov2"), default=0)
    note = (
        f"Bar = mean over all {cells} probe cells (block x denoising step) for that source and "
        "statistic. The whisker is NOT an error bar on that mean: it is how much those "
        "cells disagree with each other, so short means the signal works at every depth "
        "and step, long means it works only in places. Open diamond = the single best cell."
    )
    if external_cells:
        # The two pools differ by an order of magnitude, so the diamonds are not carrying
        # equal selection burdens even when both clear the same drawn floor.
        note += (
            f" A DINOv2 cell is one of its {external_cells} readout layers, so its best is "
            f"selected from far fewer candidates than an internal best; compare the bars."
        )
    if null is not None:
        note += "\n" + pooling_note(external_pooling, temporal_ratio)
        note += (
            f"\nTwo nulls, from {null.n_signals} label-shuffled signals (n={null.n_pairs} pairs): "
            f"a mean clears noise above {100 * null.mean_null_p95:.0f}% (grey band), but a single "
            f"best cell needs {100 * null.null_p95:.0f}% (dashed line) because a maximum selects "
            "for noise while a mean cancels it."
        )
    # A derived quantity needs its definition visible. Generation reports a rank
    # concordance, not an accuracy, and the axis label alone cannot carry that. Placed
    # outside the axes: inside, it lands on the diamonds.
    if formula:
        figure.text(
            0.99, 0.995, formula, ha="right", va="top", fontsize=8.5,
            color=palette.TEXT_SECONDARY, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=palette.SURFACE,
                      edgecolor=palette.GRID, linewidth=0.9),
        )

    figure.tight_layout(rect=(0, 0.22, 1, 0.90 if formula else 1))
    # Only now is the axes its final size, and the legends are anchored in axes fractions:
    # a legend measured before this occupies a *smaller* fraction than it ends up with, so
    # a glyph legend placed from that measurement lands on the colour legend's last row.
    # The statistic count grew from five to ten, which is what pushed it into a third row.
    if glyph_legend is not None:
        figure.canvas.draw()
        glyph_legend.set_bbox_to_anchor(
            (0.5, colours.get_window_extent()
                  .transformed(axis.transAxes.inverted()).y0 - 0.04),
            transform=axis.transAxes,
        )
    palette.caption(figure, note, below=glyph_legend or colours)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def external_comparison(
    summaries: Sequence[Any], path: Path, internal: Optional[Sequence[Any]] = None,
    pooling: str = "none", temporal_ratio: int = 4,
    value_label: str = "pairwise detection accuracy (%)",
) -> Optional[Path]:
    """The external DINOv2 path, kept in its own figure.

    Separate from F1 deliberately: it is measured on a different sample with a different
    selection, so its noise floor differs and placing it on shared axes invites a
    comparison the numbers do not support.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not summaries:
        return None
    names = [s.statistic for s in summaries]
    means = [100 * s.mean for s in summaries]
    stds = [100 * s.std for s in summaries]

    # Widen with the statistic count. At five this fitted in 7 inches; at eight the names
    # are long enough that they run into each other and into the caption.
    figure, axis = plt.subplots(figsize=(max(7.0, 1.15 * len(names) + 2.2), 4.0))
    axis.bar(np.arange(len(names)), means, yerr=stds, capsize=3, width=0.6,
             color=palette.REFERENCE, error_kw={"elinewidth": 1.0, "ecolor": palette.TEXT_MUTED},
             zorder=3)
    axis.scatter(np.arange(len(names)), [100 * s.best for s in summaries], marker="D", s=16,
                 facecolors="none", edgecolors=palette.TEXT_SECONDARY, zorder=5)
    for index, item in enumerate(summaries):
        axis.text(index + 0.32, 100 * item.mean, f"{100*item.mean:.1f}", ha="left",
                  va="center", fontsize=7.5, color=palette.TEXT_SECONDARY)
        axis.text(index, 100 * item.best + 1.2, f"{100*item.best:.1f}", ha="center",
                  va="bottom", fontsize=7, color=palette.TEXT_MUTED)
    axis.set_ylim(0, 105)
    palette.annotate_chance(axis, CHANCE)
    axis.set_xticks(np.arange(len(names)), [palette.statistic_plain(n) for n in names],
                    fontsize=8, rotation=15, ha="right")
    axis.set_ylabel(value_label)
    axis.set_title("External baseline — frozen DINOv2 (the published GeoPhys method)")
    axis.grid(axis="x", visible=False)
    # Lay out first, then caption: the caption clears the rotated tick labels by measuring
    # where they actually end, which is only knowable once the axes is in its final place.
    figure.tight_layout()
    palette.caption(
        figure,
        f"Bar = mean over all {summaries[0].n_cells} readout layers; error bar = 1 s.d.; "
        "diamond = best layer. This is the correctness gate: GeoPhys reports 77.6-80.8% "
        f"for a single backbone on LikePhys.\n{pooling_note(pooling, temporal_ratio)}",
        below=axis,
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F2: where in depth does the signal live --------------------------------


def depth_profile(
    rows: Sequence[dict], path: Path, source: str = "hidden_states", kind: str = "phi",
    value_label: str = "pairwise detection accuracy (%)",
) -> Optional[Path]:
    """Accuracy against block index: mean across denoising steps, with the spread shaded.

    The geometric analogue of the Invisible Hand's Figure 4, which found linear-probe
    accuracy peaking in the middle third of the network.

    The solid line is the **mean over the recorded denoising steps** at that depth and the
    shaded cone is +/- 1 s.d.; the faint dotted line is the maximum. Plotting only the
    maximum, as an earlier version did, shows the luckiest step at each depth and so
    traces an envelope that no single probe location achieves.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    per_statistic: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["source"] != source or row.get("kind") != kind:
            continue
        block = int(row["block"])
        if block < 0:
            continue
        per_statistic[row["statistic"]][block].append(_pct(row["accuracy"]))
    if not per_statistic:
        return None

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    for name, series in sorted(per_statistic.items()):
        blocks = sorted(series)
        means = np.array([np.mean(series[b]) for b in blocks])
        stds = np.array([np.std(series[b]) for b in blocks])
        maxima = np.array([np.max(series[b]) for b in blocks])
        colour = palette.STATISTIC_COLOURS.get(name, palette.PRIMARY)

        axis.fill_between(blocks, means - stds, means + stds, color=colour, alpha=0.15, linewidth=0)
        axis.plot(blocks, means, color=colour, label=palette.statistic_plain(name),
                  marker="o", markersize=3.5)
        axis.plot(blocks, maxima, color=colour, linewidth=0.9, linestyle=(0, (1, 2)), alpha=0.55)

    palette.annotate_chance(axis, CHANCE)
    depth = max(max(s) for s in per_statistic.values())
    axis.axvspan(depth / 3, 2 * depth / 3, color=palette.GRID, alpha=0.5, zorder=0)
    axis.text(depth / 2, axis.get_ylim()[1], "middle third", ha="center", va="top",
              fontsize=8, color=palette.TEXT_MUTED)

    axis.set_xlabel(f"{palette.source_label(source)} — block index (0 = input side)")
    axis.set_ylabel(value_label)
    axis.set_title(f"Where in depth is plausibility readable? — {palette.source_label(source)}")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8)
    palette.caption(
        figure,
        "Solid line = mean over recorded denoising steps at that depth; shaded cone = 1 s.d.; "
        "faint dotted line = the best step at that depth. The shaded band is the middle third, "
        "where 'The Invisible Hand of Physics' reports linear probes peaking.",
    )
    figure.tight_layout(rect=(0, 0.15, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F3: depth x denoising time ---------------------------------------------


def depth_time_heatmaps(
    rows: Sequence[dict],
    path: Path,
    source: str = "hidden_states",
    kind: str = "phi",
    steps_to_timestep: Optional[Mapping[int, int]] = None,
    top_k: int = 5,
) -> Optional[Path]:
    """Denoising time against depth, aggregated over statistics two ways.

    Rows are diffusion timesteps and columns are blocks, so reading down a column shows
    how one depth behaves over the trajectory.

    Two panels because "best" and "typical" are different questions and a single matrix
    cannot answer both:

    * **left, max over the statistics** -- the best case at each location, which is
      what a tuned detector would use, and is upward-biased by the same selection effect
      that inflates any maximum;
    * **right, mean over the statistics** -- whether the location is informative in
      general rather than for one lucky statistic.

    Stars mark the top ``top_k`` cells of each panel, the brightest being the best.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["source"] != source or row.get("kind") != kind:
            continue
        cells[(int(row["step"]), int(row["block"]))].append(_pct(row["accuracy"]))
    if not cells:
        return None

    steps = sorted({step for step, _ in cells})
    blocks = sorted({block for _, block in cells})
    if len(blocks) < 2 or len(steps) < 2:
        return None

    shape = (len(steps), len(blocks))
    best = np.full(shape, np.nan)
    mean = np.full(shape, np.nan)
    for (step, block), values in cells.items():
        best[steps.index(step), blocks.index(block)] = max(values)
        mean[steps.index(step), blocks.index(block)] = float(np.mean(values))

    span = max(8.0, np.nanmax(np.abs(np.concatenate([best, mean]) - CHANCE)))
    figure, axes = plt.subplots(1, 2, figsize=(6.0 + 0.34 * len(blocks), 5.0),
                                squeeze=False, layout="constrained", sharey=True)
    image = None

    for axis, array, title in (
        (axes[0][0], best, "max over the statistics"),
        (axes[0][1], mean, "mean over the statistics"),
    ):
        image = axis.imshow(array, aspect="auto", cmap=palette.diverging_cmap(),
                            vmin=CHANCE - span, vmax=CHANCE + span, interpolation="nearest")
        axis.set_title(f"{title}\nbest cell {np.nanmax(array):.1f}%", fontsize=9)
        axis.set_xlabel("block index (depth) —>")
        axis.grid(False)

        shown = blocks if len(blocks) <= 14 else blocks[:: max(1, len(blocks) // 12)]
        axis.set_xticks([blocks.index(b) for b in shown], shown, fontsize=7.5)

        # Stars on the top cells: brightest for the best, dimmer for the runners-up.
        flat = np.argsort(np.nan_to_num(array, nan=-1e9), axis=None)[::-1][:top_k]
        for rank, index in enumerate(flat):
            row_index, column = np.unravel_index(index, array.shape)
            if np.isnan(array[row_index, column]):
                continue
            axis.scatter(
                column, row_index, marker="*",
                s=190 if rank == 0 else 95,
                facecolor="#ffd400" if rank == 0 else "#ffe98a",
                edgecolor=palette.TEXT_PRIMARY, linewidths=0.7 if rank == 0 else 0.4,
                zorder=6,
            )

    labels = [
        f"t={steps_to_timestep[s]}" if steps_to_timestep and s in steps_to_timestep else f"step {s}"
        for s in steps
    ]
    axes[0][0].set_yticks(range(len(steps)), labels, fontsize=7.5)
    axes[0][0].set_ylabel("diffusion timestep  (t=0 is the video, t~999 is noise)")

    bar = figure.colorbar(image, ax=axes[0], fraction=0.03, pad=0.02)
    bar.set_label("pairwise accuracy (%)", fontsize=8.5, color=palette.TEXT_SECONDARY)
    bar.ax.axhline(CHANCE, color=palette.TEXT_PRIMARY, linewidth=1.0)

    figure.suptitle(
        f"Depth against denoising time — {palette.source_label(source)}",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=palette.TEXT_PRIMARY,
    )
    figure.get_layout_engine().set(rect=(0, 0.085, 1, 0.92))
    palette.caption(
        figure,
        f"Stars: the {top_k} strongest cells per panel, brightest = best. Left is upward-biased "
        "by taking a maximum; the right panel is the one to trust for 'is this location "
        "informative'. Diverging scale centred on chance: white is no signal.",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F4: which statistic, and do the ensembles help -------------------------


def statistic_comparison(
    rows: Sequence[dict], path: Path, source: str = "hidden_states", kind: str = "phi",
    value_label: str = "pairwise detection accuracy (%)",
) -> Optional[Path]:
    """Accuracy per geometric statistic for one source: mean, spread and best.

    Same convention as the source comparison, so the two read together: bar = mean over
    the probe grid, error bar = 1 s.d., open diamond = best cell.

    GeoPhys's headline claim is that combining the statistics beats any of them alone,
    because different signals catch different violations. The ensembles sit to the right
    of the divider, which is where that either shows up or does not.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["source"] != source or row.get("kind") != kind:
            continue
        grouped[row["statistic"]].append(_pct(row["accuracy"]))
    if not grouped:
        return None

    singles = [n for n in STATISTICS if n in grouped]
    coupling = [n for n in ("alignment", "erosion") if n in grouped]
    ensembles = [n for n in ("or", "majority") if n in grouped]
    names = singles + coupling + ensembles
    if not names:
        return None

    means = np.array([np.mean(grouped[n]) for n in names])
    stds = np.array([np.std(grouped[n]) for n in names])
    bests = np.array([np.max(grouped[n]) for n in names])

    figure, axis = plt.subplots(figsize=(1.15 * len(names) + 3.0, 4.6))
    axis.bar(np.arange(len(names)), means, yerr=stds, capsize=3, width=0.62,
             color=[palette.STATISTIC_COLOURS.get(n, palette.PRIMARY) for n in names],
             error_kw={"elinewidth": 1.0, "ecolor": palette.TEXT_MUTED}, zorder=3)
    axis.scatter(np.arange(len(names)), bests, marker="D", s=16, zorder=5,
                 facecolors="none", edgecolors=palette.TEXT_SECONDARY, linewidths=0.9)
    palette.annotate_chance(axis, CHANCE)

    boundary = len(singles) + len(coupling) - 0.5
    if ensembles:
        axis.axvline(boundary, color=palette.GRID, linewidth=1.4)
    if coupling:
        axis.axvline(len(singles) - 0.5, color=palette.GRID, linewidth=1.4,
                     linestyle=(0, (3, 3)))

    for index, (mean, std, best) in enumerate(zip(means, stds, bests)):
        axis.text(index + 0.33, mean, f"{mean:.1f}", ha="left", va="center",
                  fontsize=7.5, color=palette.TEXT_SECONDARY)
        axis.text(index, best + 1.2, f"{best:.1f}", ha="center", va="bottom",
                  fontsize=7, color=palette.TEXT_MUTED)

    axis.set_xticks(np.arange(len(names)),
                    [palette.statistic_plain(n) for n in names],
                    fontsize=8, rotation=20, ha="right")
    axis.set_ylim(0, max(105, float(bests.max()) + 8))
    axis.set_ylabel(value_label)
    axis.set_title(f"Which statistic carries the signal? — {palette.source_label(source)}")
    axis.grid(axis="x", visible=False)
    palette.caption(
        figure,
        "Bar = mean over the probe grid, error bar = 1 s.d., diamond = best cell. Dashed "
        "divider separates the eight trajectory statistics from the two flow-coupling "
        "metrics; solid divider separates both from the ensembles over the statistics.",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- F5: statistic vs its rate of change under the flow ---------------------


def drift_comparison(
    rows: Sequence[dict], path: Path,
    value_label: str = "pairwise detection accuracy (%)",
) -> Optional[Path]:
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
        paired[row["source"]].setdefault(kind, []).append(_pct(row["accuracy"]))
    complete = {s: v for s, v in paired.items() if "phi" in v and "drift" in v}
    if not complete:
        return None

    summary = {
        source: {k: (float(np.mean(v)), float(np.std(v))) for k, v in kinds.items()}
        for source, kinds in complete.items()
    }
    order = sorted(summary, key=lambda s: -summary[s]["phi"][0])
    positions = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(7.8, 4.4))
    axis.bar(positions - 0.19, [summary[s]["phi"][0] for s in order], width=0.36,
             yerr=[summary[s]["phi"][1] for s in order], capsize=2,
             error_kw={"elinewidth": 1.0, "ecolor": palette.TEXT_MUTED},
             color=palette.CATEGORICAL[0], label=r"$\varphi_\sigma$  the statistic", zorder=3)
    axis.bar(positions + 0.19, [summary[s]["drift"][0] for s in order], width=0.36,
             yerr=[summary[s]["drift"][1] for s in order], capsize=2,
             error_kw={"elinewidth": 1.0, "ecolor": palette.TEXT_MUTED},
             color=palette.CATEGORICAL[2], label=r"$\dot{g}_\sigma$  its drift under the flow", zorder=3)
    palette.annotate_chance(axis, CHANCE)

    axis.set_xticks(positions, [palette.source_label(s) for s in order], fontsize=8)
    for index, source in enumerate(order):
        for offset, kind in ((-0.19, "phi"), (0.19, "drift")):
            mean, std = summary[source][kind]
            axis.text(index + offset + 0.19, mean, f"{mean:.0f}", ha="left", va="center",
                      fontsize=7, color=palette.TEXT_SECONDARY)

    axis.set_ylim(0, 105)
    axis.set_ylabel(value_label)
    axis.set_title("Is the geometry more telling than its rate of change under the flow?")
    axis.legend(loc="upper right")
    axis.grid(axis="x", visible=False)

    # State the metric, so the reader is not guessing what "drift" means.
    axis.text(
        0.015, 0.965,
        r"$\dot{g}_\sigma(\tau)\;=\;\left\langle\, \nabla_{\bar{z}}\,\varphi_\sigma(\bar{z}(\tau)),"
        r"\;\bar{u}_\theta(z,\tau) \,\right\rangle$" "\n"
        r"$\dot{g}_\sigma<0$: the step is making the trajectory more regular",
        transform=axis.transAxes, va="top", ha="left", fontsize=8.5,
        color=palette.TEXT_SECONDARY,
        bbox=dict(facecolor=palette.SURFACE, edgecolor=palette.GRID, boxstyle="round,pad=0.45"),
    )
    palette.caption(
        figure,
        "Bars are means over the probe grid. The drift is exact only for the VAE latent, which "
        "is the ODE state itself; every other source uses a finite difference across recorded "
        "steps and is limited by the recording stride.",
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


SWEEP_METRICS = (
    ("phase_difference_corr", "PhaseLock phase correlation", True),
    ("raw_score", "motion-mask fidelity to the real clip", True),
    ("phi_curv", "mean turning angle", False),
    ("phi_accel", "acceleration", False),
    ("phi_perr", "prediction residual", False),
    ("phi_speed", "speed variation", False),
)
"""``(column, label, higher_is_better)``. The two reproduction metrics come first: they
answer whether PhaseLock's effect exists at all before the geometric ones ask whether
trajectory geometry sees it."""


def step_sweep_panels(cells: Sequence[dict], path: Path) -> Optional[Path]:
    """Every sweep metric against K, one line per blur level, in one figure.

    The whole experiment in one place. PhaseLock's claim is that a 2-step output is more
    physically consistent than a 50-step one; the confound is that it is also blurrier, so
    the same blur is applied to the generation *and* the real reference and the question
    becomes whether the ordering survives.

    Direction is not uniform across these panels and is stated per panel, because reading
    it wrongly inverts the conclusion: motion-mask fidelity and phase correlation are
    higher-is-better, while every geometric statistic is lower-is-more-regular.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    available = [
        (column, label, higher)
        for column, label, higher in SWEEP_METRICS
        if any(cell.get(column) not in (None, "") for cell in cells)
    ]
    if not available:
        return None

    sigmas = sorted({float(c["blur_sigma"]) for c in cells})
    columns = min(3, len(available))
    rows = (len(available) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.5 * rows),
                                squeeze=False)

    for index, (column, label, higher) in enumerate(available):
        axis = axes[index // columns][index % columns]
        for position, sigma in enumerate(sigmas):
            by_step: dict[int, list[float]] = defaultdict(list)
            for cell in cells:
                if float(cell["blur_sigma"]) != sigma or cell.get(column) in (None, ""):
                    continue
                by_step[int(cell["num_steps"])].append(float(cell[column]))
            if not by_step:
                continue
            steps = sorted(by_step)
            means = np.array([np.mean(by_step[s]) for s in steps])
            errors = np.array([np.std(by_step[s]) / max(1, len(by_step[s])) ** 0.5 for s in steps])
            colour = palette.SEQUENTIAL[min(1 + position, len(palette.SEQUENTIAL) - 1)]
            axis.plot(steps, means, marker="o", markersize=4, color=colour,
                      label=f"$\\sigma$ = {sigma:g}")
            axis.fill_between(steps, means - errors, means + errors, color=colour, alpha=0.16)

        axis.set_title(label, fontsize=10)
        axis.set_xlabel("denoising steps  K")
        axis.set_ylabel("higher = better" if higher else "lower = more regular", fontsize=8)
        axis.set_xticks(sorted({int(c["num_steps"]) for c in cells}))

    for spare in range(len(available), rows * columns):
        axes[spare // columns][spare % columns].axis("off")
    axes[0][0].legend(fontsize=8, loc="best", title="blur", title_fontsize=8)

    figure.suptitle("Does the few-step effect survive the blur control?",
                    x=0.01, ha="left", fontweight="bold")
    palette.caption(
        figure,
        "Blur is applied to the generation and the real reference alike, so a K=2 advantage "
        "that only exists at sigma=0 was measuring sharpness, not physics. Band = standard "
        "error over clips. Note the direction differs per panel and is labelled on each y-axis.",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def generation_quality(cells: Sequence[dict], path: Path) -> Optional[Path]:
    """Per-scenario fidelity of the generated continuation to the real one.

    Scored against ground truth by the Physics-IQ motion-mask family, which applies here
    because LikePhys is rendered with a static camera. This says nothing about geometry --
    it is the independent yardstick that arbitrates when the geometric readout and the
    fidelity ordering disagree.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("spatial_iou", "Spatial IoU"),
        ("spatiotemporal_iou", "Spatiotemporal IoU"),
        ("weighted_spatial_iou", "Weighted spatial IoU"),
        ("mse", "MSE  (lower is better)"),
    ]
    usable = [(c, l) for c, l in metrics if any(x.get(c) not in (None, "") for x in cells)]
    if not usable or not cells:
        return None

    scenarios = sorted({c["scenario"] for c in cells})
    figure, axes = plt.subplots(1, len(usable), figsize=(3.6 * len(usable), 4.4), squeeze=False)

    for axis, (column, label) in zip(axes[0], usable):
        values = [
            [float(c[column]) for c in cells if c["scenario"] == s and c.get(column) not in (None, "")]
            for s in scenarios
        ]
        means = [np.mean(v) if v else np.nan for v in values]
        order = np.argsort([-m if column != "mse" else m for m in means])
        axis.barh(
            np.arange(len(scenarios)),
            [means[i] for i in order],
            color=palette.PRIMARY, height=0.72,
        )
        axis.set_yticks(np.arange(len(scenarios)), [scenarios[i] for i in order], fontsize=7.5)
        axis.invert_yaxis()
        axis.set_title(label, fontsize=9.5)
        axis.grid(axis="y", visible=False)

    figure.suptitle("Generated continuation vs the real one, per scenario",
                    x=0.01, ha="left", fontweight="bold")
    palette.caption(
        figure,
        "Both arms start from the same first frame of a *valid* clip, so the real continuation "
        "is a physically plausible reference. Ground-truth based, so this does not assume the "
        "geometric readout measures anything.",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# -- GeoPhys Figure 3 analogue: the statistic itself, not its accuracy -------


def signal_profile(
    statistics_rows: Sequence[dict],
    path: Path,
    source: str = "hidden_states",
    x_axis: str = "block",
    steps_to_timestep: Optional[Mapping[int, int]] = None,
) -> Optional[Path]:
    """Raw statistic values for plausible vs violated clips, across depth or time.

    The analogue of GeoPhys Figure 3, which plots mean curvature per layer with plausible
    below violated at every layer. This shows every statistic rather than curvature
    alone, plus the two new flow-coupling metrics where available.

    Unlike every other figure here, the y-axis is the **statistic itself**, not a
    detection accuracy. That matters: a gap between the two curves is the raw effect,
    before any thresholding or pairing, so it shows both *whether* the classes separate
    and *in which direction*. GeoPhys's claim is that violated sits above plausible
    everywhere, since all are oriented so larger means less regular.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = [n for n in STATISTICS if f"phi_{n}" in (statistics_rows[0] if statistics_rows else {})]
    if not names:
        return None

    key = "block" if x_axis == "block" else "step"
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in statistics_rows:
        if row["source"] != source:
            continue
        try:
            position, label = int(row[key]), int(row["label"])
        except (KeyError, TypeError, ValueError):
            continue
        for name in names:
            value = row.get(f"phi_{name}")
            if value not in (None, ""):
                grouped[(name, position, label)].append(float(value))

    positions = sorted({pos for _, pos, _ in grouped})
    if len(positions) < 2:
        return None

    columns = min(len(names), 3)
    rows_count = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(4.3 * columns, 3.3 * rows_count),
                                squeeze=False, layout="constrained")

    for index, name in enumerate(names):
        axis = axes[index // columns][index % columns]
        for label, colour, text in ((0, palette.CATEGORICAL[0], "plausible"),
                                    (1, palette.CATEGORICAL[7], "violated")):
            means, stds, valid = [], [], []
            for position in positions:
                values = grouped.get((name, position, label), [])
                if values:
                    valid.append(position)
                    means.append(float(np.mean(values)))
                    stds.append(float(np.std(values)))
            if not valid:
                continue
            means, stds = np.array(means), np.array(stds)
            axis.plot(valid, means, color=colour, label=text, marker="o", markersize=3)
            axis.fill_between(valid, means - stds, means + stds, color=colour, alpha=0.16,
                              linewidth=0)
        axis.set_title(palette.statistic_label(name), fontsize=9)
        axis.set_xlabel("block index (depth)" if key == "block" else "diffusion timestep")
        if key == "step" and steps_to_timestep:
            shown = [p for p in positions if p in steps_to_timestep]
            axis.set_xticks(shown, [steps_to_timestep[p] for p in shown], fontsize=7, rotation=90)
        if index % columns == 0:
            axis.set_ylabel("statistic value")
        if index == 0:
            axis.legend(loc="best", fontsize=8)

    for spare in range(len(names), rows_count * columns):
        axes[spare // columns][spare % columns].axis("off")

    figure.suptitle(
        f"Statistic value, plausible vs violated — {palette.source_label(source)}",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=palette.TEXT_PRIMARY,
    )
    figure.get_layout_engine().set(rect=(0, 0.055, 1, 0.94))
    palette.caption(
        figure,
        "Line = mean across clips, band = 1 s.d. All are oriented so larger means less "
        "regular, so GeoPhys predicts violated (red) above plausible (blue) at every depth. "
        "This is the raw effect, before any pairing or thresholding.",
    )
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
    summaries: Optional[Sequence[Any]] = None,
    external_summaries: Optional[Sequence[Any]] = None,
    null=None,
    steps_to_timestep: Optional[Mapping[int, int]] = None,
    sweep: Optional[Sequence[dict]] = None,
    latent_profiles: Optional[dict] = None,
    statistics_rows: Optional[Sequence[dict]] = None,
    external_rows: Optional[Sequence[dict]] = None,
    external_pooling: str = "none",
    temporal_ratio: int = 4,
    value_label: str = "pairwise detection accuracy (%)",
    title: str = "Internal representations vs the external DINOv2 baseline",
    formula: str = "",
) -> list[Path]:
    """Render the whole figure set.

    Cross-source figures land at the top level; the per-source figures are repeated for
    **every** representation in its own subdirectory, so each one can be read on its own
    terms rather than only for whichever source happened to score highest.
    """
    palette.apply_style()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def add(result: Optional[Path]) -> None:
        if result is not None:
            written.append(result)

    # -- cross-source ------------------------------------------------------
    if summaries:
        add(source_comparison(summaries, directory / "01_source_comparison.png",
                              external=external_summaries, null=null,
                              external_pooling=external_pooling, temporal_ratio=temporal_ratio,
                              value_label=value_label, title=title, formula=formula))
    if external_summaries:
        add(external_comparison(external_summaries, directory / "02_external_baseline.png",
                                pooling=external_pooling, temporal_ratio=temporal_ratio))
    add(drift_comparison(rows, directory / "03_statistic_vs_drift.png",
                         value_label=value_label))
    if sweep:
        add(step_sweep(sweep, directory / "04_step_sweep.png"))
    if latent_profiles:
        add(latent_motion_profile(latent_profiles, directory / "05_latent_motion.png"))

    # -- per source --------------------------------------------------------
    # DINOv2's "depth" is its readout layer and it has no denoising trajectory, so its
    # rows are mapped onto the same schema with block = layer and a single step. That
    # makes the whole per-source figure set apply to the external baseline too.
    combined_rows = list(rows)
    combined_statistics = list(statistics_rows or [])
    if external_rows:
        combined_rows += [dict(row, source="dinov2") for row in external_rows]
    for source in sorted({row["source"] for row in combined_rows}):
        folder = directory / source
        folder.mkdir(parents=True, exist_ok=True)
        add(statistic_comparison(combined_rows, folder / "01_statistic_comparison.png",
                                 source=source, value_label=value_label))
        add(depth_profile(combined_rows, folder / "02_depth_profile.png", source=source,
                          value_label=value_label))
        add(depth_time_heatmaps(combined_rows, folder / "03_depth_vs_time.png", source=source,
                                steps_to_timestep=steps_to_timestep))
        if combined_statistics:
            # GeoPhys Figure 3 analogue: the statistic itself for plausible vs violated.
            add(signal_profile(combined_statistics, folder / "04_signal_profile_depth.png",
                               source=source, x_axis="block"))
            add(signal_profile(combined_statistics, folder / "05_signal_profile_time.png",
                               source=source, x_axis="step",
                               steps_to_timestep=steps_to_timestep))
    return written


def category_table(
    cells: Sequence[Any],
    path: Path,
    title: str = "Violation detection by category",
    value_label: str = "pairwise accuracy (%)",
) -> Optional[Path]:
    """Signals down the rows, categories across, coloured red to green.

    The aggregate says whether a representation carries the signal; this says *where it
    fails*, and the n=96 run showed that is far from uniform -- rigid-body scenarios at
    100% against `river` at chance. Two methods with the same mean can differ completely
    here, which is the point of breaking it out.

    Each cell reads ``mean +/- sd (max)``: the mean over the probe grid is the honest
    number, the spread says whether it works everywhere or in patches, and the max is what
    a best-of-N search would have found. Printing the max alone is the selection error
    this project keeps having to guard against, so it is present but parenthesised.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    if not cells:
        return None

    statistics = list(STATISTICS)
    sources = palette.ordered_sources({c.source for c in cells}, baselines_first=True)
    categories = sorted({c.category for c in cells})
    lookup = {(c.source, c.statistic, c.category): c for c in cells}

    # One header row per source, then its statistics. Two-level row labels crammed into a
    # single left margin is what made the previous version unreadable; a section header is
    # how the table this imitates does it.
    layout: list[tuple[str, str, str]] = []
    for src in sources:
        present = [s for s in statistics
                   if any((src, s, cat) in lookup for cat in categories)]
        if not present:
            continue
        layout.append(("head", src, ""))
        layout.extend(("cell", src, stat) for stat in present)
    if not any(kind == "cell" for kind, _, _ in layout):
        return None

    def value(src: str, stat: str, cat: str) -> float:
        cell = lookup.get((src, stat, cat))
        return float("nan") if cell is None else 100 * cell.mean

    # The row mean across categories, which is the column the eye actually compares. Named
    # M.Avg for continuity with the benchmark tables this mirrors.
    averages: dict[tuple[str, str], tuple[float, float]] = {}
    for kind, src, stat in layout:
        if kind != "cell":
            continue
        row = [value(src, stat, cat) for cat in categories]
        row = [v for v in row if not np.isnan(v)]
        if row:
            averages[(src, stat)] = (
                float(np.mean(row)), float(np.std(row)) if len(row) > 1 else 0.0
            )

    baselines = set(palette.BASELINE_SOURCES)
    columns = list(categories) + ["M.Avg."]

    def best_in(column: str, want_baseline: bool) -> Optional[tuple[str, str]]:
        """Best (source, statistic) in one column, within or outside the baseline block."""
        scored = [
            ((src, stat), averages[(src, stat)][0] if column == "M.Avg."
             else value(src, stat, column))
            for kind, src, stat in layout
            if kind == "cell" and ((src in baselines) == want_baseline)
            and (src, stat) in averages
        ]
        scored = [(key, v) for key, v in scored if not np.isnan(v)]
        return max(scored, key=lambda kv: kv[1])[0] if scored else None

    marks = {
        col: (best_in(col, want_baseline=False), best_in(col, want_baseline=True))
        for col in columns
    }

    # Column x: categories packed, then a gap, then M.Avg set apart -- it aggregates the
    # others and must not read as one more category.
    xs = {cat: float(i) for i, cat in enumerate(categories)}
    xs["M.Avg."] = len(categories) + 0.35
    width, height = 1.12, 1.0
    label_width = 4.6

    figure, axis = plt.subplots(figsize=(
        1.12 * (len(categories) + 1.6) + label_width,
        0.30 * len(layout) + 2.9,
    ))
    axis.set_axis_off()
    left, right = -label_width / 1.12, len(categories) + 1.1
    axis.set_xlim(left, right)
    axis.set_ylim(len(layout) - 0.5, -3.9)

    cmap = plt.get_cmap("RdYlGn")
    # Diverging about chance, so 50% reads as "nothing" rather than as a colour. Clipped
    # symmetrically: without it one 100% cell washes out every real difference.
    normalise = plt.Normalize(vmin=30, vmax=90)

    for column in columns:
        x = xs[column]
        axis.text(x, -1.25, column.replace(" ", "\n") if column != "M.Avg." else column,
                  ha="center", va="bottom", fontsize=8.5,
                  fontweight="bold" if column == "M.Avg." else "normal",
                  color=palette.TEXT_PRIMARY)
    axis.plot([left, len(categories) + 1.0], [-0.55, -0.55],
              color=palette.TEXT_PRIMARY, linewidth=1.4, clip_on=False)

    # Title and colour key share the top band, both in data coordinates so they stay put
    # however many rows there are. A vertical bar down the side of a table this tall floats
    # in the middle of nothing and reads as a sixth column.
    axis.text(left, -3.08, title, ha="left", va="center", fontsize=13,
              fontweight="bold", color=palette.TEXT_PRIMARY)
    key_left, key_right = len(categories) - 2.0, len(categories) + 1.0
    axis.imshow(np.linspace(30, 90, 256)[None, :], cmap=cmap, norm=normalise,
                extent=(key_left, key_right, -2.95, -3.21), aspect="auto", zorder=3)
    axis.add_patch(Rectangle((key_left, -3.21), key_right - key_left, 0.26,
                             facecolor="none", edgecolor=palette.TEXT_MUTED,
                             linewidth=0.6, zorder=4))
    for value in (30, 50, 70, 90):
        x = key_left + (value - 30) / 60 * (key_right - key_left)
        if value == 50:
            axis.plot([x, x], [-3.21, -2.95], color=palette.TEXT_PRIMARY, linewidth=1.2,
                      zorder=5)
        axis.text(x, -2.81, str(value), ha="center", va="center", fontsize=6.5,
                  color=palette.TEXT_MUTED)
    axis.text((key_left + key_right) / 2, -3.40, f"{value_label}   (50 = chance)",
              ha="center", va="center", fontsize=7.5, color=palette.TEXT_MUTED)

    for r, (kind, src, stat) in enumerate(layout):
        if kind == "head":
            tag = "  — external baseline" if src in baselines else ""
            axis.text(-label_width / 1.12, r, palette.source_label(src) + tag,
                      ha="left", va="center", fontsize=9.5, fontweight="bold",
                      color=palette.TEXT_PRIMARY)
            if r:
                axis.plot([-label_width / 1.12, len(categories) + 1.0],
                          [r - 0.55, r - 0.55], color=palette.TEXT_PRIMARY,
                          linewidth=1.0 if src not in baselines else 1.4, clip_on=False)
            continue

        axis.text(-label_width / 1.12 + 0.35, r, palette.statistic_plain(stat),
                  ha="left", va="center", fontsize=8, color=palette.TEXT_PRIMARY)

        for column in columns:
            x, cell = xs[column], lookup.get((src, stat, column))
            if column == "M.Avg.":
                if (src, stat) not in averages:
                    continue
                mean, spread = averages[(src, stat)]
                head, tail, best = f"{mean:.1f}", f" ±{spread:.1f}", None
            else:
                if cell is None:
                    axis.text(x, r, "–", ha="center", va="center", fontsize=8,
                              color=palette.TEXT_MUTED)
                    continue
                mean, spread, best = 100 * cell.mean, 100 * cell.std, 100 * cell.best
                head, tail = f"{mean:.0f}", f" ±{spread:.0f}"
                axis.add_patch(Rectangle(
                    (x - width / 2, r - height / 2), width, height,
                    facecolor=cmap(normalise(mean)), edgecolor="none", zorder=0,
                ))

            strong, weak = marks[column]
            key = (src, stat)
            bold = key == strong and src not in baselines
            ink = palette.TEXT_PRIMARY if column == "M.Avg." or 40 < mean < 82 else "#fff"
            axis.text(x - 0.06, r - (0.14 if best is not None else 0.0), head,
                      ha="right", va="center", fontsize=8.6,
                      fontweight="bold" if bold else "normal", color=ink)
            axis.text(x - 0.05, r - (0.14 if best is not None else 0.0), tail,
                      ha="left", va="center", fontsize=6.6, color=ink)
            if best is not None:
                axis.text(x, r + 0.24, f"({best:.0f})", ha="center", va="center",
                          fontsize=5.9, color=ink, alpha=0.75)
            if key == weak and src in baselines:
                axis.plot([x - 0.34, x + 0.34], [r + 0.40, r + 0.40],
                          color=ink, linewidth=0.9)

    palette.caption(figure, (
        "Each cell is mean ± s.d. over the probe grid, with the single best cell in "
        "parentheses; M.Avg. averages the categories. The mean is the number to read; the "
        "spread says whether the signal works everywhere or only in patches, and the max "
        "is what a best-of-N search would have found. 50% is chance (marked on the bar). "
        "Bold: best internal readout per column. Underline: best baseline per column."
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    return path


QUALITY_METRICS = (
    ("raw_score", "PhysicsIQ score"),
    ("spatial_iou", "Spatial IoU"),
    ("spatiotemporal_iou", "Spatiotemporal IoU"),
    ("weighted_spatial_iou", "Weighted spatial IoU"),
    ("mse", "MSE  (lower is better)"),
)


def signal_quality_correlation(
    statistics: Sequence[dict],
    candidates: Sequence[dict],
    path: Path,
    top: int = 6,
) -> Optional[Path]:
    """Does a signal predict how good the generation turned out?

    The plain version of the question, with no rescaling. Left: the Spearman correlation
    of every ``(source, statistic)`` against each ground-truth quality metric, so a bar
    reaching left means "larger statistic, worse generation" -- the direction a working
    detector should point, since every statistic is oriented so larger means less regular.
    Right: the actual scatter for the strongest signals, because a correlation coefficient
    hides whether the relationship is real or one outlier.

    Reading rho directly avoids the mapping that made this confusing: it is a correlation
    between two numbers per clip, not an accuracy, and nothing is being counted.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from ..experiments.generation import spearman

    fidelity = {
        column: {r["sample_id"]: float(r[column]) for r in candidates
                 if r.get(column) not in (None, "")}
        for column, _ in QUALITY_METRICS
    }
    fidelity = {k: v for k, v in fidelity.items() if len(v) >= 3}
    if not fidelity:
        return None

    # One value per clip per (source, statistic): the mean over the probe grid, matching
    # how every other figure summarises depth and denoising time.
    pooled: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in statistics:
        for name in STATISTICS:
            value = row.get(f"phi_{name}")
            if value not in (None, ""):
                pooled[(row["source"], name)][row["sample_id"]].append(float(value))
    if not pooled:
        return None
    per_signal = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in pooled.items()}

    columns = [c for c, _ in QUALITY_METRICS if c in fidelity]
    labels = dict(QUALITY_METRICS)
    keys = sorted(per_signal)
    grid = np.full((len(keys), len(columns)), np.nan)
    for r, key in enumerate(keys):
        for c, column in enumerate(columns):
            shared = sorted(set(per_signal[key]) & set(fidelity[column]))
            if len(shared) >= 3:
                grid[r, c] = spearman([per_signal[key][s] for s in shared],
                                      [fidelity[column][s] for s in shared])

    figure = plt.figure(figsize=(15.5, 0.42 * len(keys) + 4.6))
    grid_spec = figure.add_gridspec(2, max(top, 3), height_ratios=[len(keys) * 0.30, 3.4],
                                    hspace=0.55)
    axis = figure.add_subplot(grid_spec[0, :])

    width = 0.8 / len(columns)
    positions = np.arange(len(keys))
    for index, column in enumerate(columns):
        offset = (index - (len(columns) - 1) / 2) * width
        axis.barh(positions + offset, grid[:, index], height=width * 0.9,
                  color=palette.CATEGORICAL[index % len(palette.CATEGORICAL)],
                  label=labels[column])
    axis.axvline(0, color=palette.TEXT_MUTED, linewidth=1.1)
    axis.set_yticks(positions,
                    [f"{palette.source_label(s)} · {palette.statistic_plain(n)}"
                     for s, n in keys], fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("Spearman correlation with generation quality"
                    "        <-- larger statistic, worse generation")
    axis.set_title(
        "Do the signals predict how good the generation is?\n"
        "signal averaged over the probe grid first, then correlated  "
        "(figure 01 does the reverse: correlate per cell, then average)",
        loc="left", fontsize=11)
    axis.legend(fontsize=7.5, ncol=len(columns), loc="upper center",
                bbox_to_anchor=(0.5, -0.16))
    axis.grid(axis="y", visible=False)

    # The scatters. A coefficient cannot tell you whether the trend is real.
    reference = "raw_score" if "raw_score" in fidelity else columns[0]
    strongest = sorted(
        range(len(keys)),
        key=lambda r: -abs(grid[r, columns.index(reference)])
        if not np.isnan(grid[r, columns.index(reference)]) else 0,
    )[:top]
    for slot, r in enumerate(strongest):
        panel = figure.add_subplot(grid_spec[1, slot])
        source, name = keys[r]
        shared = sorted(set(per_signal[keys[r]]) & set(fidelity[reference]))
        panel.scatter([per_signal[keys[r]][s] for s in shared],
                      [fidelity[reference][s] for s in shared],
                      s=22, color=palette.STATISTIC_COLOURS.get(name, palette.PRIMARY),
                      alpha=0.85, edgecolors="none")
        rho = grid[r, columns.index(reference)]
        panel.set_title(f"{palette.source_label(source)}\n{palette.statistic_plain(name)}"
                        f"   $\\rho$={rho:+.2f}", fontsize=7.5)
        panel.set_xlabel("statistic", fontsize=7)
        if slot == 0:
            panel.set_ylabel(labels[reference], fontsize=7.5)
        panel.tick_params(labelsize=6.5)

    palette.caption(figure, (
        f"One point per generated clip: the statistic (averaged over the probe grid) "
        f"against ground-truth {labels[reference].lower()}. Negative rho is the useful "
        f"direction -- every statistic is oriented so larger means less regular, so a "
        f"working signal should fall as quality rises. This is a correlation between two "
        f"numbers per clip, not an accuracy: nothing is being counted."
    ))
    # No tight_layout: the gridspec already fixes the two-panel split, and letting
    # tight_layout re-solve it warns and shifts the legend under the bars.
    figure.subplots_adjust(bottom=0.10, top=0.94)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(figure)
    return path
