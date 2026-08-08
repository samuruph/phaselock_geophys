"""Chart styling, shared by every figure so the set reads as one system.

Colour is assigned by the *job* it does, not by taste:

* **categorical** -- identity (which statistic, which blur level). Fixed slot order,
  never cycled, never reassigned by rank, so a series keeps its hue when the set changes.
* **diverging** -- polarity. Detection accuracy is naturally diverging about chance:
  50% must read as "nothing", above and below as opposite. Blue and red are warm/cool
  poles with a neutral grey midpoint.
* **sequential** -- magnitude with a true zero. One hue, light to dark.

The categorical order below is validated for colour-vision deficiency on the adjacent
pairlist, which is what bar and line charts use.
"""

from __future__ import annotations

from typing import Any

# Validated categorical order. Adjacent pairs clear CVD deltaE >= 8 and normal-vision >= 15.
CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

PRIMARY = CATEGORICAL[0]
REFERENCE = CATEGORICAL[1]
"""The external DINOv2 baseline. A different entity from the internal sources, so it
carries its own hue rather than a shade of theirs."""

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e6e5e1"

# Diverging poles for accuracy about chance, with a neutral grey midpoint so 50%
# reads as "no signal" rather than as a colour of its own.
DIVERGING_LOW = "#1c5cab"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#b32d2d"

# One-hue sequential ramp, light to dark, for magnitudes with a true zero.
SEQUENTIAL = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#123a6c")

# Statistics keep a fixed hue wherever they appear, so a reader who learns
# "curvature is aqua" in one figure is not misled by the next.
STATISTIC_COLOURS = {
    "speed": CATEGORICAL[0],
    "curv": CATEGORICAL[2],
    "ang": CATEGORICAL[3],
    "accel": CATEGORICAL[4],
    "perr": CATEGORICAL[6],
    "or": CATEGORICAL[1],
    "majority": CATEGORICAL[5],
}

STATISTIC_LABELS = {
    "speed": r"$\varphi_{speed}$  (speed variation)",
    "curv": r"$\varphi_{curv}$  (mean turning angle)",
    "ang": r"$\varphi_{ang}$  (angle consistency)",
    "accel": r"$\varphi_{accel}$  (acceleration)",
    "perr": r"$\varphi_{perr}$  (prediction residual)",
    "or": "OR ensemble",
    "majority": "Majority ensemble",
}

SOURCE_LABELS = {
    "hidden_states": "DiT hidden states",
    "latent": "VAE latent  $x_t$",
    "x0_hat": "clean estimate  $\\hat{x}_0$",
    "velocity": "flow velocity  $u_\\theta$",
    "attention": "attention output",
    "dinov2": "DINOv2  (external)",
}


def apply_style() -> None:
    """Set global matplotlib defaults: recessive chrome, readable text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.bottom": False,
            "ytick.left": False,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "font.size": 9.5,
            "figure.dpi": 130,
        }
    )


def diverging_cmap() -> Any:
    """Blue-grey-red, for accuracy centred on chance."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "phaselock_diverging", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
    )


def sequential_cmap() -> Any:
    """One-hue light-to-dark, for magnitudes with a true zero."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("phaselock_sequential", list(SEQUENTIAL))


def statistic_label(name: str) -> str:
    return STATISTIC_LABELS.get(name, name)


def source_label(name: str) -> str:
    return SOURCE_LABELS.get(name, name)


def annotate_chance(axis, value: float = 50.0, horizontal: bool = False) -> None:
    """Mark the chance line, so "no signal" is visible rather than inferred."""
    draw = axis.axvline if horizontal else axis.axhline
    draw(value, color=TEXT_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    if horizontal:
        axis.text(
            value, 1.005, "chance", transform=axis.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=TEXT_MUTED,
        )
    else:
        axis.text(
            1.005, value, "chance", transform=axis.get_yaxis_transform(),
            ha="left", va="center", fontsize=8, color=TEXT_MUTED,
        )


def caption(figure, text: str) -> None:
    """One line under the figure saying what the reader should take from it."""
    figure.text(
        0.01, 0.005, text, ha="left", va="bottom",
        fontsize=8, color=TEXT_MUTED, wrap=True,
    )
