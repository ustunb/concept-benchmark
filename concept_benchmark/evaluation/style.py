"""Shared visual style for benchmark plots.

Colors are colorblind-safe (Wong palette), matching the paper figures.
Call :func:`set_paper_style` once before creating plots to apply the
full rcParams configuration.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ── Colors (Wong colorblind-safe palette, matching paper) ────────────

BLUE = "#0072B2"  # SelectiveAccuracy / Accuracy
BLUE_LIGHT = "#56B4E9"  # Accuracy secondary variant
GREEN = "#00BA38"  # Coverage
VERMILLION = "#D55E00"  # NetWorkAutomated
PURPLE = "#8B5CF6"  # Gain
GREY = "#404040"  # DNN baseline / neutral
RED = "#D32F2F"  # Negative bars (material red, matching paper)

# Semantic aliases
COLOR_ACCURACY = BLUE
COLOR_ACCURACY_ALT = BLUE_LIGHT
COLOR_COVERAGE = GREEN
COLOR_NET_WORK = VERMILLION
COLOR_GAIN = PURPLE
COLOR_POSITIVE = GREEN  # Positive regime bars
COLOR_NEGATIVE = RED  # Negative regime bars
COLOR_BASELINE = GREY

# ── Font sizes ───────────────────────────────────────────────────────

FONT_SIZE = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 11
FONT_SIZE_ANNOT = 11

# ── Full rcParams (matching paper's make_figures.py) ─────────────────

_PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.facecolor": "white",
    "axes.edgecolor": "#555555",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "legend.facecolor": "white",
    "legend.fontsize": FONT_SIZE_LEGEND,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
}


def set_paper_style() -> None:
    """Apply the paper's visual style globally via ``plt.rcParams``.

    Call once before creating any plots::

        from concept_benchmark.evaluation.style import set_paper_style
        set_paper_style()
    """
    plt.rcParams.update(_PAPER_RC)


# ── The paper's panels ───────────────────────────────────────────────

PANEL_BLUE = "#3B6FB6"
PANEL_RED = "#C44E52"
PANEL_GREEN = "#4C9F70"
PANEL_GREY = "#4D4D4D"
PANEL_GRID = "#DFE4E9"
PANEL_TITLE = "#8C8C8C"
PANEL_AXIS = "#737373"
PANEL_SERIF = ["cmr10", "DejaVu Serif"]  # axis text, as in the paper's body
PANEL_SANS = ["Helvetica", "Arial", "DejaVu Sans"]  # model and metric names
PANEL_COLORS = [PANEL_BLUE, PANEL_RED, PANEL_GREEN, "#8172B2", "#CCB974", "#64B5CD"]
PANEL_MARKERS = ["o", "^", "s", "D", "v", "P"]


def format_name(name: str) -> str:
    """Legend text: words with an underscore are names from the code and are set in monospace."""
    return " ".join(
        rf"$\mathtt{{{word.replace('_', chr(92) + '_')}}}$" if "_" in word else word
        for word in name.split(" ")
    )


def apply_panel_style(
    ax, xlabel: str, title: str | None = None, subtitle: str | None = None
) -> None:
    """Axes, grid, fonts and title of the paper's two main panels."""
    plt.rcParams["mathtext.fontset"] = "cm"  # math in Computer Modern, as in the paper
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.yaxis.grid(True, color=PANEL_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor(PANEL_AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=PANEL_AXIS, length=4, direction="in", labelsize=11)
    ax.tick_params(axis="y", right=False)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontfamily(PANEL_SERIF)
        tick.set_color("#404040")
    ax.set_xlabel(xlabel, fontfamily=PANEL_SERIF, fontsize=11.5, color=PANEL_GREY)
    ax.set_ylabel("")
    if title:
        heading = rf"$\mathbf{{{title}}}$" + (f" {subtitle}" if subtitle else "")
        ax.annotate(
            heading,
            xy=(0, 1.13),
            xycoords="axes fraction",
            fontfamily=PANEL_SERIF,
            fontsize=11.5,
            color=PANEL_TITLE,
            va="bottom",
        )
        ax.plot(
            [0, 1],
            [1.1, 1.1],
            transform=ax.transAxes,
            color="#B8B8B8",
            linewidth=0.6,
            clip_on=False,
        )


# ── Helpers ──────────────────────────────────────────────────────────


def pct_formatter() -> FuncFormatter:
    """Matplotlib formatter that displays values as percentages (e.g. 85%)."""
    return FuncFormatter(lambda x, _: f"{x:.0f}%")


def apply_style(ax) -> None:
    """Apply the standard paper style to a matplotlib Axes.

    This is called automatically by all plot functions. For the full
    global style (fonts, grid, legend frame), also call :func:`set_paper_style`.
    """
    # Match paper exactly: grid behind bars, subtle color
    ax.set_axisbelow(True)
    ax.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#555555")
    ax.spines["bottom"].set_edgecolor("#555555")
    ax.tick_params(labelsize=FONT_SIZE_TICK)
