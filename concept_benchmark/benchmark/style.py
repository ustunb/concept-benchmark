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

        from concept_benchmark.benchmark.style import set_paper_style
        set_paper_style()
    """
    plt.rcParams.update(_PAPER_RC)


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
