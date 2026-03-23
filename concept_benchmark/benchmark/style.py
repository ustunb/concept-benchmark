"""Shared visual style for benchmark plots.

Colors are colorblind-safe, matching the paper figures.
"""

from __future__ import annotations

from matplotlib.ticker import FuncFormatter

# ── Colors (colorblind-safe) ─────────────────────────────────────────

BLUE = "#0072B2"
BLUE_LIGHT = "#56B4E9"
GREEN = "#00BA38"
VERMILLION = "#D55E00"
PURPLE = "#8B5CF6"
GREY = "#404040"

# Semantic mapping
COLOR_ACCURACY = BLUE
COLOR_ACCURACY_ALT = BLUE_LIGHT
COLOR_COVERAGE = GREEN
COLOR_NET_WORK = VERMILLION
COLOR_GAIN = PURPLE
COLOR_BASELINE = GREY

# ── Font sizes ───────────────────────────────────────────────────────

FONT_SIZE = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 11
FONT_SIZE_ANNOT = 11

# ── Formatters ───────────────────────────────────────────────────────


def pct_formatter() -> FuncFormatter:
    """Matplotlib formatter that displays values as percentages (e.g. 85%)."""
    return FuncFormatter(lambda x, _: f"{x:.0f}%")


def apply_style(ax) -> None:
    """Apply the standard paper style to a matplotlib Axes."""
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
