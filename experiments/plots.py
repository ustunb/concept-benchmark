"""Plots for selective-automation results.

Follows the conventions in notes/figure-conventions.md, distilled from the published
CBM intervention figures: a curve over the budget anchored at k=0, raw quantities on
y, translucent same-colour bands for uncertainty and nothing else, redundant
colour/marker encoding, a shared frameless legend below the panel strip, all four
spines, no gridlines, and y limits shared across the row so that non-monotone curves
are visually obvious.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# Matches the paper: the venue style sets \sfdefault to phv, so method and metric
# names (\CBM, \DNN, \wrk) are Helvetica; maths is left in Computer Modern, which
# is also what notes/figure-conventions.md requires ("math stays in the math font").
PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def use_paper_style() -> None:
    """Apply the paper's figure fonts and sizes to the current matplotlib session."""
    plt.rcParams.update(PAPER_STYLE)


COVERAGE_COLOR = "#3B6FB6"
WORK_COLOR = "#4C9F70"
BASELINE_COLOR = "#000000"
BAND_ALPHA = 0.20


@dataclass
class AutomationSeries:
    """Coverage and net automated work over a sweep of intervention budgets."""

    budgets: Sequence[int | str]
    coverage: Sequence[float]
    net_work: Sequence[float]
    coverage_err: Sequence[float] | None = None
    net_work_err: Sequence[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.budgets)
        if not (len(self.coverage) == len(self.net_work) == n):
            raise ValueError("budgets, coverage and net_work must have equal length")
        for err in (self.coverage_err, self.net_work_err):
            if err is not None and len(err) != n:
                raise ValueError("error sequences must match the number of budgets")


def _extent(series: AutomationSeries) -> tuple[float, float]:
    lows, highs = [], []
    for values, errors in (
        (series.coverage, series.coverage_err),
        (series.net_work, series.net_work_err),
    ):
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors if errors is not None else 0.0, dtype=float)
        lows.append(float((values - errors).min()))
        highs.append(float((values + errors).max()))
    return min(lows), max(highs)


def plot_coverage_and_work(
    series_by_target: Mapping[float, AutomationSeries],
    baseline_by_target: Mapping[float, float] | None = None,
    *,
    model_label: str = "CBM",
    baseline_label: str = "DNN  (NetWorkAutomated = Coverage)",
    target_label: str = "target accuracy",
    panel_width: float = 2.1,
    panel_height: float = 2.1,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw coverage and net automated work against the intervention budget.

    One panel per target, y limits shared across the row.
    """
    targets = sorted(series_by_target)
    if not targets:
        raise ValueError("series_by_target is empty")
    baseline_by_target = baseline_by_target or {}

    lows, highs = zip(*(_extent(series_by_target[t]) for t in targets))
    low, high = min(lows), max(highs)
    if baseline_by_target:
        low = min(low, min(baseline_by_target.values()))
    pad = 0.06 * (high - low)

    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(panel_width * len(targets), panel_height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for column, target in enumerate(targets):
        series = series_by_target[target]
        ax = axes[column]
        x = np.arange(len(series.budgets))

        for values, errors, colour, marker, label in (
            (
                series.coverage,
                series.coverage_err,
                COVERAGE_COLOR,
                "o",
                f"{model_label} coverage",
            ),
            (
                series.net_work,
                series.net_work_err,
                WORK_COLOR,
                "s",
                f"{model_label} NetWorkAutomated",
            ),
        ):
            values = np.asarray(values, dtype=float)
            if errors is not None:
                errors = np.asarray(errors, dtype=float)
                ax.fill_between(
                    x,
                    values - errors,
                    values + errors,
                    color=colour,
                    alpha=BAND_ALPHA,
                    lw=0,
                )
            ax.plot(
                x,
                values,
                color=colour,
                lw=1.4,
                marker=marker,
                ms=3.6,
                label=label if column == 0 else None,
            )

        baseline = baseline_by_target.get(target)
        if baseline is not None:
            ax.plot(
                x,
                np.full_like(x, baseline, dtype=float),
                color=BASELINE_COLOR,
                lw=1.1,
                label=baseline_label if column == 0 else None,
            )

        ax.set_ylim(low - pad, high + pad)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in series.budgets])
        ax.set_xlabel("Intervention budget $k$")
        ax.set_title(rf"$\tau = {target * 100:g}\%$ {target_label}")
        for name, side in ax.spines.items():
            side.set_visible(name in ("left", "bottom"))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(targets) == 1:
        axes[0].legend(
            handles,
            labels,
            frameon=False,
            loc="center left",
            handlelength=1.6,
            borderaxespad=0.8,
        )
        fig.tight_layout()
        return fig, list(axes)
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, list(axes)
