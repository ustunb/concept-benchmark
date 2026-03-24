"""Reusable plotting functions for benchmark results.

Each function takes data (DataFrames or dicts) and returns ``(fig, ax)``.
Pass an existing ``ax`` to compose multiple plots on one figure.

Example::

    from concept_benchmark.benchmark.plots import plot_intervention_curve
    import pandas as pd

    df = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.78, 0.92, 0.94]})
    fig, ax = plot_intervention_curve(df)
    fig.savefig("intervention_curve.png")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style


def _ensure_ax(ax, figsize=(6, 4)):
    """Return (fig, ax), creating them if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


# ── Intervention curve ───────────────────────────────────────────────


def plot_intervention_curve(
    results: pd.DataFrame,
    metric: str = "accuracy",
    baseline_accuracy: float | None = None,
    label: str | None = None,
    color: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of a metric vs intervention budget *k*.

    Parameters
    ----------
    results : DataFrame
        Must have columns ``budget`` and the column named by *metric*.
    metric : str
        Column name to plot on the y-axis (default ``"accuracy"``).
    baseline_accuracy : float, optional
        If provided, draw a horizontal dashed line for the DNN baseline.
    label : str, optional
        Legend label for the line.
    color : str, optional
        Line color (default: blue).
    ax : Axes, optional
        Existing axes to plot on.
    """
    fig, ax = _ensure_ax(ax)
    color = color or style.COLOR_ACCURACY

    budgets = results["budget"].values
    values = results[metric].values * 100  # to percentage

    ax.plot(budgets, values, marker="o", color=color, linewidth=2, label=label)

    for x, y in zip(budgets, values):
        ax.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=style.FONT_SIZE_ANNOT,
            ha="center",
            clip_on=False,
        )

    if baseline_accuracy is not None:
        bval = baseline_accuracy * 100
        ax.axhline(bval, color=style.COLOR_BASELINE, linestyle="--", linewidth=1.5, label="DNN baseline")

    ax.set_xlabel("Intervention budget (k)", fontsize=style.FONT_SIZE)
    ax.set_ylabel(metric.replace("_", " ").title() + " (%)", fontsize=style.FONT_SIZE)
    ax.yaxis.set_major_formatter(style.pct_formatter())

    # Pad y-axis to avoid clipping annotations
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 1, ymax + 3)

    style.apply_style(ax)

    if label or baseline_accuracy is not None:
        ax.legend(fontsize=style.FONT_SIZE_LEGEND)

    return fig, ax


# ── Regime comparison ────────────────────────────────────────────────


def plot_regime_comparison(
    regime_df: pd.DataFrame,
    budgets: list[int] | None = None,
    regime_order: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of mean ΔAccuracy per regime with min/max error bars.

    Parameters
    ----------
    regime_df : DataFrame
        Must have columns ``regime``, ``budget``, and ``accuracy``.
        Each regime must have a ``budget=0`` row for the baseline accuracy.
    budgets : list of int, optional
        Budgets to average over for the mean bar (default ``[1, 2, 5]``).
    regime_order : list of str, optional
        Order of regimes from top to bottom.
    ax : Axes, optional
        Existing axes to plot on.
    """
    fig, ax = _ensure_ax(ax, figsize=(7, 4))
    budgets = budgets or [1, 2, 5]

    regimes = regime_order or sorted(regime_df["regime"].unique())

    means, mins, maxs, labels = [], [], [], []
    for regime in regimes:
        rdf = regime_df[regime_df["regime"] == regime]
        k0_acc = rdf[rdf["budget"] == 0]["accuracy"].values[0]
        gains = []
        for k in budgets:
            row = rdf[rdf["budget"] == k]
            if len(row) == 1:
                gains.append(float(row["accuracy"].values[0]) - k0_acc)
        if gains:
            means.append(np.mean(gains) * 100)
            mins.append(min(gains) * 100)
            maxs.append(max(gains) * 100)
            labels.append(regime.title())

    y_pos = np.arange(len(labels))
    colors = [style.COLOR_GAIN if m >= 0 else style.VERMILLION for m in means]
    xerr = [
        [m - mn for m, mn in zip(means, mins)],
        [mx - m for m, mx in zip(means, maxs)],
    ]

    ax.barh(y_pos, means, xerr=xerr, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=style.FONT_SIZE)
    ax.axvline(0, color="black", linewidth=0.8)

    for i, m in enumerate(means):
        # Always annotate on the outer end of the bar
        ax.annotate(
            f"{m:+.1f}%",
            (m, i),
            textcoords="offset points",
            xytext=(8, 0) if m >= 0 else (-8, 0),
            fontsize=style.FONT_SIZE_ANNOT,
            ha="left" if m >= 0 else "right",
            va="center",
            clip_on=False,
        )

    ax.set_xlabel(
        f"\u0394Accuracy (%, mean over k\u2208{{{','.join(map(str, budgets))}}})",
        fontsize=style.FONT_SIZE,
    )
    ax.xaxis.set_major_formatter(style.pct_formatter())
    style.apply_style(ax)
    ax.invert_yaxis()

    # Pad x-axis so annotations don't clip
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - 5, xmax + 5)

    return fig, ax


# ── Selective classification ─────────────────────────────────────────


def plot_selective_classification(
    dnn_metrics: dict[str, float],
    cbm_metrics: dict[str, float],
    metric_names: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart comparing DNN vs CBM on selective classification metrics.

    Parameters
    ----------
    dnn_metrics : dict
        Metric name → value for the DNN model.
    cbm_metrics : dict
        Metric name → value for the CBM model.
    metric_names : list of str, optional
        Which metrics to show (default: all keys in cbm_metrics).
    ax : Axes, optional
        Existing axes to plot on.
    """
    fig, ax = _ensure_ax(ax, figsize=(7, 4))
    metric_names = metric_names or list(cbm_metrics.keys())

    metric_colors = {
        "selective_acc": style.COLOR_ACCURACY,
        "selective_accuracy": style.COLOR_ACCURACY,
        "coverage": style.COLOR_COVERAGE,
        "net_work": style.COLOR_NET_WORK,
        "net_work_automated": style.COLOR_NET_WORK,
    }

    x = np.arange(len(metric_names))
    width = 0.35
    dnn_vals = [dnn_metrics.get(m, 0) * 100 for m in metric_names]
    cbm_vals = [cbm_metrics.get(m, 0) * 100 for m in metric_names]

    colors = [metric_colors.get(m, style.BLUE) for m in metric_names]
    light_colors = [c + "80" for c in colors]  # alpha via hex

    ax.bar(
        x - width / 2,
        dnn_vals,
        width,
        color=light_colors,
        edgecolor=colors,
        label="DNN",
    )
    ax.bar(x + width / 2, cbm_vals, width, color=colors, label="CBM")

    for i, (d, c) in enumerate(zip(dnn_vals, cbm_vals)):
        ax.annotate(
            f"{d:.1f}%",
            (i - width / 2, d),
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_ANNOT,
        )
        ax.annotate(
            f"{c:.1f}%",
            (i + width / 2, c),
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_ANNOT,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metric_names], fontsize=style.FONT_SIZE
    )
    ax.yaxis.set_major_formatter(style.pct_formatter())
    ax.legend(fontsize=style.FONT_SIZE_LEGEND)
    style.apply_style(ax)

    return fig, ax


# ── Alignment comparison ─────────────────────────────────────────────


def plot_alignment_comparison(
    results: dict[str, dict[str, float]],
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of CBM vs aligned CBM gain.

    Parameters
    ----------
    results : dict
        Keys are dataset names (e.g. ``"ideal"``, ``"subconcept"``).
        Values are dicts with ``"cbm_gain"`` and ``"aligned_gain"`` (as fractions).
    ax : Axes, optional
        Existing axes to plot on.
    """
    fig, ax = _ensure_ax(ax, figsize=(7, 3))

    labels = list(results.keys())
    cbm_gains = [results[k]["cbm_gain"] * 100 for k in labels]
    aligned_gains = [results[k]["aligned_gain"] * 100 for k in labels]

    y = np.arange(len(labels))
    height = 0.35

    ax.barh(y - height / 2, cbm_gains, height, color=style.COLOR_GAIN, label="CBM")
    ax.barh(
        y + height / 2, aligned_gains, height, color=style.GREY, label="Aligned CBM"
    )

    for i, (c, a) in enumerate(zip(cbm_gains, aligned_gains)):
        ax.annotate(
            f"{c:+.1f}%",
            (c, i - height / 2),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            fontsize=style.FONT_SIZE_ANNOT,
        )
        ax.annotate(
            f"{a:+.1f}%",
            (a, i + height / 2),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            fontsize=style.FONT_SIZE_ANNOT,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([name.title() for name in labels], fontsize=style.FONT_SIZE)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Gain at k=3 (%)", fontsize=style.FONT_SIZE)
    ax.xaxis.set_major_formatter(style.pct_formatter())
    ax.legend(fontsize=style.FONT_SIZE_LEGEND)
    style.apply_style(ax)

    return fig, ax


# ── Concept discovery ────────────────────────────────────────────────


def plot_concept_discovery(
    ideal_results: pd.DataFrame,
    subconcept_results: pd.DataFrame,
    dnn_accuracy: float,
    budgets: list[int] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Clustered bar chart of ideal vs subconcept accuracy across budgets.

    Parameters
    ----------
    ideal_results : DataFrame
        Must have columns ``budget`` and ``accuracy``.
    subconcept_results : DataFrame
        Same format as ideal_results.
    dnn_accuracy : float
        DNN baseline accuracy (drawn as horizontal line).
    budgets : list of int, optional
        Which budgets to show (default: ``[0, 1, 3]``).
    ax : Axes, optional
        Existing axes to plot on.
    """
    fig, ax = _ensure_ax(ax, figsize=(7, 4))
    budgets = budgets or [0, 1, 3]

    ideal_accs = []
    sub_accs = []
    for k in budgets:
        ideal_row = ideal_results[ideal_results["budget"] == k]
        sub_row = subconcept_results[subconcept_results["budget"] == k]
        ideal_accs.append(
            float(ideal_row["accuracy"].values[0]) * 100 if len(ideal_row) else 0
        )
        sub_accs.append(
            float(sub_row["accuracy"].values[0]) * 100 if len(sub_row) else 0
        )

    x = np.arange(len(budgets))
    width = 0.35

    ax.bar(
        x - width / 2, ideal_accs, width, color=style.BLUE, label="Ideal (7 concepts)"
    )
    ax.bar(
        x + width / 2,
        sub_accs,
        width,
        color=style.BLUE_LIGHT,
        label="Subconcept (12 concepts)",
    )

    ax.axhline(
        dnn_accuracy * 100,
        color=style.COLOR_BASELINE,
        linestyle="--",
        linewidth=1.5,
        label="DNN baseline",
    )

    for i, (a, b) in enumerate(zip(ideal_accs, sub_accs)):
        ax.annotate(
            f"{a:.1f}%",
            (i - width / 2, a),
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_ANNOT,
            xytext=(0, 3),
            textcoords="offset points",
            clip_on=False,
        )
        ax.annotate(
            f"{b:.1f}%",
            (i + width / 2, b),
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_ANNOT,
            xytext=(0, 3),
            textcoords="offset points",
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in budgets], fontsize=style.FONT_SIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=style.FONT_SIZE)
    ax.yaxis.set_major_formatter(style.pct_formatter())

    # Pad y-axis for annotations
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 3)

    ax.legend(fontsize=style.FONT_SIZE_LEGEND)
    style.apply_style(ax)

    return fig, ax
