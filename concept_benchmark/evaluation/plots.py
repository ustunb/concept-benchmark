"""Reusable plotting functions for benchmark results.

Each function takes data (DataFrames or dicts) and returns ``(fig, ax)``.
Pass an existing ``ax`` to compose multiple plots on one figure.

Uses ``ax.bar_label()`` for bar charts to ensure labels are properly
placed above/beside bars without overlapping.

Example::

    from concept_benchmark.evaluation.plots import plot_intervention_curve
    import pandas as pd

    df = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.78, 0.92, 0.94]})
    fig, ax = plot_intervention_curve(df)
    fig.savefig("intervention_curve.png", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import style


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_ax(ax, figsize=(6.5, 4)):
    """Return (fig, ax), creating them if ax is None."""
    style.set_paper_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _pad_axes(ax, *, top=0.08, bottom=0.02, left=0.05, right=0.05):
    """Pad axes limits by a fraction of the current data range."""
    ymin, ymax = ax.get_ylim()
    yrange = max(ymax - ymin, 1)
    ax.set_ylim(ymin - yrange * bottom, ymax + yrange * top)

    xmin, xmax = ax.get_xlim()
    xrange = max(xmax - xmin, 1)
    ax.set_xlim(xmin - xrange * left, xmax + xrange * right)


def _lighten_color(color, alpha=0.4):
    """Return a lighter version of a color by blending with white."""
    rgb = mcolors.to_rgb(color)
    return tuple(c * alpha + 1.0 * (1 - alpha) for c in rgb)


def _pct_labels(values):
    """Format an array of percentage values as label strings."""
    return [f"{v:.1f}%" for v in values]


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
    values = results[metric].values * 100

    ax.plot(budgets, values, marker="o", color=color, linewidth=2, label=label)

    bval = None
    if baseline_accuracy is not None:
        bval = baseline_accuracy * 100
        ax.axhline(
            bval,
            color=style.COLOR_BASELINE,
            linestyle="--",
            linewidth=1.5,
            label="DNN baseline",
        )

    ax.set_xlabel("Intervention budget (k)", fontsize=style.FONT_SIZE)
    ax.set_ylabel(metric.replace("_", " ").title() + " (%)", fontsize=style.FONT_SIZE)
    ax.yaxis.set_major_formatter(style.pct_formatter())
    _pad_axes(ax, top=0.12, bottom=0.08, left=0.08, right=0.05)
    style.apply_style(ax)

    if label or baseline_accuracy is not None:
        ax.legend(fontsize=style.FONT_SIZE_LEGEND, loc="upper left", framealpha=0.9)

    # Tight y-range: small margin around actual data (no empty space).
    all_y = list(values)
    if bval is not None:
        all_y.append(bval)
    data_min, data_max = min(all_y), max(all_y)
    margin = max((data_max - data_min) * 0.15, 2)
    ax.set_ylim(data_min - margin * 1.5, data_max + margin * 3)

    # If-then annotation placement per point:
    # - If next value is much higher → place below (line goes up, label out of the way)
    # - If next value is similar/lower → place above
    # - Last point → place right
    # - If label would sit on the baseline dashed line → push away
    n = len(budgets)
    for i, (bx, by) in enumerate(zip(budgets, values)):
        if i < n - 1:
            next_val = values[i + 1]
            if next_val - by > 3:
                xytext, ha, va = (0, -12), "center", "top"
            else:
                xytext, ha, va = (0, 8), "center", "bottom"
        else:
            xytext, ha, va = (10, 0), "left", "center"

        # If annotation is close to baseline, push it away from the line
        if bval is not None and abs(by - bval) < 2:
            if by <= bval:
                xytext, ha, va = (0, -12), "center", "top"
            else:
                xytext, ha, va = (0, 14), "center", "bottom"

        ax.annotate(
            f"{by:.1f}%",
            (bx, by),
            textcoords="offset points",
            xytext=xytext,
            fontsize=style.FONT_SIZE_ANNOT,
            ha=ha,
            va=va,
            color="black",
        )

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

    means, labels = [], []
    for regime in regimes:
        rdf = regime_df[regime_df["regime"] == regime]
        k0_rows = rdf[rdf["budget"] == 0]
        if len(k0_rows) == 0:
            continue
        k0_acc = k0_rows["accuracy"].values[0]
        gains = []
        for k in budgets:
            row = rdf[rdf["budget"] == k]
            if len(row) == 1:
                gains.append(float(row["accuracy"].values[0]) - k0_acc)
        if gains:
            means.append(np.mean(gains) * 100)
            labels.append(regime.title())

    if not means:
        style.apply_style(ax)
        return fig, ax

    y_pos = np.arange(len(labels))
    colors = [style.COLOR_POSITIVE if m >= 0 else style.COLOR_NEGATIVE for m in means]

    bars = ax.barh(y_pos, means, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=style.FONT_SIZE)
    ax.axvline(0, color=style.GREY, linewidth=1.0, linestyle="--", alpha=0.5)

    ax.bar_label(
        bars,
        labels=[f"{m:+.1f}%" for m in means],
        padding=5,
        fontsize=style.FONT_SIZE_ANNOT,
    )

    ax.set_xlabel(
        f"\u0394Accuracy (%, mean over k={{{','.join(map(str, budgets))}}})",
        fontsize=style.FONT_SIZE,
    )
    # Remove x-axis tick labels — just show the bars
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    style.apply_style(ax)
    ax.invert_yaxis()
    _pad_axes(ax, top=0.0, bottom=0.0, left=0.12, right=0.10)

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
    fig, ax = _ensure_ax(ax, figsize=(6.5, 4))
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
    light = [_lighten_color(c) for c in colors]

    bars_dnn = ax.bar(
        x - width / 2, dnn_vals, width, color=light, edgecolor=colors, label="DNN"
    )
    bars_cbm = ax.bar(x + width / 2, cbm_vals, width, color=colors, label="CBM")

    ax.bar_label(
        bars_dnn,
        labels=_pct_labels(dnn_vals),
        padding=3,
        fontsize=style.FONT_SIZE_ANNOT,
    )
    ax.bar_label(
        bars_cbm,
        labels=_pct_labels(cbm_vals),
        padding=3,
        fontsize=style.FONT_SIZE_ANNOT,
    )

    _label_map = {
        "selective_acc": "Selective\nAccuracy",
        "selective_accuracy": "Selective\nAccuracy",
        "coverage": "Coverage",
        "net_work": "NetWork\nAutomated",
        "net_work_automated": "NetWork\nAutomated",
    }
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_label_map.get(m, m.replace("_", " ").title()) for m in metric_names],
        fontsize=style.FONT_SIZE_TICK,
    )
    ax.yaxis.set_major_formatter(style.pct_formatter())
    _pad_axes(ax, top=0.12, bottom=0.0, left=0.0, right=0.0)
    legend_handles = [
        Patch(facecolor="#CCCCCC", edgecolor="#999999", label="DNN"),
        Patch(facecolor="#666666", edgecolor="#444444", label="CBM"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=style.FONT_SIZE_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        framealpha=0.9,
    )
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
    fig, ax = _ensure_ax(ax, figsize=(6.5, 3))

    labels = list(results.keys())
    cbm_gains = [results[k]["cbm_gain"] * 100 for k in labels]
    aligned_gains = [results[k]["aligned_gain"] * 100 for k in labels]

    y = np.arange(len(labels))
    height = 0.35

    bars_cbm = ax.barh(
        y + height / 2,
        cbm_gains,
        height,
        color=style.COLOR_GAIN,
        edgecolor="white",
        linewidth=0.5,
        label="CBM",
    )
    bars_al = ax.barh(
        y - height / 2,
        aligned_gains,
        height,
        color=style.COLOR_GAIN,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.5,
        label="Aligned CBM",
    )

    ax.bar_label(
        bars_cbm,
        labels=[f"{v:+.1f}%" for v in cbm_gains],
        padding=5,
        fontsize=style.FONT_SIZE_ANNOT,
        fontweight="bold",
    )
    ax.bar_label(
        bars_al,
        labels=[f"{v:+.1f}%" for v in aligned_gains],
        padding=5,
        fontsize=style.FONT_SIZE_ANNOT,
        fontweight="bold",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [name.title() for name in labels],
        fontsize=style.FONT_SIZE,
        fontfamily="monospace",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Gain in Accuracy over DNN", fontsize=style.FONT_SIZE)
    ax.xaxis.set_major_formatter(style.pct_formatter())
    _pad_axes(ax, top=0.0, bottom=0.0, left=0.10, right=0.10)
    ax.legend(
        fontsize=style.FONT_SIZE_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        framealpha=0.9,
    )
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
    fig, ax = _ensure_ax(ax, figsize=(6.5, 4))
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

    bars_ideal = ax.bar(
        x - width / 2,
        ideal_accs,
        width,
        color=style.BLUE,
        edgecolor="white",
        linewidth=0.5,
        label="True Concepts",
    )
    bars_sub = ax.bar(
        x + width / 2,
        sub_accs,
        width,
        color=style.BLUE_LIGHT,
        edgecolor="white",
        linewidth=0.5,
        label="Human Concepts",
    )

    baseline_y = dnn_accuracy * 100
    ax.axhline(
        baseline_y,
        color=style.COLOR_BASELINE,
        linestyle="--",
        linewidth=1.5,
    )
    # Inline label on the dashed line (right side, like the paper)
    ax.text(
        1.0,
        baseline_y,
        f"  DNN ({baseline_y:.1f}%)",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=style.FONT_SIZE_ANNOT,
        color=style.GREY,
        clip_on=False,
    )

    labels_i = ax.bar_label(
        bars_ideal,
        labels=_pct_labels(ideal_accs),
        padding=3,
        fontsize=style.FONT_SIZE_ANNOT,
    )
    labels_s = ax.bar_label(
        bars_sub,
        labels=_pct_labels(sub_accs),
        padding=3,
        fontsize=style.FONT_SIZE_ANNOT,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in budgets], fontsize=style.FONT_SIZE)
    ax.set_xlabel("CBM with Interventions", fontsize=style.FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=style.FONT_SIZE)
    ax.yaxis.set_major_formatter(style.pct_formatter())

    # Y-axis starts at 0, ceiling at max(data+padding, 100%)
    all_vals = ideal_accs + sub_accs
    y_ceil = max(max(all_vals) + 8, 100)
    ax.set_ylim(0, y_ceil)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        fontsize=style.FONT_SIZE_LEGEND,
        framealpha=0.9,
    )
    for txt in leg.get_texts():
        txt.set_fontfamily("monospace")
    style.apply_style(ax)

    # If a label crosses the baseline dashed line, nudge it just above.
    # bar_label uses an offset transform, so we convert the needed gap
    # from data units to points and increase the y-offset.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt in list(labels_i) + list(labels_s):
        bb = txt.get_window_extent(renderer)
        bb_data = ax.transData.inverted().transform(bb)
        text_ymin = bb_data[0][1]
        text_ymax = bb_data[1][1]
        if text_ymin - 0.5 <= baseline_y <= text_ymax + 0.5:
            gap_needed = baseline_y - text_ymin + 1.0
            p1 = ax.transData.transform((0, 0))
            p2 = ax.transData.transform((0, gap_needed))
            extra_pts = p2[1] - p1[1]
            x_off, y_off = txt.get_position()
            txt.set_position((x_off, y_off + extra_pts))

    return fig, ax


# ── Model comparison ─────────────────────────────────────────────────

# Per-model colors (colorblind-safe)
_MODEL_COLORS = {
    "CBM": "#0072B2",
    "CEM": "#E69F00",
    "ProbCBM": "#009E73",
}


def plot_model_comparison(
    results: dict[tuple[str, str], pd.DataFrame],
    dnn_accuracy: float,
    budgets: list[int] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart comparing models × concept sets across budgets.

    Each budget group has one bar per (model, concept_set) pair.
    Same model = same color; true concepts = full color,
    human concepts = lighter shade.

    Parameters
    ----------
    results : dict
        Keys are ``(model_name, concept_set)`` tuples, e.g.
        ``("CBM", "true_concepts")``. Values are DataFrames with
        columns ``budget`` and ``accuracy``.
    dnn_accuracy : float
        DNN baseline accuracy (drawn as horizontal line).
    budgets : list of int, optional
        Which budgets to show (default: ``[0, 1, 3]``).
    """
    fig, ax = _ensure_ax(ax, figsize=(14, 5))
    budgets = budgets or [0, 1, 3]
    concept_keys = ["true_concepts", "human_concepts"]

    # Discover models present in results
    models = []
    for model_name, cset in results:
        if model_name not in models:
            models.append(model_name)

    n_models = len(models)
    n_budgets = len(budgets)
    # Layout: within each budget group, model pairs with a small gap between models
    bar_w = 0.09
    pair_gap = 0.01  # gap between true/human within a model
    model_gap = 0.06  # extra gap between different models
    x = np.arange(n_budgets)

    baseline_y = dnn_accuracy * 100

    # Compute total group width to center the bars
    group_width = (
        n_models * 2 * bar_w
        + n_models * pair_gap
        + (n_models - 1) * model_gap
    )

    for m_idx, model_name in enumerate(models):
        base_color = _MODEL_COLORS.get(model_name, f"C{m_idx}")
        light_color = _lighten_color(base_color, alpha=0.45)

        for c_idx, ckey in enumerate(concept_keys):
            key = (model_name, ckey)
            if key not in results:
                continue
            df = results[key]
            accs = []
            for k in budgets:
                row = df[df["budget"] == k]
                accs.append(
                    float(row["accuracy"].values[0]) * 100 if len(row) else 0
                )

            # Position: model block offset + bar within pair
            block_offset = m_idx * (2 * bar_w + pair_gap + model_gap)
            bar_offset = block_offset + c_idx * (bar_w + pair_gap)
            offset = bar_offset - group_width / 2 + bar_w / 2

            is_true = ckey == "true_concepts"
            color = base_color if is_true else light_color

            label = None
            if c_idx == 0:
                label = f"{model_name} (true)"
            elif c_idx == 1:
                label = f"{model_name} (human)"

            bars = ax.bar(
                x + offset,
                accs,
                bar_w,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
            ax.bar_label(
                bars,
                labels=_pct_labels(accs),
                padding=3,
                fontsize=style.FONT_SIZE_ANNOT - 2,
            )

    ax.axhline(baseline_y, color=style.GREY, linestyle="--", linewidth=1.5)
    ax.text(
        1.0,
        baseline_y,
        f"  DNN ({baseline_y:.1f}%)",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=style.FONT_SIZE_ANNOT,
        color=style.GREY,
        clip_on=False,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in budgets], fontsize=style.FONT_SIZE)
    ax.set_xlabel("Intervention budget", fontsize=style.FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=style.FONT_SIZE)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(style.pct_formatter())
    style.apply_style(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(models),
        fontsize=style.FONT_SIZE_LEGEND - 1,
        framealpha=0.9,
        columnspacing=0.8,
    )
    fig.tight_layout()

    return fig, ax
