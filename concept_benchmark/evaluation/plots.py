"""Reusable plotting functions for benchmark results.

Each function takes data (DataFrames, dicts or arrays) and returns ``(fig, ax)``; pass an existing ``ax`` to
compose plots. Results tables may hold several runs: rows that share a budget and group are averaged, and a
``seed`` column gives the mean with a standard-error band.

Outcome plots show what happens (``plot_intervention_curve``, ``plot_intervention_heatmap``,
``plot_alignment_comparison``, ``plot_automation``); diagnostic plots show why (``plot_concept_report``,
``plot_answer_reliance``, ``plot_confidence``).

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
    """Format an array of percentage values as label strings (empty for missing values)."""
    return ["" if np.isnan(v) else f"{v:.1f}%" for v in values]


def _summarize_runs(frame: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    """Mean and standard error of `value` over the rows that share `by` (one row per run)."""
    grouped = frame.groupby(by, sort=False)[value]
    summary = grouped.mean().rename("mean").to_frame()
    summary["se"] = grouped.sem().fillna(0.0)
    return summary.reset_index()


SERIES_COLORS = [
    style.BLUE,
    style.VERMILLION,
    style.GREEN,
    style.PURPLE,
    style.BLUE_LIGHT,
    style.RED,
]


# ── Intervention curve ───────────────────────────────────────────────


def plot_intervention_curve(
    results: pd.DataFrame,
    metric: str = "accuracy",
    baseline_accuracy: float | list[float] | None = None,
    label: str | None = None,
    color: str | None = None,
    group: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of a metric against the intervention budget *k*.

    Parameters
    ----------
    results : DataFrame
        Must have columns ``budget`` and the column named by *metric*. Rows that share a budget (several
        runs) are averaged and drawn with a standard-error band.
    metric : str
        Column name to plot on the y-axis (default ``"accuracy"``).
    baseline_accuracy : float or list of float, optional
        Accuracy of the DNN (one value per run); drawn as a dashed line at its mean.
    label : str, optional
        Legend label for the line (single series).
    color : str, optional
        Line color (single series; default: blue).
    group : str, optional
        Column that splits the rows into one line each (e.g. ``"model_family"`` or ``"concept_source"``).
        The largest budget of each line is drawn at the same position and labeled ``max`` when lines differ.
    ax : Axes, optional
        Existing axes to plot on.
    """
    fig, ax = _ensure_ax(ax)
    if group is None:
        series = [(label, results, color or style.COLOR_ACCURACY)]
    else:
        names = list(dict.fromkeys(results[group]))
        series = [
            (
                str(name),
                results[results[group] == name],
                SERIES_COLORS[i % len(SERIES_COLORS)],
            )
            for i, name in enumerate(names)
        ]

    if results.empty or results[metric].isna().all():
        raise ValueError(f"no {metric!r} values to plot")
    summaries = [
        (
            name,
            _summarize_runs(rows, ["budget"], metric).sort_values("budget"),
            line_color,
        )
        for name, rows, line_color in series
    ]
    budget_lists = [list(summary["budget"]) for _, summary, _ in summaries]
    is_same = all(budgets == budget_lists[0] for budgets in budget_lists)
    is_same_but_largest = not is_same and all(
        budgets[:-1] == budget_lists[0][:-1] for budgets in budget_lists
    )
    if (
        is_same or is_same_but_largest
    ):  # each line's largest budget shares the last position
        positions = {
            budget: i for budgets in budget_lists for i, budget in enumerate(budgets)
        }
        tick_labels = [str(b) for b in budget_lists[0]]
        if is_same_but_largest:
            tick_labels[-1] = "max"
    else:  # lines with different budgets: one position per budget that occurs
        every_budget = sorted(
            {budget for budgets in budget_lists for budget in budgets}
        )
        positions = {budget: i for i, budget in enumerate(every_budget)}
        tick_labels = [str(b) for b in every_budget]

    plotted = []
    for name, summary, line_color in summaries:
        x = np.array([positions[budget] for budget in summary["budget"]])
        values, errors = (
            summary["mean"].to_numpy() * 100,
            summary["se"].to_numpy() * 100,
        )
        ax.plot(x, values, marker="o", color=line_color, linewidth=2, label=name)
        if errors.any():
            ax.fill_between(
                x,
                values - errors,
                values + errors,
                color=line_color,
                alpha=0.2,
                linewidth=0,
            )
        plotted.append((x, values))
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels)

    baseline = None
    if baseline_accuracy is not None:
        baseline = float(np.mean(baseline_accuracy)) * 100
        ax.axhline(
            baseline,
            color=style.COLOR_BASELINE,
            linestyle="--",
            linewidth=1.5,
            label="DNN baseline",
        )

    ax.set_xlabel("Intervention budget (k)", fontsize=style.FONT_SIZE)
    ax.set_ylabel(metric.replace("_", " ").title() + " (%)", fontsize=style.FONT_SIZE)
    ax.yaxis.set_major_formatter(style.pct_formatter())
    style.apply_style(ax)
    if group is not None or label or baseline is not None:
        ax.legend(fontsize=style.FONT_SIZE_LEGEND, loc="best", framealpha=0.9)

    all_y = np.concatenate(
        [values for _, values in plotted]
        + ([[baseline]] if baseline is not None else [])
    )
    margin = max((all_y.max() - all_y.min()) * 0.15, 2)
    ax.set_ylim(all_y.min() - margin * 1.5, all_y.max() + margin * 3)
    ax.set_xlim(-0.3, len(tick_labels) - 0.5)

    if len(plotted) == 1:  # a single line has room for its values
        x, values = plotted[0]
        for i, (bx, by) in enumerate(zip(x, values)):
            if i == len(x) - 1:
                xytext, ha, va = (10, 0), "left", "center"
            elif values[i + 1] - by > 3:  # the line rises: keep the label below it
                xytext, ha, va = (0, -12), "center", "top"
            else:
                xytext, ha, va = (0, 8), "center", "bottom"
            if (
                baseline is not None and abs(by - baseline) < 2
            ):  # keep clear of the dashed line
                xytext, ha, va = (
                    ((0, -12), "center", "top")
                    if by <= baseline
                    else ((0, 14), "center", "bottom")
                )
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


# ── Intervention heatmap ─────────────────────────────────────────────


def plot_intervention_heatmap(
    results: pd.DataFrame,
    rows: str = "model_family",
    columns: tuple[str, ...] = ("concept_source", "intervention_source"),
    metric: str = "accuracy",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of the change in accuracy that interventions bring, per model and setting.

    Each cell is the mean over runs of ``metric`` at the budgets above zero minus ``metric`` at budget zero:
    blue cells are settings where interventions help, red cells where they hurt.

    Parameters
    ----------
    results : DataFrame
        Must have columns ``budget``, *metric*, *rows* and every name in *columns*; a ``seed`` column
        separates runs.
    rows : str
        Column whose values become the rows (default ``"model_family"``).
    columns : tuple of str
        Columns whose value combinations become the heatmap columns.
    """
    columns = list(columns)
    run_keys = [rows, *columns] + (["seed"] if "seed" in results.columns else [])
    changes = []
    for key, run in results.groupby(run_keys, sort=False, dropna=False):
        before = run.loc[run["budget"] == 0, metric]
        after = run.loc[run["budget"] != 0, metric]
        if len(before) and len(after):
            changes.append((*key, 100 * (after.mean() - before.iloc[0])))
    if not changes:
        raise ValueError("no run has both a budget of 0 and a larger budget")
    table = (
        pd.DataFrame(changes, columns=[*run_keys, "change"])
        .groupby([rows, *columns], sort=False, dropna=False)["change"]
        .mean()
        .unstack(columns)
    )
    fig, ax = _ensure_ax(
        ax, figsize=(1.3 * table.shape[1] + 2.5, 0.7 * table.shape[0] + 1.8)
    )
    limit = max(float(np.nanmax(np.abs(table.to_numpy()))), 1.0)
    image = ax.imshow(
        table.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
    )
    for (r, c), value in np.ndenumerate(table.to_numpy()):
        if not np.isnan(value):
            ax.text(
                c,
                r,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                fontsize=style.FONT_SIZE_ANNOT,
                color="white" if abs(value) > 0.6 * limit else "black",
            )
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(
        [
            " / ".join(map(str, c)) if isinstance(c, tuple) else str(c)
            for c in table.columns
        ],
        fontsize=style.FONT_SIZE_TICK,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([str(r) for r in table.index], fontsize=style.FONT_SIZE)
    ax.grid(False)
    fig.colorbar(
        image, ax=ax, label="\u0394 " + metric.replace("_", " ").title() + " (%)"
    )
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
    results: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Bar chart of a constrained against an unconstrained model, before and after interventions.

    Parameters
    ----------
    results : DataFrame
        One row per run with columns ``concepts`` (concept set), ``model`` (e.g. ``"CBM"`` and
        ``"Constrained CBM"``) and ``accuracy_before``; an ``accuracy_after`` column adds a second panel
        with the accuracy after interventions. Several rows per concept set and model are averaged and
        drawn with standard-error bars.
    ax : Axes, optional
        Existing axes to plot on (single panel only).
    """
    if results.empty:
        raise ValueError("no alignment results to plot")
    metrics = [m for m in ("accuracy_before", "accuracy_after") if m in results.columns]
    titles = {
        "accuracy_before": "Accuracy before interventions",
        "accuracy_after": "Accuracy after interventions",
    }
    style.set_paper_style()
    if ax is not None:
        fig, axes = ax.figure, [ax]
        metrics = metrics[:1]
    else:
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(5.5 * len(metrics), 3), sharey=True, squeeze=False
        )
        axes = list(axes[0])
    concept_sets = list(dict.fromkeys(results["concepts"]))
    models = list(dict.fromkeys(results["model"]))
    height = 0.8 / len(models)
    for panel, metric in zip(axes, metrics):
        summary = _summarize_runs(results, ["concepts", "model"], metric).set_index(
            ["concepts", "model"]
        )
        for m, model in enumerate(models):
            means = (
                np.array(
                    [summary["mean"].get((c, model), np.nan) for c in concept_sets]
                )
                * 100
            )
            errors = (
                np.array([summary["se"].get((c, model), 0.0) for c in concept_sets])
                * 100
            )
            y = (
                np.arange(len(concept_sets))
                + (len(models) - 1) / 2 * height
                - m * height
            )
            bars = panel.barh(
                y,
                means,
                height,
                xerr=errors if errors.any() else None,
                color=style.COLOR_GAIN,
                alpha=1.0 if m == 0 else 0.35,
                edgecolor="white",
                linewidth=0.5,
                label=model,
            )
            panel.bar_label(
                bars,
                labels=_pct_labels(means),
                padding=5,
                fontsize=style.FONT_SIZE_ANNOT,
            )
        panel.set_yticks(np.arange(len(concept_sets)))
        panel.set_yticklabels(
            concept_sets, fontsize=style.FONT_SIZE, fontfamily="monospace"
        )
        panel.set_xlabel(titles[metric], fontsize=style.FONT_SIZE)
        panel.xaxis.set_major_formatter(style.pct_formatter())
        low = float(results[metrics].min().min()) * 100
        panel.set_xlim(max(0.0, low - 10), 100)
        panel.invert_yaxis()
        style.apply_style(panel)
    axes[0].legend(
        fontsize=style.FONT_SIZE_LEGEND,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(models),
    )
    return fig, (axes[0] if len(axes) == 1 else np.array(axes))


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
        n_models * 2 * bar_w + n_models * pair_gap + (n_models - 1) * model_gap
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
                accs.append(float(row["accuracy"].values[0]) * 100 if len(row) else 0)

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


# ── Automation ───────────────────────────────────────────────────────


def plot_automation(
    results: pd.DataFrame,
    n_instances: int,
    n_concepts: int,
    baseline_coverage: float | list[float] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of coverage and net work automated against the number of concept checks allowed.

    Net work automated is the coverage minus the share of concepts a human checked, so the gap between the
    two lines is what the checks cost.

    Parameters
    ----------
    results : DataFrame
        Must have columns ``budget``, ``coverage_after`` and ``total_concept_checks`` (as written by the
        sudoku pipeline). Rows that share a budget (several runs) are averaged with a standard-error band.
    n_instances, n_concepts : int
        Number of test instances and of concepts per instance.
    baseline_coverage : float or list of float, optional
        Coverage of a model without interventions (e.g. the DNN), drawn as a dashed line.
    """
    fig, ax = _ensure_ax(ax)
    runs = results.assign(
        net_work=results["coverage_after"]
        - results["total_concept_checks"] / (n_instances * n_concepts)
    )
    budgets = sorted(runs["budget"].unique())
    x = np.arange(len(budgets))
    for column, name, line_color, marker in (
        ("coverage_after", "Coverage", style.COLOR_COVERAGE, "o"),
        ("net_work", "Net work automated", style.COLOR_NET_WORK, "s"),
    ):
        summary = _summarize_runs(runs, ["budget"], column).sort_values("budget")
        values, errors = (
            summary["mean"].to_numpy() * 100,
            summary["se"].to_numpy() * 100,
        )
        ax.plot(x, values, marker=marker, color=line_color, linewidth=2, label=name)
        if errors.any():
            ax.fill_between(
                x,
                values - errors,
                values + errors,
                color=line_color,
                alpha=0.2,
                linewidth=0,
            )
    if baseline_coverage is not None:
        ax.axhline(
            float(np.mean(baseline_coverage)) * 100,
            color=style.COLOR_BASELINE,
            linestyle="--",
            linewidth=1.5,
            label="DNN baseline",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["max" if b == n_concepts else str(b) for b in budgets])
    ax.set_xlabel("Concept checks per instance (k)", fontsize=style.FONT_SIZE)
    ax.set_ylabel("Share of instances (%)", fontsize=style.FONT_SIZE)
    ax.yaxis.set_major_formatter(style.pct_formatter())
    ax.legend(fontsize=style.FONT_SIZE_LEGEND, loc="best", framealpha=0.9)
    style.apply_style(ax)
    return fig, ax


# ── Diagnostics ──────────────────────────────────────────────────────


def plot_concept_report(
    mask: np.ndarray,
    concept_proba: np.ndarray,
    concept_answers: np.ndarray,
    concepts_true: np.ndarray,
    concept_names: list[str],
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Per concept: how accurate the detector is, how often it is intervened on, how accurate the intervener is.

    An intervener helps only if it is accurate on the concepts the model asks about, so a concept with a high
    share of interventions and a low intervener accuracy explains interventions that do not help. The inputs
    are the intervention records of the robot pipeline (``--dump-interventions``).

    Parameters
    ----------
    mask : array (n, m) of bool
        Concepts intervened on.
    concept_proba : array (n, m)
        Predicted concept probabilities before interventions.
    concept_answers : array (n, m)
        Concept values after interventions (the intervener's answers where ``mask`` is set).
    concepts_true : array (n, m)
        True concept values.
    concept_names : list of str
        One name per concept.
    """
    mask = np.asarray(mask, dtype=bool)
    truth = np.asarray(concepts_true) >= 0.5
    detector = ((np.asarray(concept_proba) >= 0.5) == truth).mean(axis=0) * 100
    asked = mask.sum(axis=0)
    share = asked / max(1, mask.sum()) * 100
    is_right = (np.asarray(concept_answers) >= 0.5) == truth
    intervener = (
        np.where(
            asked > 0, (is_right & mask).sum(axis=0) / np.maximum(asked, 1), np.nan
        )
        * 100
    )

    fig, ax = _ensure_ax(ax, figsize=(7, 0.45 * len(concept_names) + 1.8))
    y = np.arange(len(concept_names))
    height = 0.27
    for offset, values, name, bar_color in (
        (-height, detector, "Detector accuracy", style.COLOR_ACCURACY),
        (0.0, share, "Share of interventions", style.GREY),
        (height, intervener, "Intervener accuracy", style.COLOR_COVERAGE),
    ):
        ax.barh(
            y + offset,
            np.nan_to_num(values),
            height,
            color=bar_color,
            edgecolor="white",
            label=name,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        concept_names, fontsize=style.FONT_SIZE_TICK, fontfamily="monospace"
    )
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(style.pct_formatter())
    ax.invert_yaxis()
    ax.legend(
        fontsize=style.FONT_SIZE_LEGEND,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )
    style.apply_style(ax)
    return fig, ax


def plot_answer_reliance(
    results: pd.DataFrame,
    group: str = "model_family",
    metric: str = "accuracy",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Change in accuracy when interventions supply the model's own answers against the true values.

    Interventions with the model's own answers change no concept value, so any change in accuracy comes from
    the intervention mechanism. A model whose bars are similar does not rely on the concept values.

    Parameters
    ----------
    results : DataFrame
        Rows of the robot pipeline run with ``--intervention-sources self perfect``: columns ``budget``,
        *metric*, ``intervention_source`` and *group*; a ``seed`` column separates runs.
    group : str
        Column whose values get one pair of bars each (default ``"model_family"``).
    """
    fig, ax = _ensure_ax(ax)
    names = list(dict.fromkeys(results[group]))
    run_keys = [group, "intervention_source"] + (
        ["seed"] if "seed" in results.columns else []
    )
    changes = []
    for key, run in results.groupby(run_keys, sort=False, dropna=False):
        before = run.loc[run["budget"] == 0, metric]
        after = run.loc[run["budget"] != 0, metric]
        if len(before) and len(after):
            changes.append((*key, 100 * (after.mean() - before.iloc[0])))
    if not changes:
        raise ValueError("no run has both a budget of 0 and a larger budget")
    summary = _summarize_runs(
        pd.DataFrame(changes, columns=[*run_keys, "change"]),
        [group, "intervention_source"],
        "change",
    ).set_index([group, "intervention_source"])
    width = 0.38
    x = np.arange(len(names))
    for offset, source, name, bar_color in (
        (-width / 2, "self", "Own answers", style.GREY),
        (width / 2, "perfect", "True values", style.COLOR_ACCURACY),
    ):
        means = np.array([summary["mean"].get((n, source), np.nan) for n in names])
        errors = np.array([summary["se"].get((n, source), 0.0) for n in names])
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=errors if errors.any() else None,
            color=bar_color,
            edgecolor="white",
            label=name,
        )
        ax.bar_label(
            bars,
            labels=["" if np.isnan(m) else f"{m:+.1f}%" for m in means],
            padding=3,
            fontsize=style.FONT_SIZE_ANNOT,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(n) for n in names],
        fontsize=style.FONT_SIZE,
        **(
            {"rotation": 20, "ha": "right", "rotation_mode": "anchor"}
            if len(names) > 3
            else {}
        ),
    )
    ax.set_ylabel(
        "\u0394 " + metric.replace("_", " ").title() + " (%)", fontsize=style.FONT_SIZE
    )
    ax.legend(fontsize=style.FONT_SIZE_LEGEND, loc="best", framealpha=0.9)
    _pad_axes(ax, top=0.15, bottom=0.15, left=0.0, right=0.0)
    style.apply_style(ax)
    return fig, ax


def plot_confidence(
    prob_positive: np.ndarray,
    y_true: np.ndarray,
    abstention_threshold: float,
    class_names: tuple[str, str] = ("invalid", "valid"),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram of the predicted probability of the positive class, with the band where the model abstains.

    A selective classifier abstains when the probability lies in ``[t, 1 - t]``. Instances of one class that
    pile up inside the band are deferred no matter how the other class is handled. The inputs are saved by
    the sudoku pipeline (stage ``diagnose``).

    Parameters
    ----------
    prob_positive : array (n,)
        Predicted probability of the positive class.
    y_true : array (n,)
        True labels (0/1).
    abstention_threshold : float
        The threshold ``t`` fitted on the validation set.
    """
    fig, ax = _ensure_ax(ax)
    prob_positive, y_true = (
        np.asarray(prob_positive, dtype=float),
        np.asarray(y_true).astype(int),
    )
    bins = np.linspace(0, 1, 41)
    for label, bar_color in ((0, style.COLOR_NEGATIVE), (1, style.COLOR_ACCURACY)):
        ax.hist(
            prob_positive[y_true == label],
            bins=bins,
            color=bar_color,
            alpha=0.6,
            label=class_names[label],
        )
    ax.axvspan(
        abstention_threshold,
        1 - abstention_threshold,
        color=style.GREY,
        alpha=0.12,
        label="Abstains",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        f"Predicted probability of {class_names[1]}", fontsize=style.FONT_SIZE
    )
    ax.set_ylabel("Instances", fontsize=style.FONT_SIZE)
    ax.legend(fontsize=style.FONT_SIZE_LEGEND, loc="upper center", framealpha=0.9)
    style.apply_style(ax)
    return fig, ax
