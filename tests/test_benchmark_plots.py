"""Smoke tests for concept_benchmark.evaluation.plots.

These verify that each plot function runs without error and returns
(fig, ax). No visual comparison — just crash-free execution.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from concept_benchmark.evaluation.plots import (
    plot_alignment_comparison,
    plot_answer_reliance,
    plot_automation,
    plot_concept_discovery,
    plot_concept_report,
    plot_confidence,
    plot_intervention_curve,
    plot_intervention_heatmap,
    plot_selective_classification,
)


def _robot_runs(
    sources=("perfect",), families=("cbm", "cem"), seeds=(1, 2, 3)
) -> pd.DataFrame:
    """Results rows as the robot pipeline writes them, for several runs."""
    rng = np.random.default_rng(0)
    rows = []
    for family in families:
        for source in sources:
            for seed in seeds:
                for budget in (0, 1, 3, 12):
                    gain = 0.02 * budget if source == "perfect" else -0.01 * budget
                    rows.append(
                        {
                            "model_family": family,
                            "concept_source": "human_concepts",
                            "intervention_source": source,
                            "seed": seed,
                            "budget": budget,
                            "accuracy": 0.78 + gain + rng.normal(0, 0.005),
                        }
                    )
    return pd.DataFrame(rows)


def test_plot_intervention_curve():
    df = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.78, 0.92, 0.94]})
    fig, ax = plot_intervention_curve(df, baseline_accuracy=0.87)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_plot_intervention_curve_on_existing_ax():
    fig, ax = plt.subplots()
    df = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.78, 0.92, 0.94]})
    fig2, ax2 = plot_intervention_curve(df, ax=ax)
    assert fig2 is fig
    assert ax2 is ax
    plt.close(fig)


def test_plot_selective_classification():
    dnn = {"selective_acc": 0.82, "coverage": 0.055}
    cbm = {"selective_acc": 0.98, "coverage": 0.876}
    fig, ax = plot_selective_classification(dnn, cbm)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_concept_discovery():
    ideal = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.8673, 0.9734, 0.9767]})
    sub = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.7812, 0.9212, 0.9439]})
    fig, ax = plot_concept_discovery(ideal, sub, dnn_accuracy=0.8746)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── Edge cases ──────────────────────────────────────────────────────


def test_intervention_curve_single_point():
    df = pd.DataFrame({"budget": [0], "accuracy": [0.85]})
    fig, ax = plot_intervention_curve(df)
    plt.close(fig)


def test_intervention_curve_value_near_baseline():
    """Annotation should shift when value ≈ baseline."""
    df = pd.DataFrame({"budget": [0, 1], "accuracy": [0.87, 0.95]})
    fig, ax = plot_intervention_curve(df, baseline_accuracy=0.875)
    plt.close(fig)


def test_intervention_curve_wide_range():
    df = pd.DataFrame(
        {"budget": [0, 1, 3, 5, 10], "accuracy": [0.10, 0.50, 0.80, 0.95, 0.99]}
    )
    fig, ax = plot_intervention_curve(df)
    plt.close(fig)


def test_selective_classification_zero_values():
    dnn = {"selective_acc": 0.0, "coverage": 0.0}
    cbm = {"selective_acc": 0.95, "coverage": 0.9}
    fig, ax = plot_selective_classification(dnn, cbm)
    plt.close(fig)


def test_selective_classification_equal_values():
    dnn = {"selective_acc": 0.5, "coverage": 0.5}
    cbm = {"selective_acc": 0.5, "coverage": 0.5}
    fig, ax = plot_selective_classification(dnn, cbm)
    plt.close(fig)


def test_concept_discovery_single_budget():
    ideal = pd.DataFrame({"budget": [0], "accuracy": [0.87]})
    sub = pd.DataFrame({"budget": [0], "accuracy": [0.78]})
    fig, ax = plot_concept_discovery(ideal, sub, dnn_accuracy=0.87)
    plt.close(fig)


def test_concept_discovery_close_values():
    """Values within 1% of each other — annotations shouldn't overlap."""
    ideal = pd.DataFrame({"budget": [0], "accuracy": [0.860]})
    sub = pd.DataFrame({"budget": [0], "accuracy": [0.855]})
    fig, ax = plot_concept_discovery(ideal, sub, dnn_accuracy=0.865)
    plt.close(fig)


def test_intervention_curve_averages_runs_per_group():
    fig, ax = plot_intervention_curve(
        _robot_runs(), group="model_family", baseline_accuracy=[0.87, 0.88, 0.89]
    )
    assert [line.get_label() for line in ax.get_lines()[:2]] == ["cbm", "cem"]
    assert len(ax.collections) == 2  # one standard-error band per model
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0", "1", "3", "12"]
    plt.close(fig)


def test_intervention_curve_labels_unequal_largest_budgets_as_max():
    runs = _robot_runs(families=("cbm",))
    other = runs.assign(
        concept_source="true_concepts", budget=runs["budget"].replace(12, 7)
    )
    fig, ax = plot_intervention_curve(pd.concat([runs, other]), group="concept_source")
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0", "1", "3", "max"]
    plt.close(fig)


def test_intervention_heatmap_shows_the_change_from_no_interventions():
    runs = _robot_runs(sources=("perfect", "llm"))
    fig, ax = plot_intervention_heatmap(runs)
    values = ax.images[0].get_array()
    assert values.shape == (2, 2)  # two models x (one concept set x two interveners)
    assert (values[:, 0] > 0).all() and (values[:, 1] < 0).all()
    plt.close(fig)


def test_alignment_comparison_draws_one_panel_per_metric():
    rows = [
        {
            "concepts": c,
            "model": m,
            "seed": s,
            "accuracy_before": before,
            "accuracy_after": after,
        }
        for c in ("true_concepts", "human_concepts")
        for m, before, after in (("CBM", 0.84, 0.92), ("Constrained CBM", 0.90, 0.91))
        for s in (1, 2)
    ]
    fig, axes = plot_alignment_comparison(pd.DataFrame(rows))
    assert len(axes) == 2
    plt.close(fig)
    fig, ax = plot_alignment_comparison(
        pd.DataFrame(rows).drop(columns="accuracy_after")
    )
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_automation_shows_coverage_and_net_work():
    rows = [
        {
            "seed": s,
            "budget": b,
            "coverage_after": 0.8 + 0.01 * i,
            "total_concept_checks": 40 * b,
        }
        for s in (1, 2)
        for i, b in enumerate((0, 1, 3, 27))
    ]
    fig, ax = plot_automation(
        pd.DataFrame(rows), n_instances=200, n_concepts=27, baseline_coverage=0.03
    )
    coverage, net_work = ax.get_lines()[0].get_ydata(), ax.get_lines()[1].get_ydata()
    assert coverage[0] == net_work[0]  # no checks, no cost
    assert net_work[-1] < coverage[-1]
    plt.close(fig)


def test_concept_report_handles_concepts_never_asked():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, size=(50, 4))
    proba = rng.random((50, 4))
    mask = np.zeros((50, 4), dtype=bool)
    mask[:, 0] = True
    fig, ax = plot_concept_report(
        mask, proba, truth.astype(float), truth, ["a", "b", "c", "d"]
    )
    assert len(ax.patches) == 12  # three bars per concept
    plt.close(fig)


def test_answer_reliance_compares_own_answers_with_true_values():
    fig, ax = plot_answer_reliance(_robot_runs(sources=("self", "perfect")))
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Own answers",
        "True values",
    ]
    plt.close(fig)


def test_confidence_marks_the_abstention_band():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    fig, ax = plot_confidence(np.where(y == 1, 0.9, 0.05), y, abstention_threshold=0.2)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "invalid",
        "valid",
        "Abstains",
    ]
    plt.close(fig)
