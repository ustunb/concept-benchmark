"""Smoke tests for concept_benchmark.evaluation.plots.

These verify that each plot function runs without error and returns
(fig, ax). No visual comparison — just crash-free execution.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import pandas as pd

from concept_benchmark.evaluation.plots import (
    plot_alignment_comparison,
    plot_concept_discovery,
    plot_intervention_curve,
    plot_regime_comparison,
    plot_selective_classification,
)


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


def test_plot_regime_comparison():
    rows = []
    for regime in ["baseline", "expert", "machine"]:
        for k in [0, 1, 2, 5]:
            acc = 0.78 + 0.14 * (k > 0) if regime == "baseline" else 0.78 - 0.1 * k
            rows.append({"regime": regime, "budget": k, "accuracy": acc})
    df = pd.DataFrame(rows)
    fig, ax = plot_regime_comparison(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_selective_classification():
    dnn = {"selective_acc": 0.82, "coverage": 0.055}
    cbm = {"selective_acc": 0.98, "coverage": 0.876}
    fig, ax = plot_selective_classification(dnn, cbm)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_alignment_comparison():
    results = {
        "ideal": {"cbm_gain": 0.102, "aligned_gain": -0.003},
        "subconcept": {"cbm_gain": 0.069, "aligned_gain": -0.080},
    }
    fig, ax = plot_alignment_comparison(results)
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


def test_regime_comparison_single_regime():
    rows = [{"regime": "baseline", "budget": 0, "accuracy": 0.78}]
    rows += [
        {"regime": "baseline", "budget": k, "accuracy": 0.78 + 0.05 * k}
        for k in [1, 2, 5]
    ]
    fig, ax = plot_regime_comparison(pd.DataFrame(rows))
    plt.close(fig)


def test_regime_comparison_many_regimes():
    rows = []
    for regime in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        for k in [0, 1, 2, 5]:
            rows.append({"regime": regime, "budget": k, "accuracy": 0.5 + 0.01 * k})
    fig, ax = plot_regime_comparison(pd.DataFrame(rows))
    plt.close(fig)


def test_regime_comparison_missing_budget_0():
    """Regime without budget=0 should be skipped, not crash."""
    rows = [{"regime": "a", "budget": 1, "accuracy": 0.9}]
    fig, ax = plot_regime_comparison(pd.DataFrame(rows))
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


def test_alignment_all_negative():
    results = {
        "a": {"cbm_gain": -0.05, "aligned_gain": -0.10},
        "b": {"cbm_gain": -0.02, "aligned_gain": -0.15},
    }
    fig, ax = plot_alignment_comparison(results)
    plt.close(fig)


def test_alignment_single_dataset():
    results = {"ideal": {"cbm_gain": 0.10, "aligned_gain": -0.003}}
    fig, ax = plot_alignment_comparison(results)
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
