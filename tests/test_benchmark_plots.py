"""Smoke tests for concept_benchmark.benchmark.plots.

These verify that each plot function runs without error and returns
(fig, ax). No visual comparison — just crash-free execution.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import pandas as pd

from concept_benchmark.benchmark.plots import (
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
