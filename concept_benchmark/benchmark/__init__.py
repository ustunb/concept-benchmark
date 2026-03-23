"""Evaluation metrics and plotting for concept bottleneck model benchmarks.

Metrics::

    from concept_benchmark.benchmark import accuracy, gain, selective_accuracy

    acc = accuracy(y_pred, y_true)
    g = gain(y_pred, y_true, baseline_accuracy=0.87)

Plots::

    from concept_benchmark.benchmark import plot_intervention_curve

    fig, ax = plot_intervention_curve(results_df)
"""

from .metrics import (
    accuracy,
    coverage,
    delta_accuracy,
    gain,
    net_work_automated,
    selective_accuracy,
)
from .plots import (
    plot_alignment_comparison,
    plot_concept_discovery,
    plot_intervention_curve,
    plot_regime_comparison,
    plot_selective_classification,
)

__all__ = [
    # metrics
    "accuracy",
    "coverage",
    "delta_accuracy",
    "gain",
    "net_work_automated",
    "selective_accuracy",
    # plots
    "plot_alignment_comparison",
    "plot_concept_discovery",
    "plot_intervention_curve",
    "plot_regime_comparison",
    "plot_selective_classification",
]
