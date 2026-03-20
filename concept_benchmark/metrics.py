from __future__ import annotations

__all__ = ["compute_selective_metric"]

import numpy as np


def compute_selective_metric(pred_probs, y_true, abstention_threshold=0.5):
    selected_y = pred_probs.copy()
    selected_y[pred_probs < abstention_threshold] = 0
    selected_y[
        (pred_probs >= abstention_threshold) & (pred_probs <= 1 - abstention_threshold)
    ] = np.nan  # abstain
    selected_y[pred_probs > 1 - abstention_threshold] = 1

    abstain = np.isnan(selected_y)

    if not np.any(~abstain):
        coverage = 0.0
        selective_accuracy = 1.0
    else:
        coverage = (~abstain).mean()
        selective_accuracy = (selected_y[~abstain] == y_true[~abstain]).mean()

    return {
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "abstention_threshold": abstention_threshold,
    }
