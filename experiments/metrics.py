from __future__ import annotations

__all__ = ["compute_selective_metric"]

import numpy as np

_ABSTAIN = -10


def compute_selective_metric(pred_probs, y_true, abstention_threshold=0.5):
    """Compute selective accuracy and coverage for a confidence threshold.

    Predictions are classified as positive (``P(y=1) > 1 - abstention_threshold``),
    negative (``P(y=1) < abstention_threshold``), or abstentions
    (``P(y=1)`` in ``[abstention_threshold, 1 - abstention_threshold]``).
    Abstained predictions are excluded from the accuracy calculation.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probabilities of the positive class, shape ``(N,)``.
    y_true : np.ndarray
        Ground-truth binary labels, shape ``(N,)``.
    abstention_threshold : float, optional
        Confidence threshold in ``[0, 0.5]``.  Default ``0.5`` (no
        abstentions).

    Returns
    -------
    dict
        ``{"coverage": float, "selective_accuracy": float, "abstention_threshold": float}``
        where *coverage* is the fraction of non-abstained predictions and
        *selective_accuracy* is accuracy on the non-abstained subset.
    """
    selected_y = pred_probs.copy()
    selected_y[pred_probs < abstention_threshold] = 0
    selected_y[(pred_probs >= abstention_threshold) & (pred_probs <= 1 - abstention_threshold)] = _ABSTAIN
    selected_y[pred_probs > 1 - abstention_threshold] = 1

    abstain = selected_y == _ABSTAIN

    if not np.any(~abstain):
        coverage = 0.0
        selective_accuracy = 1.0
    else:
        coverage = (~abstain).mean()
        selective_accuracy = (selected_y[~abstain] == y_true[~abstain]).mean()

    return {"coverage": coverage, "selective_accuracy": selective_accuracy, "abstention_threshold": abstention_threshold}
