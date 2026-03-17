from __future__ import annotations

__all__ = ["calc_metric"]

import numpy as np

_ABSTAIN = -10


def calc_metric(pred_probs, y_true, tau=0.5):
    """Compute selective accuracy and coverage for a confidence threshold.

    Predictions are classified as positive (``P(y=1) > 1 - tau``), negative
    (``P(y=1) < tau``), or abstentions (``P(y=1)`` in ``[tau, 1 - tau]``).
    Abstained predictions are excluded from the accuracy calculation.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probabilities of the positive class, shape ``(N,)``.
    y_true : np.ndarray
        Ground-truth binary labels, shape ``(N,)``.
    tau : float, optional
        Confidence threshold in ``[0, 0.5]``.  Default ``0.5`` (no
        abstentions).

    Returns
    -------
    dict
        ``{"coverage": float, "selective_accuracy": float, "tau": float}``
        where *coverage* is the fraction of non-abstained predictions and
        *selective_accuracy* is accuracy on the non-abstained subset.
    """
    selected_y = pred_probs.copy()
    selected_y[pred_probs < tau] = 0
    selected_y[(pred_probs >= tau) & (pred_probs <= 1 - tau)] = _ABSTAIN
    selected_y[pred_probs > 1 - tau] = 1

    abstain = selected_y == _ABSTAIN

    if not np.any(~abstain):
        coverage = 0.0
        selective_accuracy = 1.0
    else:
        coverage = (~abstain).mean()
        selective_accuracy = (selected_y[~abstain] == y_true[~abstain]).mean()

    return {"coverage": coverage, "selective_accuracy": selective_accuracy, "tau": tau}
