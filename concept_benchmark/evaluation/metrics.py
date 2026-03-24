"""Pure metric functions for evaluating concept bottleneck models.

All functions take numpy arrays and return floats. No model objects,
no datasets — just predictions and labels.

Decision-support metrics (robot benchmark):
    :func:`accuracy`, :func:`delta_accuracy`, :func:`gain`

Automation metrics (sudoku benchmark):
    :func:`selective_accuracy`, :func:`coverage`, :func:`net_work_automated`
"""

from __future__ import annotations

import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correct predictions.

    Parameters
    ----------
    y_pred : array of shape (N,)
        Predicted labels.
    y_true : array of shape (N,)
        Ground-truth labels.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if len(y_pred) == 0:
        return float("nan")
    return float((y_pred == y_true).mean())


def delta_accuracy(
    y_pred_after: np.ndarray,
    y_pred_before: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Improvement in accuracy from interventions.

    ΔAccuracy = accuracy(after) − accuracy(before)

    Parameters
    ----------
    y_pred_after : array of shape (N,)
        Predictions after interventions.
    y_pred_before : array of shape (N,)
        Predictions before interventions.
    y_true : array of shape (N,)
        Ground-truth labels.
    """
    return accuracy(y_pred_after, y_true) - accuracy(y_pred_before, y_true)


def gain(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    baseline_accuracy: float,
) -> float:
    """Gain over a baseline predictor.

    Gain = accuracy(predictions) − baseline_accuracy

    Parameters
    ----------
    y_pred : array of shape (N,)
        Predictions (typically after interventions).
    y_true : array of shape (N,)
        Ground-truth labels.
    baseline_accuracy : float
        Accuracy of the baseline model (e.g. DNN).
    """
    return accuracy(y_pred, y_true) - float(baseline_accuracy)


def selective_accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Accuracy on non-abstained samples.

    The model abstains on samples where ``max(confidence) < threshold``.
    Selective accuracy is computed only on the remaining samples.

    Parameters
    ----------
    y_pred : array of shape (N,)
        Predicted labels.
    y_true : array of shape (N,)
        Ground-truth labels.
    confidence : array of shape (N,)
        Confidence score per sample (e.g. max predicted probability).
    threshold : float
        Minimum confidence to make a prediction (default 0.5).
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    confidence = np.asarray(confidence)
    kept = confidence >= threshold
    if not kept.any():
        return float("nan")
    return float((y_pred[kept] == y_true[kept]).mean())


def coverage(
    confidence: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Fraction of samples where the model does not abstain.

    Parameters
    ----------
    confidence : array of shape (N,)
        Confidence score per sample.
    threshold : float
        Minimum confidence to make a prediction (default 0.5).
    """
    confidence = np.asarray(confidence)
    if len(confidence) == 0:
        return float("nan")
    return float((confidence >= threshold).mean())


def net_work_automated(
    confidence: np.ndarray,
    threshold: float,
    n_interventions: np.ndarray,
    n_concepts: int,
) -> float:
    """Net fraction of work automated after accounting for intervention cost.

    NetWorkAutomated = coverage − mean(n_interventions / n_concepts)

    A value near 1 means most work is automated with few interventions.
    A value near 0 or negative means interventions cost more than they save.

    Parameters
    ----------
    confidence : array of shape (N,)
        Confidence score per sample.
    threshold : float
        Abstention threshold.
    n_interventions : array of shape (N,)
        Number of concepts intervened on per sample.
    n_concepts : int
        Total number of concepts.
    """
    cov = coverage(confidence, threshold)
    n_interventions = np.asarray(n_interventions, dtype=float)
    avg_cost = float(n_interventions.mean()) / max(n_concepts, 1)
    return cov - avg_cost
