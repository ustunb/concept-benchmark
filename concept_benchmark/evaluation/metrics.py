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


# ── Interventions ─────────────────────────────────────────────────────


def intervention_metrics(
    mask: np.ndarray,
    concepts_before: np.ndarray,
    concepts_after: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    y_true: np.ndarray,
    accuracy_before: float,
) -> dict:
    """What an intervention did: its accuracy, and how many predictions and concepts it touched.

    Parameters
    ----------
    mask : bool array of shape (N, C)
        Concepts a human was asked about.
    concepts_before, concepts_after : arrays of shape (N, C)
        Concept probabilities the model predicted, and the values after the answers were put in.
    y_prob_before, y_prob_after : arrays of shape (N, K)
        Label probabilities before and after the intervention.
    y_true : array of shape (N,)
    accuracy_before : float
        Accuracy without interventions, for the gain.
    """
    mask = np.asarray(mask, dtype=bool)
    y_true = np.asarray(y_true).astype(int)
    y_pred_before = np.argmax(y_prob_before, axis=1)
    y_pred_after = np.argmax(y_prob_after, axis=1)
    edits = (np.asarray(concepts_before) >= 0.5) != (np.asarray(concepts_after) >= 0.5)
    n_samples = len(y_true)
    accuracy_after = float((y_pred_after == y_true).mean())
    return {
        "accuracy": accuracy_after,
        "accuracy_gain": accuracy_after - accuracy_before,
        "predictions_intervened_on": int(np.sum(np.any(mask, axis=1))),
        "interventions_rate": float(np.sum(np.any(mask, axis=1)) / n_samples),
        "predictions_changed": int(np.sum(y_pred_after != y_pred_before)),
        "avg_edits_per_intervention": float(edits.sum()) / n_samples,
        "total_concept_confirmations": int(mask.sum()),
        "total_concept_edits_made": int(edits.sum()),
    }


# ── Abstention: the paper's selective-classification protocol ────────

_TIE_MARGIN = 1e-6  # see abstention_threshold


def abstention_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    target_acc: float,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
    """Abstention threshold at which the kept predictions reach ``target_acc``, and their coverage.

    A model abstains on ``t <= p <= 1 - t`` (see :func:`selective_at`). The threshold is the largest ``t``
    whose kept predictions, scored with ``decision_threshold``, reach the target; fit it on validation
    predictions. Returns ``(None, None)`` when no threshold reaches the target.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    min_prob = np.minimum(prob_pos, 1.0 - prob_pos)
    candidates = np.unique(np.concatenate(([0.0], min_prob)))
    candidates = candidates[(candidates >= 0.0) & (candidates <= 0.5)]
    candidates.sort()
    for t in candidates[::-1]:
        mask = min_prob <= t
        if not np.any(mask):
            continue
        preds = (prob_pos[mask] >= decision_threshold).astype(int)
        acc = float((preds == y_true[mask]).mean())
        if acc >= target_acc:
            coverage = float(mask.mean())
            # Predictions exactly at confidence t are kept here, so the returned threshold must keep them
            # too under the abstention rule t <= p <= 1 - t used at test time and by the strategies:
            # move it just above t, but never past the next confidence level that this fit dropped.
            dropped = candidates[candidates > t]
            gap = (float(dropped.min()) if dropped.size else 0.5) - float(t)
            return float(t) + min(_TIE_MARGIN, 0.5 * gap), coverage
    return None, None


def classwise_thresholds(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    target_acc: float,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
    """One confidence threshold per predicted class (alternative to :func:`abstention_threshold`).

    Returns ``(t_pos, t_neg)``: keep a positive prediction if ``p >= t_pos`` and a negative prediction if
    ``p <= t_neg``, each chosen as the largest set of predictions of that class with accuracy >= target_acc.
    A side is ``None`` when no set of its predictions reaches the target.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    positive = prob_pos >= decision_threshold

    def fit(confidence: np.ndarray, correct: np.ndarray) -> float | None:
        best = None
        for c in np.unique(confidence):
            if correct[confidence >= c].mean() >= target_acc:
                best = float(c) if best is None else min(best, float(c))
        return best

    t_pos = fit(prob_pos[positive], y_true[positive] == 1) if positive.any() else None
    # negatives are fitted on -p, so that the threshold is one of the probabilities themselves
    t_neg = (
        fit(-prob_pos[~positive], y_true[~positive] == 0) if (~positive).any() else None
    )
    return t_pos, (None if t_neg is None else -t_neg)


def selective_at_classwise(y_true, prob_pos, t_pos, t_neg, decision_t):
    """Selective accuracy and coverage under one threshold per predicted class."""
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    positive = prob_pos >= decision_t
    covered = np.zeros_like(positive)
    if t_pos is not None:
        covered |= positive & (prob_pos >= t_pos)
    if t_neg is not None:
        covered |= ~positive & (prob_pos <= t_neg)
    if not covered.any():
        return float("nan"), 0.0
    return float((positive[covered].astype(int) == y_true[covered]).mean()), float(
        covered.mean()
    )


def decision_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Decision threshold on P(positive) with the highest accuracy (midpoint of the tied best), and that accuracy."""
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101, dtype=float)
    best_acc = -1.0
    best_thresholds = []
    for t in thresholds:
        preds = (prob_pos >= t).astype(int)
        acc = float((preds == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_thresholds = [float(t)]
        elif acc == best_acc:
            best_thresholds.append(float(t))
    best_t = (
        0.5 * (min(best_thresholds) + max(best_thresholds)) if best_thresholds else 0.5
    )
    return best_t, best_acc


def selective_kept(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    t: float | None,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float]:
    """Selective accuracy and coverage of the predictions with ``min(p, 1 - p) <= t``, the rule the fit uses."""
    if t is None:
        return None, 0.0
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    min_prob = np.minimum(prob_pos, 1.0 - prob_pos)
    mask = min_prob <= t
    if not np.any(mask):
        return None, 0.0
    preds = (prob_pos[mask] >= decision_threshold).astype(int)
    acc = float((preds == y_true[mask]).mean())
    coverage = float(mask.mean())
    return acc, coverage


def selective_at(y_true, prob_pos, abstention_t, decision_t):
    """Selective accuracy and coverage under the intervention abstention rule.

    Mirrors the abstention expression used by ConceptualSafeguardsStrategy so the
    coverage figure is unchanged, but scores kept predictions with the tuned
    decision threshold instead of a hard 0.5 cut.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    abstain = (prob_pos >= abstention_t) & (prob_pos <= 1.0 - abstention_t)
    covered = ~abstain
    if not covered.any():
        return float("nan"), 0.0
    preds = (prob_pos[covered] >= decision_t).astype(int)
    return float((preds == y_true[covered]).mean()), float(covered.mean())
