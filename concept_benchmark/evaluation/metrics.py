"""Pure metric functions for evaluating concept bottleneck models.

All functions take numpy arrays and return floats. No model objects,
no datasets — just predictions and labels.

Decision support (robots):
    :func:`accuracy`, :func:`delta_accuracy`, :func:`gain`, :func:`intervention_metrics`

Automation (sudoku):
    :func:`selective_accuracy`, :func:`coverage`, :func:`net_work_automated`, and the paper's abstention
    protocol: :func:`decision_threshold`, :func:`abstention_threshold`, :func:`selective_at`
    (:func:`classwise_thresholds`, :func:`selective_at_classwise` fit one threshold per predicted class)
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
    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("intervention_metrics needs at least one prediction, got none")
    y_pred_before = np.argmax(y_prob_before, axis=1)
    y_pred_after = np.argmax(y_prob_after, axis=1)
    edits = (np.asarray(concepts_before) >= 0.5) != (np.asarray(concepts_after) >= 0.5)
    n_intervened = int(np.sum(np.any(mask, axis=1)))
    accuracy_after = float((y_pred_after == y_true).mean())
    return {
        "accuracy": accuracy_after,
        "accuracy_gain": accuracy_after - accuracy_before,
        "predictions_intervened_on": n_intervened,
        "interventions_rate": n_intervened / n_samples,
        "predictions_changed": int(np.sum(y_pred_after != y_pred_before)),
        "avg_edits_per_intervention": (
            float(edits.sum()) / n_intervened if n_intervened else 0.0
        ),
        "total_concept_confirmations": int(mask.sum()),
        "total_concept_edits_made": int(edits.sum()),
    }


# ── Abstention: the paper's selective-classification protocol ────────

_TIE_MARGIN = 1e-6  # see abstention_threshold


class PlattScaling:
    """Platt scaling of a label probability: ``p -> sigmoid(a * logit(p) + b)``, fitted on validation.

    Abstention with a guarantee (conceptual safeguards, Proposition 1) needs a calibrated label
    probability. A concept bottleneck that propagates concept uncertainty through an AND of many concepts
    is underconfident about the positive class; this one-parameter-pair fit on validation corrects that
    before the gate ``[t, 1 - t]`` is applied. ``fit`` returns the object; ``__call__`` applies it.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    @staticmethod
    def _logit(prob_positive: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(prob_positive, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    def fit(self, prob_positive: np.ndarray, y_true: np.ndarray) -> "PlattScaling":
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y_true).astype(int)
        if np.unique(y).size < 2:
            return self
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(self._logit(prob_positive)[:, None], y)
        self.a, self.b = float(model.coef_[0, 0]), float(model.intercept_[0])
        return self

    def __call__(self, prob_positive: np.ndarray) -> np.ndarray:
        z = np.clip(self.a * self._logit(prob_positive) + self.b, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))


def abstention_mask(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Which predictions a model with abstention threshold ``threshold`` abstains on.

    ``y_prob`` is either P(positive) of shape (N,) or class probabilities of shape (N, K). A prediction is
    abstained on when its margin to the decision, ``min(p, 1 - p)`` for two classes and ``1 - max_k p_k``
    otherwise, is at least ``threshold``. This is the one rule shared by the fit, the measures and the
    intervention strategies.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
        margin = 1.0 - y_prob.max(axis=1)
    else:
        prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        margin = np.minimum(prob_positive, 1.0 - prob_positive)
    return margin >= threshold


def abstention_threshold(
    y_true: np.ndarray,
    prob_positive: np.ndarray,
    target_accuracy: float,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
    """Abstention threshold at which the kept predictions reach ``target_accuracy``, and their coverage.

    A model abstains on ``min(p, 1 - p) >= t`` (see :func:`abstention_mask`). The threshold is the largest ``t``
    whose kept predictions, scored with ``decision_threshold``, reach the target; fit it on validation
    predictions. Returns ``(None, None)`` when no threshold reaches the target.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_positive = np.asarray(prob_positive, dtype=float).reshape(-1)
    min_prob = np.minimum(prob_positive, 1.0 - prob_positive)
    candidates = np.unique(np.concatenate(([0.0], min_prob)))
    candidates = candidates[(candidates >= 0.0) & (candidates <= 0.5)]
    candidates.sort()
    for t in candidates[::-1]:
        mask = min_prob <= t
        if not np.any(mask):
            continue
        preds = (prob_positive[mask] >= decision_threshold).astype(int)
        acc = float((preds == y_true[mask]).mean())
        if acc >= target_accuracy:
            coverage = float(mask.mean())
            # Predictions exactly at confidence t are kept here, so the returned threshold must keep them
            # too under the abstention rule min(p, 1 - p) >= t used at test time and by the strategies:
            # move it just above t, but never past the next confidence level that this fit dropped.
            dropped = candidates[candidates > t]
            gap = (float(dropped.min()) if dropped.size else 0.5) - float(t)
            return float(t) + min(_TIE_MARGIN, 0.5 * gap), coverage
    return None, None


def classwise_thresholds(
    y_true: np.ndarray,
    prob_positive: np.ndarray,
    target_accuracy: float,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
    """One confidence threshold per predicted class (alternative to :func:`abstention_threshold`).

    Returns ``(t_pos, t_neg)``: keep a positive prediction if ``p >= t_pos`` and a negative prediction if
    ``p <= t_neg``, each chosen as the largest set of predictions of that class with accuracy >= target_accuracy.
    A side is ``None`` when no set of its predictions reaches the target.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_positive = np.asarray(prob_positive, dtype=float).reshape(-1)
    positive = prob_positive >= decision_threshold

    def fit(confidence: np.ndarray, correct: np.ndarray) -> float | None:
        best = None
        for c in np.unique(confidence):
            if correct[confidence >= c].mean() >= target_accuracy:
                best = float(c) if best is None else min(best, float(c))
        return best

    t_pos = (
        fit(prob_positive[positive], y_true[positive] == 1) if positive.any() else None
    )
    # negatives are fitted on -p, so that the threshold is one of the probabilities themselves
    t_neg = (
        fit(-prob_positive[~positive], y_true[~positive] == 0)
        if (~positive).any()
        else None
    )
    return t_pos, (None if t_neg is None else -t_neg)


def selective_at_classwise(y_true, prob_positive, t_pos, t_neg, decision_threshold):
    """Selective accuracy and coverage under one threshold per predicted class."""
    y_true = np.asarray(y_true).astype(int)
    prob_positive = np.asarray(prob_positive, dtype=float).reshape(-1)
    positive = prob_positive >= decision_threshold
    covered = np.zeros_like(positive)
    if t_pos is not None:
        covered |= positive & (prob_positive >= t_pos)
    if t_neg is not None:
        covered |= ~positive & (prob_positive <= t_neg)
    if not covered.any():
        return float("nan"), 0.0
    return float((positive[covered].astype(int) == y_true[covered]).mean()), float(
        covered.mean()
    )


def decision_threshold(
    y_true: np.ndarray,
    prob_positive: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Decision threshold on P(positive) with the highest accuracy, and the accuracy at that threshold.

    Among the tied best thresholds of the grid, the midpoint of the longest run of consecutive ones is
    returned, so that the threshold lies inside a region of best accuracy rather than between two of them.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_positive = np.asarray(prob_positive, dtype=float).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101, dtype=float)
    thresholds = np.sort(np.asarray(thresholds, dtype=float))
    if thresholds.size == 0:
        return 0.5, _accuracy_at(y_true, prob_positive, 0.5)
    accuracies = np.array([_accuracy_at(y_true, prob_positive, t) for t in thresholds])
    is_best = accuracies == accuracies.max()
    best_run, run_start = None, None
    for i, flag in enumerate(np.append(is_best, False)):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if best_run is None or i - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, i)
            run_start = None
    start, end = best_run
    best_t = 0.5 * (float(thresholds[start]) + float(thresholds[end - 1]))
    return best_t, _accuracy_at(y_true, prob_positive, best_t)


def _accuracy_at(
    y_true: np.ndarray, prob_positive: np.ndarray, threshold: float
) -> float:
    return float(((prob_positive >= threshold).astype(int) == y_true).mean())


def selective_kept(
    y_true: np.ndarray,
    prob_positive: np.ndarray,
    threshold: float | None,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float]:
    """Selective accuracy and coverage of the predictions with ``min(p, 1 - p) <= threshold``, the rule the fit uses."""
    if threshold is None:
        return None, 0.0
    y_true = np.asarray(y_true).astype(int)
    prob_positive = np.asarray(prob_positive, dtype=float).reshape(-1)
    min_prob = np.minimum(prob_positive, 1.0 - prob_positive)
    mask = min_prob <= threshold
    if not np.any(mask):
        return None, 0.0
    preds = (prob_positive[mask] >= decision_threshold).astype(int)
    acc = float((preds == y_true[mask]).mean())
    coverage = float(mask.mean())
    return acc, coverage


def selective_at(y_true, prob_positive, abstention_threshold, decision_threshold):
    """Selective accuracy and coverage of a model that abstains on ``min(p, 1 - p) >= t``.

    The kept predictions (see :func:`abstention_mask`, the rule of the intervention strategies) are scored
    with ``decision_threshold``, not with a 0.5 cut. Coverage is 0 and accuracy ``nan`` when nothing is kept.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_positive = np.asarray(prob_positive, dtype=float).reshape(-1)
    covered = ~abstention_mask(prob_positive, abstention_threshold)
    if not covered.any():
        return float("nan"), 0.0
    preds = (prob_positive[covered] >= decision_threshold).astype(int)
    return float((preds == y_true[covered]).mean()), float(covered.mean())
