"""Alignment module for concept-based models.

Provides two alignment approaches:

1. **Constrained retraining** (``retrain_aligned``): Retrain the frontend with
   monotonicity (sign) constraints via CVXPY.  This is the approach used in
   the paper — it constrains *which direction* each concept should push the
   prediction while letting the optimizer pick the best magnitudes.

2. **Direct weight replacement** (``test_alignment``): Replace frontend weights
   with exact human-specified values.  Useful for ablation studies.
"""
from __future__ import annotations

__all__ = [
    "ConstrainedFrontEndModel",
    "retrain_aligned",
    "align_frontend_weights",
    "test_alignment",
]

import copy
from typing import Dict, List, Optional

import numpy as np

from concept_benchmark.models import FrontEndModel


# ── Constrained retraining via CVXPY ─────────────────────────────────

class _MockModel:
    """Minimal sklearn-like model wrapping raw weights for FrontEndModel compat."""

    def __init__(self, weights, bias):
        self.coef_ = np.array([weights])
        self.intercept_ = np.array([bias])


class ConstrainedFrontEndModel(FrontEndModel):
    """Frontend that fits logistic regression with sign constraints via CVXPY.

    Args:
        concept_names: List of concept name strings.
        monotonicity_constraints: ``{concept_name: sign}`` where *sign* is
            ``+1`` (weight must be >= 0) or ``-1`` (weight must be <= 0).
    """

    def __init__(
        self,
        concept_names: List[str],
        monotonicity_constraints: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model = None
        self.concept_names = concept_names
        self._mono: Dict[int, int] = {}
        for name, sign in (monotonicity_constraints or {}).items():
            idx = concept_names.index(name)
            self._mono[idx] = sign

    def fit(self, C: np.ndarray, y: np.ndarray, fit_params=None) -> None:
        if not self._mono:
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(
                random_state=42, max_iter=1000, solver="lbfgs",
                penalty="l2", C=1.0, n_jobs=-1,
            )
            self.model.fit(C, y)
            return

        import cvxpy as cp

        n_samples, n_features = C.shape
        y_bin = (y == 1).astype(int)
        w = cp.Variable(n_features)
        b = cp.Variable()

        logits = C @ w + b
        loss = cp.sum(cp.logistic(-cp.multiply(2 * y_bin - 1, logits)))
        constraints = []
        for idx, sign in self._mono.items():
            constraints.append(w[idx] >= 0 if sign == 1 else w[idx] <= 0)

        problem = cp.Problem(cp.Minimize(loss), constraints)
        problem.solve()

        if problem.status in ("infeasible", "unbounded"):
            print(f"Warning: CVXPY {problem.status}, falling back to unconstrained")
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(
                random_state=42, max_iter=1000, solver="lbfgs",
                penalty="l2", C=1.0, n_jobs=-1,
            )
            self.model.fit(C, y)
            return

        self.model = _MockModel(w.value, b.value)

    def predict(self, C: np.ndarray) -> np.ndarray:
        logits = C @ self.model.coef_[0] + self.model.intercept_[0]
        return (logits >= 0).astype(int)

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        logits = C @ self.model.coef_[0] + self.model.intercept_[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])


def retrain_aligned(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_test: np.ndarray,
    y_test: np.ndarray,
    concept_names: List[str],
    original_frontend: FrontEndModel,
    monotonicity_constraints: Dict[str, int],
) -> dict:
    """Retrain the frontend with sign constraints and compare to original.

    This is the paper's alignment approach: keep the same concept detector,
    but retrain the frontend logistic regression with monotonicity
    constraints (e.g. ``{"has_knees": 1}`` means has_knees must have a
    positive weight).

    Args:
        h_train: Concept predictions on training set.
        y_train: Training labels.
        h_test: Concept predictions on test set.
        y_test: Test labels.
        concept_names: List of concept names.
        original_frontend: The original (unconstrained) trained frontend.
        monotonicity_constraints: ``{concept_name: sign}`` dict.

    Returns:
        Dict with original_accuracy, aligned_accuracy, accuracy_change,
        predictions_changed, aligned_weights.
    """
    aligned_fe = ConstrainedFrontEndModel(
        concept_names=concept_names,
        monotonicity_constraints=monotonicity_constraints,
    )
    aligned_fe.fit(h_train, y_train)

    original_probs = original_frontend.predict_proba(h_test)
    aligned_probs = aligned_fe.predict_proba(h_test)
    original_preds = original_probs.argmax(1)
    aligned_preds = aligned_probs.argmax(1)

    test_labels = y_test.astype(int)
    original_acc = float((original_preds == test_labels).mean())
    aligned_acc = float((aligned_preds == test_labels).mean())

    # Extract learned weights
    aligned_weights = {}
    for i, concept in enumerate(concept_names):
        aligned_weights[concept] = float(aligned_fe.model.coef_[0, i])
    aligned_weights["bias"] = float(aligned_fe.model.intercept_[0])

    print("\n=== Aligned Frontend Weights ===")
    for i, concept in enumerate(concept_names):
        print(f"  {concept}: {aligned_fe.model.coef_[0, i]:.4f}")
    print(f"  bias: {aligned_fe.model.intercept_[0]:.4f}")

    return {
        "original_accuracy": original_acc,
        "aligned_accuracy": aligned_acc,
        "accuracy_change": float(aligned_acc - original_acc),
        "predictions_changed": int(np.sum(original_preds != aligned_preds)),
        "aligned_weights": aligned_weights,
    }


# ── Direct weight replacement ────────────────────────────────────────

def align_frontend_weights(frontend_model, concept_names, weight_dict):
    """Directly set frontend model weights for alignment.

    Args:
        frontend_model: Trained FrontEndModel instance.
        concept_names: List of concept names (in training order).
        weight_dict: Dict mapping concept names to weights, plus 'bias' key.

    Returns:
        Modified frontend model.
    """
    lr_model = frontend_model.model
    n_concepts = len(concept_names)
    new_coef = np.zeros((1, n_concepts))

    for concept_name, weight in weight_dict.items():
        if concept_name == "bias":
            continue
        if concept_name in concept_names:
            concept_idx = concept_names.index(concept_name)
            new_coef[0, concept_idx] = weight

    new_bias = weight_dict.get("bias", 0.0)
    lr_model.coef_ = new_coef
    lr_model.intercept_ = np.array([new_bias])

    return frontend_model


def test_alignment(h_test, align_params, fe, test):
    """Test alignment by replacing frontend weights with specified values.

    Args:
        h_test: Concept predictions on test set.
        align_params: Dict of alignment weights.
        fe: Original frontend model.
        test: Test dataset split.

    Returns:
        Dict with alignment statistics.
    """
    test_labels = test.y.astype(int)
    original_frontend = fe
    aligned_frontend = copy.deepcopy(fe)
    aligned_frontend = align_frontend_weights(
        aligned_frontend, test.concepts, align_params
    )

    original_probs = original_frontend.predict_proba(h_test)
    aligned_probs = aligned_frontend.predict_proba(h_test)
    original_preds = original_probs.argmax(1)
    aligned_preds = aligned_probs.argmax(1)

    original_acc = (original_preds == test_labels).mean()
    aligned_acc = (aligned_preds == test_labels).mean()

    aligned_preds = aligned_frontend.predict(h_test)

    return {
        "original_accuracy": float(original_acc),
        "aligned_accuracy": float(aligned_acc),
        "accuracy_change": float(aligned_acc - original_acc),
        "predictions_changed": int(np.sum(original_preds != aligned_preds)),
    }
