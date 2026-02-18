"""Alignment module for concept-based models.

Functions for aligning concept-based model frontends with human preferences
and computing alignment metrics.  Extracted from scripts/utils/robot_alignment.py.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np


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


def compute_alignment_metrics(
    frontend_model,
    concept_names,
    human_alignment_dict,
    test_predictions,
    test_labels,
):
    """Compute alignment metrics between model and human preferences.

    Args:
        frontend_model: The trained frontend model.
        concept_names: List of concept names.
        human_alignment_dict: Dictionary of human alignment preferences.
        test_predictions: Model predictions on test set.
        test_labels: True labels for test set.

    Returns:
        Dictionary containing alignment metrics.
    """
    model_weights = {}
    for i, concept in enumerate(concept_names):
        if hasattr(frontend_model.model, "coef_"):
            model_weights[concept] = frontend_model.model.coef_[0, i]
        else:
            model_weights[concept] = 0.0

    if hasattr(frontend_model.model, "intercept_"):
        model_weights["bias"] = frontend_model.model.intercept_[0]
    else:
        model_weights["bias"] = 0.0

    alignment_score = 0.0
    total_concepts = 0

    for concept, human_pref in human_alignment_dict.items():
        if concept in model_weights and concept != "bias":
            model_weight = model_weights[concept]
            if (human_pref > 0 and model_weight > 0) or (
                human_pref < 0 and model_weight < 0
            ):
                alignment_score += 1.0
            elif human_pref == 0 and abs(model_weight) < 0.1:
                alignment_score += 1.0
            total_concepts += 1

    alignment_percentage = (
        alignment_score / total_concepts if total_concepts > 0 else 0.0
    )

    if len(test_predictions) > 0 and len(test_labels) > 0:
        binary_preds = (test_predictions > 0.5).astype(int)
        accuracy = np.mean(binary_preds == test_labels)
    else:
        accuracy = 0.0

    return {
        "alignment_percentage": alignment_percentage,
        "aligned_concepts": alignment_score,
        "total_concepts": total_concepts,
        "model_weights": model_weights,
        "human_preferences": human_alignment_dict,
        "prediction_accuracy": accuracy,
    }


def validate_alignment_configuration(human_alignment_dict, concept_names):
    """Validate that a human alignment configuration is valid for the given concepts.

    Returns:
        dict with 'valid' boolean and 'issues' list.
    """
    issues = []

    for concept in human_alignment_dict:
        if concept != "bias" and concept not in concept_names:
            issues.append(
                f"Concept '{concept}' not found in available concepts: {concept_names}"
            )

    for concept, weight in human_alignment_dict.items():
        if not isinstance(weight, (int, float)):
            issues.append(f"Weight for '{concept}' must be numeric, got {type(weight)}")
        elif abs(weight) > 100:
            issues.append(f"Weight for '{concept}' seems unusually large: {weight}")

    return {"valid": len(issues) == 0, "issues": issues}


def test_alignment(h_test, align_params, fe, test):
    """Test alignment of a frontend model with specified weights.

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

    print("\n=== Aligned Frontend Weights ===")
    for i, concept in enumerate(test.concepts):
        print(f"  {concept}: {aligned_frontend.model.coef_[0, i]:.4f}")
    print(f"  bias: {aligned_frontend.model.intercept_[0]:.4f}")

    aligned_preds = aligned_frontend.predict(h_test)

    alignment_stats = {
        "original_accuracy": float(original_acc),
        "aligned_accuracy": float(aligned_acc),
        "accuracy_change": float(aligned_acc - original_acc),
        "predictions_changed": int(np.sum(original_preds != aligned_preds)),
    }
    return alignment_stats
