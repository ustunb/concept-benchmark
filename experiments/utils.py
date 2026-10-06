"""Alignment of a trained CBM (repo-only); training and evaluation blocks are in experiments.evaluate."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


# ── Alignment ────────────────────────────────────────────────────────


def run_alignment(
    concept_based_model,
    train_dataset,
    test_dataset,
    monotonicity_constraints: dict[str, int],
    save_path: Path | None = None,
) -> dict:
    """Run alignment: retrain frontend with sign constraints, compare to original.

    Parameters
    ----------
    concept_based_model : ConceptBasedModel
        Trained ConceptBasedModel.
    train_dataset : ConceptDatasetSample
        Training split (for retraining the frontend).
    test_dataset : ConceptDatasetSample
        Test split (for evaluation).
    monotonicity_constraints : dict
        ``{concept_name: sign}`` where sign is +1 (positive weight) or
        -1 (negative weight).
    save_path : Path, optional
        Optional path to save results as JSON.

    Returns
    -------
    dict
        Dict with ``original_accuracy``, ``aligned_accuracy``,
        ``accuracy_change``, ``predictions_changed``, ``aligned_weights``.
    """
    from experiments.alignment import retrain_aligned

    # Use ground-truth concepts for training (matching the paper where both
    # original and aligned frontends are trained on GT labels).
    # Test uses predicted concepts (binarised at 0.5, matching cbm.predict()).
    concept_preds_train = train_dataset.C.astype(np.float32)
    concept_preds_test = concept_based_model.concept_detector.predict(
        test_dataset
    ).astype(np.float32)

    stats = retrain_aligned(
        concept_preds_train=concept_preds_train,
        y_train=train_dataset.y.astype(int),
        concept_preds_test=concept_preds_test,
        y_test=test_dataset.y.astype(int),
        concept_names=list(test_dataset.concepts),
        original_frontend=concept_based_model.label_predictor,
        monotonicity_constraints=monotonicity_constraints,
    )

    logger.info("\n=== Alignment Results ===")
    logger.info("  Original accuracy: %.4f", stats["original_accuracy"])
    logger.info("  Aligned accuracy:  %.4f", stats["aligned_accuracy"])
    logger.info("  Accuracy change:   %+.4f", stats["accuracy_change"])
    logger.info("  Predictions changed: %d", stats["predictions_changed"])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types for JSON serialization
        serializable = {
            k: (
                v
                if not isinstance(v, dict)
                else {kk: float(vv) for kk, vv in v.items()}
            )
            for k, v in stats.items()
        }
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("  Saved to %s", save_path)

    return stats
