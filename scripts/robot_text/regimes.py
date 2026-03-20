"""Robot text pipeline — intervention regime dispatch and helpers."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from concept_benchmark.ext.fileutils import load

logger = logging.getLogger(__name__)


# Lazy imports for intervention modules
_intervention_imported = False


def _ensure_intervention_imports():
    global _intervention_imported
    if not _intervention_imported:
        global ConceptInterventionRunner, InterventionConfig, KFlipInterventionStrategy
        from experiments.intervention import (
            ConceptInterventionRunner,
            InterventionConfig,
        )
        from experiments.kflip import KFlipInterventionStrategy

        _intervention_imported = True


def _run_text_regime(config, regime, model, data, budgets, threshold):
    """Run one intervention regime for the text benchmark.

    Returns a DataFrame with rows for each budget.
    """
    _ensure_intervention_imports()

    # Select model and human accuracy per regime
    if regime == "baseline":
        regime_model = model
        human_acc = config.intervention_accuracy
    elif regime == "expert":
        regime_model = model
        human_acc = config.expert_intervention_accuracy
    elif regime == "subjective":
        regime_model = load(config.get_model_path("cbm_subjective"))
        human_acc = config.subjective_intervention_accuracy
    elif regime == "machine":
        # Use existing LabelFreeDetector from robot_text/lfcbm.py
        regime_model = load(config.get_model_path("lfcbm"))
        human_acc = config.intervention_accuracy
    else:
        raise ValueError(f"Unknown regime: {regime!r}")

    c_preds = regime_model.concept_detector.predict(data.test)
    base_pred = regime_model.predict(data.test)
    base_acc = float(np.mean(base_pred == data.test.y))

    runner = ConceptInterventionRunner(regime_model)

    rows = []
    # k=0 baseline
    rows.append(
        {
            "budget": 0,
            "threshold": threshold,
            "accuracy": base_acc,
            "predictions_intervened_on": 0,
            "predictions_changed": 0,
            "total_concept_confirmations": 0,
            "total_concept_edits_made": 0,
        }
    )

    rng = np.random.default_rng(config.seed)
    err_prob = 1.0 - human_acc

    for budget in budgets:
        interv_config = InterventionConfig(
            max_concepts_per_instance=budget,
            random_state=config.seed,
            score_threshold=threshold,
            intervention_noise_rate=err_prob,
        )
        strategy = KFlipInterventionStrategy(
            use_exact_k=(config.intervention_strategy == "exactly_k"),
        )

        result = runner.run(
            strategy=strategy,
            config=interv_config,
            dataset=data.test,
            concept_proba=c_preds,
            labels=data.test.y.astype(int),
        )

        mask = result.mask
        C_gt = data.test.C.astype(np.float32)
        C_after = result.C_intervened.copy()

        mistake_draw = rng.random(C_after.shape) < err_prob
        mistakes = mask & mistake_draw
        C_after[mistakes] = 1.0 - C_gt[mistakes]
        result.C_intervened = C_after

        C_pred_binary = (result.C_pred >= 0.5).astype(int)
        C_final_binary = (result.C_intervened >= 0.5).astype(int)
        actual_edits_mask = C_pred_binary != C_final_binary
        result.y_prob_after = regime_model.label_predictor.predict_proba(C_final_binary)
        result.y_pred_after = np.argmax(result.y_prob_after, axis=1)

        y_pred_before = np.argmax(result.y_prob_before, axis=1)
        num_preds_change = int(np.sum(result.y_pred_after != y_pred_before))
        acc_intervened = float(np.mean(result.y_pred_after == data.test.y.astype(int)))
        n_intervened = int(np.sum(mask))

        rows.append(
            {
                "budget": budget,
                "threshold": threshold,
                "accuracy": acc_intervened,
                "predictions_intervened_on": int(np.sum(np.any(mask, axis=1))),
                "predictions_changed": num_preds_change,
                "total_concept_confirmations": n_intervened,
                "total_concept_edits_made": int(np.sum(actual_edits_mask)),
            }
        )

    regime_df = pd.DataFrame(rows)
    regime_df["regime"] = regime
    return regime_df
