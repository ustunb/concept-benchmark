"""Robot pipeline — result collection and aggregation."""
from __future__ import annotations

import json
import logging

import pandas as pd

from concept_benchmark.utils import compute_accuracy, determine_device, get_loader_config
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load
from experiments.models import RobotClassifierCNN

from .regimes import InterventionSettings, _test_interventions

logger = logging.getLogger(__name__)


def _dataset_label(cfg: RobotBenchmarkConfig) -> str:
    """Return a human-readable dataset label for a config."""
    return "subconcept" if cfg.concept_preset == "foot_subtypes" else "ideal"


def collect_results(
    configs: list[RobotBenchmarkConfig] | None = None,
) -> pd.DataFrame:
    """Aggregate all robot results into a single flat CSV.

    Produces one row per (dataset, model, budget) combination with columns:
      dataset, model, budget, threshold, accuracy, gain,
      predictions_intervened_on, avg_concepts_per_sample, predictions_changed

    Reads saved artifacts only — no model retraining.
    """
    if configs is None:
        configs = [RobotBenchmarkConfig.default_ideal()]

    device = determine_device()
    loader_config = get_loader_config()
    rows = []

    # ── Per-config: DNN, CBM, interventions, alignment ───────────────
    for cfg in configs:
        label = _dataset_label(cfg)

        # Load this config's dataset
        data = load(cfg.get_dataset_path())

        # DNN accuracy
        dnn_path = cfg.get_model_path("dnn")
        dnn_accuracy = None
        if dnn_path.exists():
            dnn_weights = load(dnn_path)
            dnn = RobotClassifierCNN(input_size=cfg.input_size).to(device)
            dnn.load_state_dict(dnn_weights)
            test_loader = data.test.loader(shuffle=False, **loader_config)
            dnn_accuracy = compute_accuracy(dnn, test_loader, device)
            rows.append({
                "dataset": label,
                "model": "dnn",
                "budget": "",
                "threshold": "",
                "accuracy": round(dnn_accuracy, 4),
                "gain": 0.0,
                "predictions_intervened_on": "",
                "avg_concepts_per_sample": "",
                "predictions_changed": "",
            })

        # CBM no-intervention (k=0)
        cbm_path = cfg.get_model_path("cbm")
        if not cbm_path.exists():
            logger.warning("CBM not found for %s, skipping: %s", label, cbm_path)
            continue
        cbm = load(cbm_path)
        cbm_acc = float((cbm.predict(data.test) == data.test.y).mean())
        gain_ref = dnn_accuracy if dnn_accuracy is not None else cbm_acc
        rows.append({
            "dataset": label,
            "model": "cbm",
            "budget": 0,
            "threshold": "",
            "accuracy": round(cbm_acc, 4),
            "gain": round(cbm_acc - gain_ref, 4),
            "predictions_intervened_on": "",
            "avg_concepts_per_sample": "",
            "predictions_changed": "",
        })

        # CBM with interventions (k>0)
        results_path = cfg.get_results_path("cbm")
        if results_path.exists():
            interv_df = pd.read_csv(results_path)
            # Filter to baseline regime if column present
            if "regime" in interv_df.columns:
                interv_df = interv_df[interv_df["regime"] == "baseline"]
            # Use threshold=0.2 as the canonical threshold for the summary
            t02 = interv_df[(interv_df["threshold"] == 0.2) & (interv_df["budget"] > 0)]
            for _, row in t02.iterrows():
                budget = int(row["budget"])
                acc = float(row["accuracy"])
                pio = int(row["predictions_intervened_on"])
                tcc = int(row["total_concept_confirmations"])
                avg_cps = round(tcc / pio, 2) if pio > 0 else 0.0
                rows.append({
                    "dataset": label,
                    "model": "cbm",
                    "budget": budget,
                    "threshold": 0.2,
                    "accuracy": round(acc, 4),
                    "gain": round(acc - gain_ref, 4),
                    "predictions_intervened_on": pio,
                    "avg_concepts_per_sample": avg_cps,
                    "predictions_changed": int(row["predictions_changed"]),
                })

        # Aligned CBM
        align_path = cfg.get_alignment_results_path()
        if align_path.exists():
            with open(align_path) as f:
                align_data = json.load(f)
            aligned_acc = float(align_data["aligned_accuracy"])
            rows.append({
                "dataset": label,
                "model": "aligned_cbm",
                "budget": 0,
                "threshold": "",
                "accuracy": round(aligned_acc, 4),
                "gain": round(aligned_acc - gain_ref, 4),
                "predictions_intervened_on": "",
                "avg_concepts_per_sample": "",
                "predictions_changed": "",
            })

            # Aligned CBM with intervention at k=3
            aligned_weights = align_data.get("aligned_weights")
            if aligned_weights is not None:
                from experiments.alignment import align_frontend_weights
                import copy as _copy

                # Load the config's own dataset so concept shapes match
                cfg_data = load(cfg.get_dataset_path())
                aligned_fe = _copy.deepcopy(cbm.label_predictor)
                aligned_fe = align_frontend_weights(
                    aligned_fe, list(cfg_data.test.concepts), aligned_weights,
                )
                c_preds = cbm.concept_detector.predict(cfg_data.test)
                isettings = InterventionSettings(
                    seed=cfg.seed,
                    budgets=[3],
                    intervention_accuracy=cfg.intervention_accuracy,
                    intervention_threshold=0.2,
                )
                _, _, int_results = _test_interventions(
                    prob_test=c_preds,
                    settings=isettings,
                    acc_det=aligned_acc,
                    fe=aligned_fe,
                    test=cfg_data.test,
                )
                for key, res in int_results.items():
                    pio = int(res["predictions_intervened_on"])
                    tcc = int(res["total_concept_confirmations"])
                    avg_cps = round(tcc / pio, 2) if pio > 0 else 0.0
                    rows.append({
                        "dataset": label,
                        "model": "aligned_cbm",
                        "budget": 3,
                        "threshold": 0.2,
                        "accuracy": round(float(res["accuracy"]), 4),
                        "gain": round(float(res["accuracy"]) - gain_ref, 4),
                        "predictions_intervened_on": pio,
                        "avg_concepts_per_sample": avg_cps,
                        "predictions_changed": int(res["predictions_changed"]),
                    })

    final_df = pd.DataFrame(rows)
    cfg0 = configs[0]
    out_path = cfg0.get_collect_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(final_df), out_path)
    return final_df
