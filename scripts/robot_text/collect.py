"""Robot text pipeline — result collection and aggregation."""
from __future__ import annotations

import json
import logging

import pandas as pd

from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import results_dir

logger = logging.getLogger(__name__)


def collect_results(
    configs: list[RobotBenchmarkConfig] | None = None,
) -> pd.DataFrame:
    """Aggregate robot text results into a single CSV.

    Reads saved artifacts only — no model retraining.
    """
    if configs is None:
        configs = [RobotBenchmarkConfig(data_type="text")]

    rows: list[dict] = []
    for cfg in configs:
        results_path = cfg.get_results_path("cbm")
        if results_path.exists():
            df = pd.read_csv(results_path)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "seed": cfg.seed,
                        "model": "cbm",
                        "budget": int(r["budget"]),
                        "threshold": float(r["threshold"]),
                        "accuracy": float(r["accuracy"]),
                        "predictions_intervened_on": int(
                            r["predictions_intervened_on"]
                        ),
                        "predictions_changed": int(r["predictions_changed"]),
                    }
                )

        # DNN metrics
        dnn_path = cfg.get_model_path("dnn")
        if dnn_path.exists():
            dnn_data = load(dnn_path)
            if isinstance(dnn_data, dict) and "metrics" in dnn_data:
                rows.append(
                    {
                        "seed": cfg.seed,
                        "model": "dnn",
                        "budget": "",
                        "threshold": "",
                        "accuracy": dnn_data["metrics"].get("accuracy", ""),
                        "predictions_intervened_on": "",
                        "predictions_changed": "",
                    }
                )

        # Alignment
        align_path = cfg.get_alignment_results_path()
        if align_path.exists():
            with open(align_path) as f:
                align_data = json.load(f)
            rows.append(
                {
                    "seed": cfg.seed,
                    "model": "aligned_cbm",
                    "budget": 0,
                    "threshold": "",
                    "accuracy": float(align_data["aligned_accuracy"]),
                    "predictions_intervened_on": "",
                    "predictions_changed": align_data.get("predictions_changed", ""),
                }
            )

    final_df = pd.DataFrame(rows)
    out_path = results_dir / "robot_text_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(final_df), out_path)
    return final_df
