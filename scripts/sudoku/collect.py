"""Sudoku pipeline — result collection and aggregation."""
from __future__ import annotations

import json
import logging

import pandas as pd

from concept_benchmark.config import SudokuBenchmarkConfig

logger = logging.getLogger(__name__)


def _dataset_label(cfg: SudokuBenchmarkConfig) -> str:
    """Human-readable dataset label for the summary CSV."""
    return f"mc{cfg.max_cell_swaps}"


def collect_results(
    configs: list[SudokuBenchmarkConfig] | None = None,
) -> pd.DataFrame:
    """Aggregate all sudoku results into a single flat CSV.

    Produces one row per (dataset, model, budget) combination with columns:
      dataset, model, budget, target_accuracy, raw_test_acc,
      selective_acc, selective_cov, predictions_intervened_on,
      avg_concepts_per_sample, predictions_changed

    Reads saved artifacts only — no model retraining.
    """
    if configs is None:
        configs = [SudokuBenchmarkConfig.default()]

    rows: list[dict] = []

    for cfg in configs:
        label = _dataset_label(cfg)
        target = cfg.target_accuracy

        # ── Selective CSV: DNN + CS at the default target_accuracy ────
        sel_csv = (
            cfg.get_results_path("selective", data_type="tabular")
            .with_suffix(".csv")
        )
        if sel_csv.exists():
            sel_df = pd.read_csv(sel_csv)
            # Normalise column name (mc9 uses "tau", mc21 uses "target_accuracy")
            if "tau" in sel_df.columns:
                sel_df = sel_df.rename(columns={"tau": "target_accuracy"})

            for model in ["dnn", "cs"]:
                model_df = sel_df[
                    (sel_df["model"] == model)
                    & (sel_df["target_accuracy"] == target)
                ]
                if model_df.empty:
                    continue
                r = model_df.iloc[0]
                sel_acc = r["selective_acc"]
                rows.append({
                    "dataset": label,
                    "model": model,
                    "budget": 0 if model == "cs" else "",
                    "target_accuracy": target,
                    "raw_test_acc": round(float(r["raw_test_acc"]), 4),
                    "selective_acc": round(float(sel_acc), 4) if pd.notna(sel_acc) else "",
                    "selective_cov": round(float(r["selective_cov"]), 4),
                    "predictions_intervened_on": "",
                    "avg_concepts_per_sample": "",
                    "predictions_changed": "",
                })

        # ── Intervention CSV: CS at k > 0 ────────────────────────────
        interv_csv = (
            cfg.get_results_path("interventions", data_type="tabular")
            .with_suffix(".csv")
        )
        if interv_csv.exists():
            interv_df = pd.read_csv(interv_csv)
            for _, r in interv_df.iterrows():
                budget = int(r["budget"])
                if budget == 0:
                    continue  # already have k=0 from selective CSV
                pio = int(r["predictions_intervened_on"])
                tcc = int(r["total_concept_checks"])
                avg_cps = round(tcc / pio, 2) if pio > 0 else 0.0
                sel_acc = r.get("selective_accuracy_after")
                cov = r.get("coverage_after")
                rows.append({
                    "dataset": label,
                    "model": "cs",
                    "budget": budget,
                    "target_accuracy": target,
                    "raw_test_acc": "",
                    "selective_acc": round(float(sel_acc), 4) if pd.notna(sel_acc) else "",
                    "selective_cov": round(float(cov), 4) if pd.notna(cov) else "",
                    "predictions_intervened_on": pio,
                    "avg_concepts_per_sample": avg_cps,
                    "predictions_changed": int(r["total_concept_edits_made"]),
                })

        # ── Alignment JSON ───────────────────────────────────────────
        align_path = cfg.get_alignment_results_path(data_type="tabular")
        if align_path.exists():
            with open(align_path) as f:
                align_data = json.load(f)
            rows.append({
                "dataset": label,
                "model": "aligned_cs",
                "budget": 0,
                "target_accuracy": "",
                "raw_test_acc": round(float(align_data["aligned_accuracy"]), 4),
                "selective_acc": "",
                "selective_cov": "",
                "predictions_intervened_on": "",
                "avg_concepts_per_sample": "",
                "predictions_changed": align_data.get("predictions_changed", ""),
            })

    final_df = pd.DataFrame(rows)
    cfg0 = configs[0]
    out_path = cfg0.get_collect_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(final_df), out_path)
    return final_df
