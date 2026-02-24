"""Sudoku validation benchmark pipeline.

Provides functions to run each stage of the sudoku benchmark programmatically.
"""
from __future__ import annotations

import copy
import logging
import platform
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from concept_benchmark.benchmarks._common import (
    compute_accuracy,
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
)
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.models import (
    ConceptBasedModel,
    ConceptDetector,
)
from concept_benchmark.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionConfig,
)

# Backward compatibility: allow unpickling models saved under old class paths.
# Saved CS models reference scripts.sudoku_demo.sudoku_models at pickle time.
import sys as _sys
import types as _types
if "scripts.sudoku_demo.sudoku_models" not in _sys.modules:
    import concept_benchmark.models as _compat_models
    if "scripts" not in _sys.modules:
        _sys.modules["scripts"] = _types.ModuleType("scripts")
    if "scripts.sudoku_demo" not in _sys.modules:
        _sd = _types.ModuleType("scripts.sudoku_demo")
        _sys.modules["scripts.sudoku_demo"] = _sd
        _sys.modules["scripts"].sudoku_demo = _sd
    _sys.modules["scripts.sudoku_demo.sudoku_models"] = _compat_models

logger = logging.getLogger(__name__)


# ── Stage: setup_dataset ──────────────────────────────────────────────

def setup_dataset(config: SudokuBenchmarkConfig) -> None:
    """Generate sudoku dataset (image + tabular + OCR sidecar)."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m", "concept_benchmark.synthetic.sudoku_ocr.make_ocr_dataset",
        "--n", str(config.n),
        "--n-samples", str(config.n_samples),
        "--valid-ratio", str(config.valid_ratio),
        "--max-corrupt", str(config.max_corrupt),
        "--seed", str(config.seed),
    ]
    subprocess.run(cmd, check=True)


# ── Stage: train_ocr ──────────────────────────────────────────────────

def train_ocr(config: SudokuBenchmarkConfig) -> None:
    """Train OCR digit recognizer."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m", "concept_benchmark.synthetic.sudoku_ocr.train_ocr_fast",
        "--seed", str(config.seed),
        "--max-corrupt", str(config.max_corrupt),
    ]
    subprocess.run(cmd, check=True)


# ── Stage: train_cs ───────────────────────────────────────────────────

def train_cs(
    config: SudokuBenchmarkConfig,
    data=None,
) -> ConceptBasedModel:
    """Train a concept supervision model (concept detector + frontend).

    Returns the trained CBM.
    """
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        tab_dir = config.get_dataset_path(data_type="tabular")
        data = load(tab_dir / "sudoku_dataset.pkl")
        data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=config.seed)
        data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

    if config.concept_missing_mech != "none":
        if config.concept_missing <= 0.0:
            raise ValueError(
                "concept_missing must be > 0 when concept_missing_mech is not 'none'"
            )
        data.sample_concept_missingness(
            p=config.concept_missing,
            mechanism=config.concept_missing_mech,
            rng=np.random.default_rng(config.seed),
        )
        data.training.concept_missing = True
        data.validation.concept_missing = True

    _macos = platform.system() == "Darwin"
    loader_config = {
        "device": device,
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }
    torch.manual_seed(config.seed)

    from concept_benchmark.models import (
        GroupPoolingConceptSudokuCNN as SudokuConceptModel,
    )

    model = SudokuConceptModel()
    cd = ConceptDetector(model=model)
    cbm = ConceptBasedModel(concept_detector=cd, propagate=True)
    cbm.fit(
        train_dataset=data.training,
        valid_dataset=data.validation,
        freeze=False,
        concept_embed_params={"shuffle": False, **loader_config},
        fit_params={
            "epochs": config.cs_epochs,
            "lr": 1e-3,
            "patience": config.cs_patience,
            **loader_config,
        },
    )

    test_pred = cbm.predict(data.test)
    logger.info("Test Accuracy: %s", np.mean(test_pred == data.test.y))

    save(cbm, config.get_model_path("cs", data_type="tabular"), overwrite=True)
    return cbm


# ── Stage: train_dnn ──────────────────────────────────────────────────

def train_dnn(
    config: SudokuBenchmarkConfig,
    data=None,
) -> dict:
    """Train an end-to-end DNN baseline for sudoku.

    Returns the best state_dict.
    """
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        tab_dir = config.get_dataset_path(data_type="tabular")
        data = load(tab_dir / "sudoku_dataset.pkl")
        data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=config.seed)
        data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

    torch.manual_seed(config.seed)

    from concept_benchmark.models import SudokuValidatorCNN as DNNSudokuModel

    model = DNNSudokuModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader_config = get_loader_config(device)
    train_loader = data.training.loader(shuffle=True, **loader_config)
    valid_loader = data.validation.loader(shuffle=False, **loader_config)
    test_loader = data.test.loader(shuffle=False, **loader_config)

    model.to(device)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        model.train()
        for X, _, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                batch_loss = criterion(outputs.squeeze(), y.float())
                val_loss_sum += batch_loss.item()
                val_batches += 1
        avg_val_loss = val_loss_sum / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if config.patience > 0 and epochs_no_improve >= config.patience:
                logger.info(
                    "Early stopping at epoch %d with best val loss %.6f",
                    epoch + 1, best_val_loss,
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_acc = compute_accuracy(model, train_loader, device=device)
    valid_acc = compute_accuracy(model, valid_loader, device=device)
    test_acc = compute_accuracy(model, test_loader, device=device)
    logger.info("Training Accuracy: %.2f%%", train_acc * 100)
    logger.info("Validation Accuracy: %.2f%%", valid_acc * 100)
    logger.info("Test Accuracy: %.2f%%", test_acc * 100)

    weights = best_state_dict if best_state_dict is not None else model.state_dict()
    save(weights, config.get_model_path("dnn", data_type="tabular"), overwrite=True)
    return weights


# ── Stage: run_interventions ──────────────────────────────────────────

def run_interventions(
    config: SudokuBenchmarkConfig,
    cs_model: Optional[ConceptBasedModel] = None,
    data=None,
) -> pd.DataFrame:
    """Run conceptual safeguards interventions on the sudoku CS model.

    Returns the intervention results DataFrame.
    """
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        img_dir = config.get_dataset_path(data_type="image")
        data = load(img_dir / "ocr_inferred_full_dataset.pkl")
        data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=config.seed)
        data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

    if cs_model is None:
        cs_model = load(config.get_model_path("cs", data_type="tabular"))
        cs_model._random_state = config.seed

    # Find selective accuracy threshold on validation set
    cs_probs, cs_y = _cs_val_probs(cs_model, data.validation)
    decision_threshold, cs_val_acc = _decision_threshold_sweep(cs_y, cs_probs)
    cs_t, cs_cov = _selective_accuracy_threshold(
        cs_y, cs_probs, config.target_accuracy, decision_threshold
    )

    if cs_t is None:
        raise ValueError(
            "Could not find a tau for conceptual safeguards at the target accuracy."
        )

    # Test set selective metrics
    cs_test_probs, cs_test_y = _cs_val_probs(cs_model, data.test)
    cs_sel_acc, cs_sel_cov = _selective_metrics(
        cs_test_y, cs_test_probs, cs_t, decision_threshold
    )

    # Run interventions at different budgets
    cs_runner = ConceptInterventionRunner(cs_model)
    cs_strategy = ConceptualSafeguardsStrategy()

    no_interv = {
        "budget": 0,
        "accuracy": cs_sel_acc,
        "predictions_intervened_on": 0,
        "total_concept_checks": 0,
        "total_concept_edits_made": 0,
        "selective_accuracy_after": cs_sel_acc,
        "coverage_after": cs_sel_cov,
    }

    rows = [no_interv]
    for budget in [1, 3, 27]:
        interv_cfg = InterventionConfig(
            tau=cs_t,
            max_concepts_per_instance=budget,
            random_state=config.seed,
        )
        result = cs_runner.run(cs_strategy, interv_cfg, data.test)
        acc_intervened = float((result.y_pred_after == data.test.y).mean())
        predictions_intervened_on = int(np.sum(np.any(result.mask, axis=1)))
        total_concept_checks = int(np.sum(result.mask))
        pred_binary = (result.C_pred >= 0.5).astype(int)
        final_binary = (result.C_intervened >= 0.5).astype(int)
        total_concept_edits_made = int(np.sum(pred_binary != final_binary))
        selective_acc_after = result.strat_metrics.get("selective_acc_after", None)
        coverage_after = result.strat_metrics.get("coverage_after", None)
        rows.append({
            "budget": budget,
            "accuracy": acc_intervened,
            "predictions_intervened_on": predictions_intervened_on,
            "total_concept_checks": total_concept_checks,
            "total_concept_edits_made": total_concept_edits_made,
            "selective_accuracy_after": selective_acc_after,
            "coverage_after": coverage_after,
        })

    cs_intervention_df = pd.DataFrame(rows)

    csv_path = (
        config.get_results_path("interventions", data_type="tabular")
        .with_suffix(".csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cs_intervention_df.to_csv(csv_path, index=False)
    logger.info("Saved intervention results to %s", csv_path)

    return cs_intervention_df


# ── Stage: align ─────────────────────────────────────────────────────

def align(
    config: SudokuBenchmarkConfig,
    cs_model: Optional[ConceptBasedModel] = None,
    data=None,
) -> dict:
    """Run alignment test on the trained CS model.

    Replaces the frontend weights with human-aligned weights (all 27
    constraints positive with equal weight, AND semantics) and compares
    original vs aligned accuracy.

    Returns dict with original_accuracy, aligned_accuracy, accuracy_change,
    predictions_changed.
    """
    import json as _json

    if data is None:
        tab_dir = config.get_dataset_path(data_type="tabular")
        data = load(tab_dir / "sudoku_dataset.pkl")
        data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=config.seed)
        data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

    if cs_model is None:
        cs_model = load(config.get_model_path("cs", data_type="tabular"))

    from concept_benchmark.alignment import test_alignment

    # Threshold at 0.5 to match cbm.predict() binarisation
    h_test = (cs_model.concept_detector.predict(data.test) > 0.5).astype(np.float32)
    stats = test_alignment(
        h_test=h_test,
        align_params=config.get_alignment_weights(),
        fe=cs_model.front_end_model,
        test=data.test,
    )

    logger.info("=== Alignment Results ===")
    logger.info("  Original accuracy: %.4f", stats['original_accuracy'])
    logger.info("  Aligned accuracy:  %.4f", stats['aligned_accuracy'])
    logger.info("  Accuracy change:   %+.4f", stats['accuracy_change'])
    logger.info("  Predictions changed: %s", stats['predictions_changed'])

    save_path = config.get_alignment_results_path()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        _json.dump(stats, f, indent=2)
    logger.info("  Saved to %s", save_path)

    return stats


# ── Stage: collect_results ────────────────────────────────────────────

def _dataset_label(cfg: SudokuBenchmarkConfig) -> str:
    """Human-readable dataset label for the summary CSV."""
    label = f"mc{cfg.max_corrupt}"
    if cfg.concept_missing_mech != "none":
        label += f"_{cfg.concept_missing_mech}"
        if cfg.concept_missing > 0:
            label += f"_{int(cfg.concept_missing * 100)}"
    return label


def collect_results(
    configs: Optional[List[SudokuBenchmarkConfig]] = None,
) -> pd.DataFrame:
    """Aggregate all sudoku results into a single flat CSV.

    Produces one row per (dataset, model, budget) combination with columns:
      dataset, model, budget, target_accuracy, raw_test_acc,
      selective_acc, selective_cov, predictions_intervened_on,
      avg_concepts_per_sample, predictions_changed

    Reads saved artifacts only — no model retraining.
    """
    import json

    from concept_benchmark.paths import results_dir

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
    out_path = results_dir / "sudoku_demo_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(final_df), out_path)
    return final_df


# ── Stage: run (orchestrator) ─────────────────────────────────────────

def run(
    config: Optional[SudokuBenchmarkConfig] = None,
    stages: Optional[List[str]] = None,
) -> None:
    """Run the full sudoku benchmark pipeline.

    Args:
        config: Benchmark configuration. Defaults to default().
        stages: List of stages to run. Default: all.
    """
    patch_macos_dataloader()

    if config is None:
        config = SudokuBenchmarkConfig.default()
    if stages is None:
        stages = ["setup", "ocr", "cs", "dnn", "intervene", "selective", "align", "collect"]

    device = determine_device()
    logger.info(
        "=== Sudoku Benchmark === seed=%d, stages=%s, device=%s",
        config.seed, stages, device,
    )

    if "setup" in stages:
        logger.info("=== Stage: setup ===")
        setup_dataset(config)

    if "ocr" in stages:
        logger.info("=== Stage: ocr ===")
        train_ocr(config)

    if "cs" in stages:
        logger.info("=== Stage: cs ===")
        train_cs(config)

    if "dnn" in stages:
        logger.info("=== Stage: dnn ===")
        train_dnn(config)

    if "intervene" in stages:
        logger.info("=== Stage: intervene ===")
        df = run_interventions(config)
        logger.info("=== Intervention Results ===\n%s", df.to_string(index=False))

    if "selective" in stages:
        logger.info("=== Stage: selective ===")
        sel_df = compute_selective_results(config)
        logger.info("=== Selective Metrics ===\n%s", sel_df.to_string(index=False))

    if "align" in stages:
        logger.info("=== Stage: align ===")
        align(config)

    if "collect" in stages:
        logger.info("=== Stage: collect ===")
        collect_results([config])

    logger.info("Pipeline complete!")


# ── Helper functions ──────────────────────────────────────────────────

def _selective_accuracy_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    target_acc: float,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
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
            return float(t), coverage
    return None, None


def _decision_threshold_sweep(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
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
    best_t = 0.5 * (min(best_thresholds) + max(best_thresholds)) if best_thresholds else 0.5
    return best_t, best_acc


def _selective_metrics(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    t: float | None,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float]:
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


def _cs_val_probs(model, dataset):
    probas = model.predict_proba(dataset)
    if probas.ndim == 1:
        prob_pos = probas
    else:
        prob_pos = probas[:, 1]
    y_true = np.asarray(dataset.y)
    return prob_pos, y_true


def _dnn_val_probs(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for X, _, y in loader:
            X = X.to(device)
            probs = model(X).squeeze(-1).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_y)


# ── Stage: compute and save selective metrics ─────────────────────────

def compute_selective_results(
    config: SudokuBenchmarkConfig,
    cs_model: Optional[ConceptBasedModel] = None,
    dnn_weights: Optional[dict] = None,
    data=None,
    target_accuracies: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute selective accuracy and coverage at multiple target accuracy
    thresholds for both DNN and CS models.  Saves results as CSV.

    target_accuracy is the minimum selective accuracy we require on the
    validation set.  Only predictions the model is confident enough about
    are kept (selective classification).  For each target we report the
    resulting selective accuracy and coverage on the test set.

    Columns: model, target_accuracy, raw_test_acc, selective_acc, selective_cov
    """
    patch_macos_dataloader()
    device = determine_device()

    if target_accuracies is None:
        target_accuracies = [0.55, 0.60, 0.65, 0.70, 0.75,
                             0.80, 0.85, 0.90, 0.95, 0.99, 1.00]

    # Load OCR-inferred data
    if data is None:
        img_dir = config.get_dataset_path(data_type="image")
        data = load(img_dir / "ocr_inferred_full_dataset.pkl")
        data.generate_cvindices(
            strata=data.y, total_folds_for_cv=[5], seed=config.seed
        )
        data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

    loader_cfg = get_loader_config(device)
    val_loader = data.validation.loader(shuffle=False, **loader_cfg)
    tst_loader = data.test.loader(shuffle=False, **loader_cfg)

    rows: list[dict] = []

    # ---- DNN selective metrics ----
    if dnn_weights is None:
        dnn_weights = load(config.get_model_path("dnn", data_type="tabular"))

    from concept_benchmark.models import SudokuValidatorCNN

    dnn = SudokuValidatorCNN()
    dnn.load_state_dict(dnn_weights)
    dnn.to(device)

    dnn_val_probs, dnn_val_y = _dnn_val_probs(dnn, val_loader, device)
    dnn_dt, _ = _decision_threshold_sweep(dnn_val_y, dnn_val_probs)
    dnn_test_probs, dnn_test_y = _dnn_val_probs(dnn, tst_loader, device)
    dnn_raw_acc = float(
        ((dnn_test_probs >= dnn_dt).astype(int) == dnn_test_y.astype(int)).mean()
    )

    for tau in target_accuracies:
        confidence_t, _ = _selective_accuracy_threshold(
            dnn_val_y, dnn_val_probs, tau, dnn_dt
        )
        sel_acc, sel_cov = _selective_metrics(
            dnn_test_y, dnn_test_probs, confidence_t, dnn_dt
        )
        rows.append({
            "model": "dnn",
            "target_accuracy": tau,
            "raw_test_acc": dnn_raw_acc,
            "selective_acc": sel_acc,
            "selective_cov": sel_cov,
        })

    # ---- CS selective metrics ----
    if cs_model is None:
        cs_model = load(config.get_model_path("cs", data_type="tabular"))
        cs_model._random_state = config.seed

    cs_val_probs, cs_val_y = _cs_val_probs(cs_model, data.validation)
    cs_dt, _ = _decision_threshold_sweep(cs_val_y, cs_val_probs)
    cs_test_probs, cs_test_y = _cs_val_probs(cs_model, data.test)
    cs_raw_acc = float(
        ((cs_test_probs >= cs_dt).astype(int) == cs_test_y.astype(int)).mean()
    )

    for tau in target_accuracies:
        confidence_t, _ = _selective_accuracy_threshold(
            cs_val_y, cs_val_probs, tau, cs_dt
        )
        sel_acc, sel_cov = _selective_metrics(
            cs_test_y, cs_test_probs, confidence_t, cs_dt
        )
        rows.append({
            "model": "cs",
            "target_accuracy": tau,
            "raw_test_acc": cs_raw_acc,
            "selective_acc": sel_acc,
            "selective_cov": sel_cov,
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = (
        config.get_results_path("selective", data_type="tabular")
        .with_suffix(".csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved selective metrics to %s", csv_path)

    return df
