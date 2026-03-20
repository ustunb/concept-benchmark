"""Sudoku pipeline — selective accuracy helpers and compute_selective_results."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from concept_benchmark.utils import (
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
)
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.ext.fileutils import load
from experiments.models import ConceptBasedModel

logger = logging.getLogger(__name__)


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
    cs_model: ConceptBasedModel | None = None,
    dnn_weights: dict | None = None,
    data=None,
    target_accuracies: list[float] | None = None,
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

    # Load evaluation data: OCR-inferred (image mode) or tabular
    if data is None:
        if config.data_type == "image":
            img_dir = config.get_dataset_path(data_type="image")
            data = load(img_dir / "ocr_inferred_full_dataset.pkl")
        else:
            tab_dir = config.get_dataset_path(data_type="tabular")
            data = load(tab_dir / "sudoku_dataset.pkl")
        data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=config.seed)

    loader_cfg = get_loader_config()
    val_loader = data.validation.loader(shuffle=False, **loader_cfg)
    tst_loader = data.test.loader(shuffle=False, **loader_cfg)

    rows: list[dict] = []

    # ---- DNN selective metrics ----
    if dnn_weights is None:
        dnn_weights = load(config.get_model_path("dnn", data_type="tabular"))

    from experiments.models import SudokuValidatorCNN

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
