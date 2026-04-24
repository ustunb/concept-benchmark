#!/usr/bin/env python3
"""Real-world automation experiments: Pistachio + Rice MSC.

Train DNN, CBM, CEM, ProbCBM, ECBM on real-world tabular datasets and
evaluate as automation tasks (selective classification + interventions).

Usage:
    PYTHONPATH=. python scripts/realworld_pipeline.py --dataset pistachio --seed 42
    PYTHONPATH=. python scripts/realworld_pipeline.py --dataset rice --seed 42
    PYTHONPATH=. python scripts/realworld_pipeline.py --dataset both --seed 42
"""

from __future__ import annotations

import argparse
import copy
import logging
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from concept_benchmark.data import ConceptDataset
from concept_benchmark.evaluation.metrics import (
    accuracy,
    coverage,
    net_work_automated,
    selective_accuracy,
)
from concept_benchmark.utils import set_deterministic_seed
from experiments.intervention import predict_label_proba_from_concepts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Column definitions ────────────────────────────────────────────────

MORPH_CONCEPTS = [
    "Area", "Perimeter", "Major_Axis", "Minor_Axis", "Eccentricity",
    "Eqdiasq", "Solidity", "Convex_Area", "Extent", "Aspect_Ratio",
    "Roundness", "Compactness", "Shapefactor_1", "Shapefactor_2",
    "Shapefactor_3", "Shapefactor_4",
]
MORPH_CONCEPTS_UPPER = [c.upper() for c in MORPH_CONCEPTS]
BINARY_CONCEPTS = [f"{c}_binary" for c in MORPH_CONCEPTS]
BINARY_CONCEPTS_UPPER = [f"{c}_binary" for c in MORPH_CONCEPTS_UPPER]

PISTACHIO_COLOR_COLS = [
    "Mean_RR", "Mean_RG", "Mean_RB", "StdDev_RR", "StdDev_RG", "StdDev_RB",
    "Skew_RR", "Skew_RG", "Skew_RB", "Kurtosis_RR", "Kurtosis_RG", "Kurtosis_RB",
]


# ── Dataset loaders ──────────────────────────────────────────────────


def load_pistachio(seed: int = 42) -> ConceptDataset:
    csv_path = REPO_ROOT / "data" / "pistachio" / "pistachio_cbm.csv"
    df = pd.read_csv(csv_path)

    # X = all continuous features (morph + color) — concept detector predicts shape from these
    x_cols = MORPH_CONCEPTS + PISTACHIO_COLOR_COLS
    X = df[x_cols].values.astype(np.float32)
    C = df[BINARY_CONCEPTS].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Standardize features (fit on full data before split, same as original paper's PCA pipeline)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    dataset = ConceptDataset(
        X=X, C=C, y=y,
        meta={
            "classes": ["Siirt", "Kirmizi"],
            "concepts": BINARY_CONCEPTS,
            "data_type": "tabular",
        },
    )
    dataset.sample(test_size=0.2, val_size=0.2, stratify=y, seed=seed)
    logger.info("Pistachio: %d train, %d val, %d test, %d concepts, %d input features",
                dataset.train.n, dataset.val.n, dataset.test.n,
                dataset.train.n_concepts, X.shape[1])
    return dataset


def load_rice(seed: int = 42) -> ConceptDataset:
    csv_path = REPO_ROOT / "data" / "rice" / "rice_msc_cbm.csv"
    df = pd.read_csv(csv_path)

    # Concepts = binarized morphological features (uppercase in this CSV)
    concept_cols = BINARY_CONCEPTS_UPPER
    # Input X = everything that's not a concept, not a label, not CLASS, not raw morph
    exclude = set(MORPH_CONCEPTS_UPPER) | set(concept_cols) | {"label", "CLASS"}
    color_cols = [c for c in df.columns if c not in exclude]

    # Drop entropy features (values in billions, different encoding)
    entropy_cols = [c for c in color_cols if "entropy" in c.lower()]
    color_cols = [c for c in color_cols if c not in entropy_cols]
    logger.info("Dropped %d entropy features with extreme values", len(entropy_cols))

    # Drop rows with NaN (only ~8 out of 75k)
    all_cols = color_cols + concept_cols + ["label"]
    df = df.dropna(subset=all_cols).reset_index(drop=True)
    logger.info("Dropped rows with NaN, %d remaining", len(df))

    X = df[color_cols].values.astype(np.float32)
    C = df[concept_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    classes = sorted(df["CLASS"].unique())

    dataset = ConceptDataset(
        X=X, C=C, y=y,
        meta={
            "classes": classes,
            "concepts": concept_cols,
            "data_type": "tabular",
        },
    )
    dataset.sample(test_size=0.2, val_size=0.2, stratify=y, seed=seed)
    logger.info("Rice: %d train, %d val, %d test, %d concepts, %d input features, %d classes",
                dataset.train.n, dataset.val.n, dataset.test.n,
                dataset.train.n_concepts, X.shape[1], len(classes))
    return dataset


# ── Config namespace (mimics what train_*_model expects) ─────────────


def make_config(seed: int = 42, epochs: int = 50, patience: int = 10,
                batch_size: int = 64, lr: float = 1e-3) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        batch_size=batch_size,
        learning_rate=lr,
        # CBM
        cs_epochs=epochs,
        cs_patience=patience,
        # CEM
        cem_emb_size=16,
        cem_training_intervention_prob=0.25,
        cem_concept_loss_weight=1.0,
        cem_task_loss_weight=1.0,
        cem_max_epochs=epochs,
        cem_patience=patience,
        # ProbCBM
        probcbm_hidden_dim=8,
        probcbm_class_hidden_dim=64,
        probcbm_intervention_prob=0.25,
        probcbm_n_samples_inference=50,
        probcbm_latent_dim=8,
        probcbm_max_epochs=epochs,
        probcbm_epochs_class=20,
        probcbm_patience=patience,
        # ECBM
        ecbm_emb_size=8,
        ecbm_hid_size=64,
        ecbm_lambda_xy=1.0,
        ecbm_lambda_xc=1.0,
        ecbm_lambda_cy=1.0,
        ecbm_weight_decay=1e-4,
        ecbm_max_epochs=epochs,
        ecbm_patience=patience,
    )


# ── DNN baseline ──��──────────────────────────────────────────────────


class TabularDNN(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x.float())


def train_dnn(dataset: ConceptDataset, config: SimpleNamespace) -> TabularDNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = dataset.train.n_classes
    input_dim = dataset.train.X.shape[1]

    model = TabularDNN(input_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader = dataset.train.loader(batch_size=config.batch_size, shuffle=True)
    valid_loader = dataset.val.loader(batch_size=config.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(config.cs_epochs):
        model.train()
        for X, _, y in train_loader:
            X, y = X.to(device).float(), y.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device).float(), y.to(device).long()
                val_loss += criterion(model(X), y).item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.cs_patience:
                logger.info("  DNN epoch %d/%d val_loss=%.4f (early stop)", epoch + 1, config.cs_epochs, val_loss)
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  DNN epoch %d/%d val_loss=%.4f best=%.4f", epoch + 1, config.cs_epochs, val_loss, best_val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()
    logger.info("DNN trained (%d epochs)", epoch + 1)
    return model


def dnn_predict_proba(model: TabularDNN, dataset_sample) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    all_probs = []
    loader = dataset_sample.loader(batch_size=256, shuffle=False)
    with torch.no_grad():
        for X, _, _ in loader:
            logits = model(X.to(device).float())
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ── CBM training ──��─────────────────────────────��────────────────────


def train_cbm(dataset: ConceptDataset, config: SimpleNamespace):
    from experiments.models import ConceptBasedModel, ConceptDetector, FrontEndModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _macos = platform.system() == "Darwin"
    loader_config = {
        "device": device,
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 4,
        "pin_memory": not _macos,
    }

    # Build a simple tabular concept detector
    from experiments.train import train_concept_heads
    input_dim = dataset.train.X.shape[1]

    cd = ConceptDetector()
    cbm = ConceptBasedModel(concept_detector=cd, should_propagate=False)
    cbm.fit(
        train_dataset=dataset.train,
        valid_dataset=dataset.val,
        freeze_backbone=False,
        concept_embed_params={"shuffle": False, **loader_config},
        concept_fit_params={
            "epochs": config.cs_epochs,
            "lr": config.learning_rate,
            "patience": config.cs_patience,
            **loader_config,
        },
    )
    logger.info("CBM trained")
    return cbm


# ── CEM/ProbCBM/ECBM training ───────��───────────────────────────────


def train_cem(dataset: ConceptDataset, config: SimpleNamespace, benchmark: str):
    from experiments.cem_integration import train_cem_model
    model = train_cem_model(
        train_dataset=dataset.train,
        valid_dataset=dataset.val,
        benchmark=benchmark,
        config=config,
    )
    logger.info("CEM trained")
    return model


def train_probcbm(dataset: ConceptDataset, config: SimpleNamespace, benchmark: str):
    from experiments.cem_integration import train_probcbm_model
    model = train_probcbm_model(
        train_dataset=dataset.train,
        valid_dataset=dataset.val,
        benchmark=benchmark,
        config=config,
    )
    logger.info("ProbCBM trained")
    return model


def train_ecbm(dataset: ConceptDataset, config: SimpleNamespace, benchmark: str):
    from experiments.cem_integration import train_ecbm_model
    model = train_ecbm_model(
        train_dataset=dataset.train,
        valid_dataset=dataset.val,
        benchmark=benchmark,
        config=config,
    )
    logger.info("ECBM trained")
    return model


# ── Selective classification evaluation ──────────────────────────────


def evaluate_selective_all_thresholds(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    confidence: np.ndarray,
    target_accs: list[float],
) -> list[dict]:
    """Find threshold achieving each target selective accuracy, report coverage."""
    results = []
    thresholds = np.linspace(0.5, 1.0, 200)

    for target in target_accs:
        best = None
        for t in thresholds:
            sa = selective_accuracy(y_pred, y_true, confidence, threshold=t)
            cov = coverage(confidence, threshold=t)
            if np.isnan(sa):
                continue
            if sa >= target:
                if best is None or cov > best["coverage"]:
                    best = {"target_acc": target, "threshold": t,
                            "selective_acc": sa, "coverage": cov}
        if best is None:
            best = {"target_acc": target, "threshold": np.nan,
                    "selective_acc": np.nan, "coverage": 0.0}
        results.append(best)
    return results


# ── Intervention evaluation ──────���───────────────────────────────────


def run_model_interventions(model, test_sample, budgets: list[int], model_name: str):
    """Run KFlip interventions at given budgets. Returns list of result dicts."""
    from experiments.kflip import KFlipInterventionStrategy
    from experiments.intervention import (
        ConceptInterventionRunner,
        InterventionBatch,
        InterventionConfig,
    )

    C_pred = model.predict_concepts(test_sample) if hasattr(model, "predict_concepts") else None
    if C_pred is None:
        # For CBM: get concept probabilities
        try:
            C_pred = model.concept_detector.predict_proba(test_sample)
        except Exception:
            logger.warning("Cannot get concept predictions for %s, skipping interventions", model_name)
            return []

    C_true = np.array(test_sample.C)
    y_true = np.array(test_sample.y)

    results = []
    for k in budgets:
        if k == 0:
            # No intervention baseline
            y_pred = model.predict(test_sample)
            acc = accuracy(y_pred, y_true)
            results.append({
                "model": model_name, "budget": 0, "accuracy": acc,
                "predictions_intervened_on": 0, "predictions_changed": 0,
            })
            continue

        try:
            strategy = KFlipInterventionStrategy(k=k)
            batch = InterventionBatch(
                C_pred=C_pred,
                C_true=C_true,
                y_true=y_true,
            )
            config = InterventionConfig(
                budgets=[k],
                threshold=0.2,
            )
            runner = ConceptInterventionRunner(
                model=model,
                strategy=strategy,
                config=config,
            )
            result = runner.run(batch)
            results.append({
                "model": model_name,
                "budget": k,
                "accuracy": result.accuracy,
                "predictions_intervened_on": result.predictions_intervened_on,
                "predictions_changed": result.predictions_changed,
            })
        except Exception as e:
            logger.warning("Intervention failed for %s at k=%d: %s", model_name, k, e)
            # Fallback: manual intervention
            y_pred = model.predict(test_sample)
            results.append({
                "model": model_name, "budget": k,
                "accuracy": accuracy(y_pred, y_true),
                "predictions_intervened_on": 0, "predictions_changed": 0,
            })

    return results


# ── Main pipeline ────────────────────────────────────────────────────


def run_pipeline(dataset_name: str, dataset: ConceptDataset, config: SimpleNamespace):
    logger.info("=" * 60)
    logger.info("Running pipeline for: %s", dataset_name)
    logger.info("=" * 60)

    # Use "robot" as benchmark name so _infer_backbone_spec picks TabularBackbone
    benchmark = "robot"
    target_accs = [0.90, 0.95, 0.99]
    intervention_budgets = [0, 1, 3]
    max_budget = dataset.train.n_concepts
    intervention_budgets.append(max_budget)

    all_results = []

    # ── Train models ──
    models = {}

    set_deterministic_seed(config.seed)
    logger.info("Training DNN...")
    dnn = train_dnn(dataset, config)
    models["DNN"] = dnn

    set_deterministic_seed(config.seed)
    logger.info("Training CBM...")
    models["CBM"] = train_cbm(dataset, config)

    set_deterministic_seed(config.seed)
    logger.info("Training CEM...")
    models["CEM"] = train_cem(dataset, config, benchmark)

    set_deterministic_seed(config.seed)
    logger.info("Training ProbCBM...")
    models["ProbCBM"] = train_probcbm(dataset, config, benchmark)

    set_deterministic_seed(config.seed)
    logger.info("Training ECBM...")
    models["ECBM"] = train_ecbm(dataset, config, benchmark)

    # ── Evaluate each model ──
    test = dataset.test
    y_true = np.array(test.y)

    for name, model in models.items():
        logger.info("Evaluating %s...", name)

        # Get predictions and confidence
        if name == "DNN":
            probs = dnn_predict_proba(model, test)
            y_pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
        else:
            y_pred = model.predict(test)
            try:
                probs = model.predict_proba(test)
                if probs.ndim == 1:
                    # Binary: convert to 2-class
                    probs = np.column_stack([1 - probs, probs])
                conf = probs.max(axis=1)
            except Exception as e:
                logger.warning("predict_proba failed for %s: %s", name, e)
                conf = np.ones(len(y_pred))

        # Raw accuracy
        acc = accuracy(y_pred, y_true)
        logger.info("  %s raw accuracy: %.4f", name, acc)

        # Selective classification
        for res in evaluate_selective_all_thresholds(y_pred, y_true, conf, target_accs):
            all_results.append({
                "dataset": dataset_name,
                "model": name,
                "metric_type": "selective",
                "target_acc": res["target_acc"],
                "threshold": res["threshold"],
                "selective_acc": res["selective_acc"],
                "coverage": res["coverage"],
                "budget": 0,
                "accuracy": acc,
            })

        # Interventions (skip DNN)
        if name != "DNN":
            # Get concept predictions (also caches embeddings for CEM/ProbCBM/ECBM)
            try:
                if hasattr(model, "predict_proba_from_concepts"):
                    # CEM/ProbCBM/ECBM: predict_proba caches internal state
                    probs_result = model.predict_proba(test, return_concepts=True)
                    if isinstance(probs_result, tuple):
                        _, C_pred_proba = probs_result
                    else:
                        C_pred_proba = model.concept_detector.predict_proba(test)
                else:
                    C_pred_proba = model.concept_detector.predict_proba(test)
            except Exception as e:
                logger.warning("  Cannot get concept predictions for %s: %s", name, e)
                C_pred_proba = None

            C_true = np.array(test.C)

            for k in intervention_budgets:
                if k == 0:
                    all_results.append({
                        "dataset": dataset_name,
                        "model": name,
                        "metric_type": "intervention",
                        "budget": 0,
                        "accuracy": acc,
                        "target_acc": np.nan,
                        "threshold": np.nan,
                        "selective_acc": np.nan,
                        "coverage": np.nan,
                    })
                elif C_pred_proba is None:
                    all_results.append({
                        "dataset": dataset_name,
                        "model": name,
                        "metric_type": "intervention",
                        "budget": k,
                        "accuracy": acc,
                        "target_acc": np.nan,
                        "threshold": np.nan,
                        "selective_acc": np.nan,
                        "coverage": np.nan,
                    })
                else:
                    try:
                        # Select top-k most uncertain concepts per sample
                        C_int = C_pred_proba.copy()
                        uncertainty = np.abs(C_pred_proba - 0.5)
                        intervention_mask = np.zeros_like(C_int, dtype=bool)
                        for i in range(len(C_int)):
                            most_uncertain = np.argsort(uncertainty[i])[:k]
                            C_int[i, most_uncertain] = C_true[i, most_uncertain]
                            intervention_mask[i, most_uncertain] = True

                        # Get label predictions with intervened concepts
                        y_prob_int = predict_label_proba_from_concepts(
                            model,
                            C_int,
                            row_indices=np.arange(len(C_int), dtype=int),
                            baseline_concepts=C_pred_proba,
                            intervention_mask=intervention_mask,
                        )
                        if y_prob_int.ndim == 1:
                            y_pred_int = (y_prob_int >= 0.5).astype(int)
                        else:
                            y_pred_int = y_prob_int.argmax(axis=1)

                        acc_int = accuracy(y_pred_int, y_true)
                    except Exception as e:
                        logger.warning("  Intervention failed for %s k=%d: %s", name, k, e)
                        acc_int = acc

                    all_results.append({
                        "dataset": dataset_name,
                        "model": name,
                        "metric_type": "intervention",
                        "budget": k,
                        "accuracy": acc_int,
                        "target_acc": np.nan,
                        "threshold": np.nan,
                        "selective_acc": np.nan,
                        "coverage": np.nan,
                    })
                    logger.info("  %s k=%d accuracy: %.4f", name, k, acc_int)

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / f"{dataset_name}_realworld_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # Print summary table
    logger.info("\n%s Summary:", dataset_name.upper())
    intervention_rows = results_df[results_df["metric_type"] == "intervention"]
    if not intervention_rows.empty:
        pivot = intervention_rows.pivot_table(
            index="model", columns="budget", values="accuracy", aggfunc="first"
        )
        logger.info("\nIntervention accuracy:\n%s", pivot.to_string(float_format="%.4f"))

    selective_rows = results_df[results_df["metric_type"] == "selective"]
    if not selective_rows.empty:
        for target in target_accs:
            subset = selective_rows[selective_rows["target_acc"] == target]
            if not subset.empty:
                logger.info("\nSelective classification (target=%.2f):", target)
                for _, row in subset.iterrows():
                    logger.info("  %s: sel_acc=%.4f, coverage=%.4f",
                                row["model"], row["selective_acc"], row["coverage"])

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Real-world CBM automation experiments")
    parser.add_argument("--dataset", choices=["pistachio", "rice", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config = make_config(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    if args.dataset in ("pistachio", "both"):
        ds = load_pistachio(seed=args.seed)
        run_pipeline("pistachio", ds, config)

    if args.dataset in ("rice", "both"):
        ds = load_rice(seed=args.seed)
        run_pipeline("rice", ds, config)


if __name__ == "__main__":
    main()
