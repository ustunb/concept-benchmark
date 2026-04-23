#!/usr/bin/env python3
"""Real-world automation experiments: Pistachio + Rice.

Train DNN, CBM, CEM, ProbCBM, ECBM on real-world image datasets and
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
import torchvision.transforms as T

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

# Image size for all models (resize to this)
IMG_SIZE = 224

# ImageNet normalization
IMG_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


# ── Dataset loaders ──────────────────────────────────────────────────


def load_pistachio(seed: int = 42) -> ConceptDataset:
    data_dir = REPO_ROOT / "data" / "pistachio"
    csv_path = data_dir / "pistachio_cbm.csv"
    df = pd.read_csv(csv_path)

    # Build image paths by scanning actual files on disk
    kirmizi_dir = data_dir / "kirmizi" / "images"
    siirt_dir = data_dir / "siirt" / "images"
    kirmizi_files = sorted(kirmizi_dir.glob("*.jpg")) if kirmizi_dir.exists() else []
    siirt_files = sorted(siirt_dir.glob("*.jpg")) if siirt_dir.exists() else []

    image_paths = []
    k_idx, s_idx = 0, 0
    for _, row in df.iterrows():
        if row["label"] == 1:
            image_paths.append(str(kirmizi_files[k_idx].relative_to(data_dir)))
            k_idx += 1
        else:
            image_paths.append(str(siirt_files[s_idx].relative_to(data_dir)))
            s_idx += 1

    X = np.array(image_paths, dtype=object)
    C = df[BINARY_CONCEPTS].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    dataset = ConceptDataset(
        X=X, C=C, y=y,
        meta={
            "classes": ["Siirt", "Kirmizi"],
            "concepts": BINARY_CONCEPTS,
            "data_type": "image",
            "resolution": 600,
        },
        base_dir=str(data_dir),
        transform=IMG_TRANSFORM,
    )
    dataset.sample(test_size=0.2, val_size=0.2, stratify=y, seed=seed)
    logger.info("Pistachio: %d train, %d val, %d test, %d concepts, images %dx%d",
                dataset.train.n, dataset.val.n, dataset.test.n,
                dataset.train.n_concepts, 600, 600)
    return dataset


def load_rice(seed: int = 42) -> ConceptDataset:
    data_dir = REPO_ROOT / "data" / "rice"
    csv_path = data_dir / "rice_msc_cbm.csv"
    img_dir = data_dir / "rice_images" / "Rice_Image_Dataset"
    df = pd.read_csv(csv_path)

    # Drop rows with NaN
    concept_cols = BINARY_CONCEPTS_UPPER
    df = df.dropna(subset=concept_cols + ["label"]).reset_index(drop=True)

    classes = sorted(df["CLASS"].unique())
    class_to_idx = {name: i for i, name in enumerate(classes)}

    # Build image paths by scanning actual files on disk
    class_files = {}
    for cls in classes:
        cls_dir = img_dir / cls
        if cls_dir.exists():
            class_files[cls] = sorted(cls_dir.glob("*.jpg"))
        else:
            class_files[cls] = []

    image_paths = []
    class_counters = {c: 0 for c in classes}
    for _, row in df.iterrows():
        cls = row["CLASS"]
        idx = class_counters[cls]
        image_paths.append(str(class_files[cls][idx].relative_to(img_dir)))
        class_counters[cls] += 1

    X = np.array(image_paths, dtype=object)
    C = df[concept_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    dataset = ConceptDataset(
        X=X, C=C, y=y,
        meta={
            "classes": classes,
            "concepts": concept_cols,
            "data_type": "image",
            "resolution": 250,
        },
        base_dir=str(img_dir),
        transform=IMG_TRANSFORM,
    )
    dataset.sample(test_size=0.2, val_size=0.2, stratify=y, seed=seed)
    logger.info("Rice: %d train, %d val, %d test, %d concepts, %d classes, images %dx%d",
                dataset.train.n, dataset.val.n, dataset.test.n,
                dataset.train.n_concepts, len(classes), 250, 250)
    return dataset


# ── Config namespace ─────────────────────────────────────────────────


def make_config(seed: int = 42, epochs: int = 50, patience: int = 10,
                batch_size: int = 32, lr: float = 1e-3) -> SimpleNamespace:
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


# ── DNN baseline (CNN on images) ─────────────────────────────────────


def train_dnn(dataset: ConceptDataset, config: SimpleNamespace):
    from experiments.models import RobotClassifierCNN
    from concept_benchmark.utils import determine_device

    device = determine_device()
    n_classes = dataset.train.n_classes

    model = RobotClassifierCNN(num_classes=n_classes, input_size=IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    _macos = platform.system() == "Darwin"
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 4,
        "pin_memory": not _macos,
    }
    train_loader = dataset.train.loader(shuffle=True, **loader_kwargs)
    valid_loader = dataset.val.loader(shuffle=False, **loader_kwargs)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(config.cs_epochs):
        model.train()
        for X, _, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            if n_classes <= 2:
                loss = criterion(outputs.squeeze(), y.float())
            else:
                loss = criterion(outputs, y.long())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                if n_classes <= 2:
                    batch_loss = criterion(outputs.squeeze(), y.float())
                else:
                    batch_loss = criterion(outputs, y.long())
                val_loss += batch_loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.cs_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()
    logger.info("DNN trained (%d epochs)", epoch + 1)
    return model


def dnn_predict_proba(model, dataset_sample) -> np.ndarray:
    from concept_benchmark.utils import determine_device
    device = determine_device()
    model = model.to(device)
    model.eval()
    all_probs = []
    loader = dataset_sample.loader(batch_size=64, shuffle=False)
    n_classes = dataset_sample.n_classes
    with torch.no_grad():
        for X, _, _ in loader:
            X = X.to(device)
            outputs = model(X)
            if n_classes <= 2:
                p = outputs.squeeze().cpu()
                probs = torch.stack([1 - p, p], dim=-1).numpy()
            else:
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            all_probs.append(probs)
    model.cpu()
    return np.concatenate(all_probs, axis=0)


# ── CBM training (CNN concept detector) ──────────────────────────────


def train_cbm(dataset: ConceptDataset, config: SimpleNamespace):
    from experiments.models import ConceptBasedModel, ConceptDetector, RobotConceptClassifier
    from concept_benchmark.utils import determine_device

    device = determine_device()
    _macos = platform.system() == "Darwin"
    loader_config = {
        "device": device,
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 4,
        "pin_memory": not _macos,
    }

    n_concepts = dataset.train.n_concepts
    concept_model = RobotConceptClassifier(num_concepts=n_concepts, input_size=IMG_SIZE)
    cd = ConceptDetector(model=concept_model)
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


# ── CEM/ProbCBM/ECBM training ────────────────────────────────────────


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
    # Use actual unique confidence values for precise threshold search
    unique_conf = np.unique(confidence)
    thresholds = np.concatenate([unique_conf, np.linspace(0.5, 1.0, 500)])
    thresholds = np.unique(thresholds)

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


# ── Main pipeline ────────────────────────────────────────────────────


def run_pipeline(dataset_name: str, dataset: ConceptDataset, config: SimpleNamespace):
    logger.info("=" * 60)
    logger.info("Running pipeline for: %s", dataset_name)
    logger.info("=" * 60)

    # Use "robot" as benchmark name so _infer_backbone_spec picks RobotImageBackbone
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
                if isinstance(probs, tuple):
                    probs = probs[0]
                if probs.ndim == 1:
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
            try:
                if hasattr(model, "predict_proba_from_concepts"):
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
                        "dataset": dataset_name, "model": name,
                        "metric_type": "intervention", "budget": 0,
                        "accuracy": acc, "target_acc": np.nan,
                        "threshold": np.nan, "selective_acc": np.nan,
                        "coverage": np.nan,
                    })
                elif C_pred_proba is None:
                    all_results.append({
                        "dataset": dataset_name, "model": name,
                        "metric_type": "intervention", "budget": k,
                        "accuracy": acc, "target_acc": np.nan,
                        "threshold": np.nan, "selective_acc": np.nan,
                        "coverage": np.nan,
                    })
                else:
                    try:
                        C_int = C_pred_proba.copy()
                        uncertainty = np.abs(C_pred_proba - 0.5)
                        intervention_mask = np.zeros_like(C_int, dtype=bool)
                        for i in range(len(C_int)):
                            most_uncertain = np.argsort(uncertainty[i])[:k]
                            C_int[i, most_uncertain] = C_true[i, most_uncertain]
                            intervention_mask[i, most_uncertain] = True

                        y_prob_int = predict_label_proba_from_concepts(
                            model, C_int,
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
                        "dataset": dataset_name, "model": name,
                        "metric_type": "intervention", "budget": k,
                        "accuracy": acc_int, "target_acc": np.nan,
                        "threshold": np.nan, "selective_acc": np.nan,
                        "coverage": np.nan,
                    })
                    logger.info("  %s k=%d accuracy: %.4f", name, k, acc_int)

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / f"{dataset_name}_realworld_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # Print summary
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
    parser.add_argument("--batch-size", type=int, default=32)
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
