#!/usr/bin/env python3
"""PPE Compliance automation experiment.

Train DNN, CBM, CEM, ProbCBM, ECBM on construction-site PPE images and
evaluate as automation task (selective classification).

Usage:
    PYTHONPATH=. python scripts/ppe_pipeline.py --seed 42
    PYTHONPATH=. python scripts/ppe_pipeline.py --seed 42 --epochs 30
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

CONCEPTS = ["has_helmet", "has_gloves", "has_vest", "has_boots", "has_goggles"]

IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Dataset loader ───────────────────────────────────────────────────


def load_ppe(seed: int = 42) -> ConceptDataset:
    data_dir = REPO_ROOT / "data" / "ppe"
    csv_path = data_dir / "ppe_cbm.csv"
    df = pd.read_csv(csv_path)

    X = df["image_path"].values.astype(object)
    C = df[CONCEPTS].values.astype(np.float32)
    y = df["compliant"].values.astype(np.int64)

    dataset = ConceptDataset(
        X=X, C=C, y=y,
        meta={
            "classes": ["non_compliant", "compliant"],
            "concepts": CONCEPTS,
            "data_type": "image",
            "resolution": 640,
        },
        base_dir=str(data_dir),
        transform=IMG_TRANSFORM,
    )

    # Use provided splits
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    dataset.sample(
        test_size=test_mask.sum() / len(df),
        val_size=val_mask.sum() / len(df),
        stratify=y,
        seed=seed,
    )

    logger.info(
        "PPE: %d train, %d val, %d test, %d concepts, images 640x640",
        dataset.train.n, dataset.val.n, dataset.test.n, dataset.train.n_concepts,
    )
    return dataset


# ── Config ───────────────────────────────────────────────────────────


def make_config(seed=42, epochs=50, patience=10, batch_size=16, lr=5e-5):
    return SimpleNamespace(
        seed=seed,
        batch_size=batch_size,
        learning_rate=lr,
        cs_epochs=epochs,
        cs_patience=patience,
        cem_emb_size=16,
        cem_training_intervention_prob=0.25,
        cem_concept_loss_weight=1.0,
        cem_task_loss_weight=1.0,
        cem_max_epochs=epochs,
        cem_patience=patience,
        probcbm_hidden_dim=8,
        probcbm_class_hidden_dim=64,
        probcbm_intervention_prob=0.25,
        probcbm_n_samples_inference=50,
        probcbm_latent_dim=8,
        probcbm_max_epochs=epochs,
        probcbm_epochs_class=20,
        probcbm_patience=patience,
        ecbm_emb_size=8,
        ecbm_hid_size=64,
        ecbm_lambda_xy=1.0,
        ecbm_lambda_xc=1.0,
        ecbm_lambda_cy=1.0,
        ecbm_weight_decay=1e-4,
        ecbm_max_epochs=epochs,
        ecbm_patience=patience,
        use_vit_backbone=True,
    )


# ── DNN (ViT backbone) ──────────────────────────────────────────────

from transformers import ViTModel


class ViTDNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for name, p in self.vit.named_parameters():
            if "encoder.layer.10" not in name and "encoder.layer.11" not in name and "layernorm" not in name:
                p.requires_grad = False
        self.head = nn.Linear(768, n_cls)

    def forward(self, x):
        return self.head(self.vit(pixel_values=x).last_hidden_state[:, 0, :])


def train_dnn(dataset: ConceptDataset, config: SimpleNamespace):
    from concept_benchmark.utils import determine_device

    device = determine_device()
    n_classes = dataset.train.n_classes

    model = ViTDNN(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

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
            X, y = X.to(device), y.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device).long()
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


def dnn_predict_proba(model, dataset_sample) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    loader = dataset_sample.loader(batch_size=32, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for X, _, _ in loader:
            X = X.to(device)
            probs = torch.softmax(model(X), dim=-1).cpu().numpy()
            all_probs.append(probs)
    model.cpu()
    return np.concatenate(all_probs, axis=0)


# ── CBM (ViT concept detector) ──────────────────────────────────────


def train_cbm(dataset: ConceptDataset, config: SimpleNamespace):
    from experiments.models import ConceptBasedModel, ConceptDetector, RobotViTConceptClassifier
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
    concept_model = RobotViTConceptClassifier(num_concepts=n_concepts)
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


# ── CEM / ProbCBM / ECBM ────────────────────────────────────────────


def train_cem(dataset, config, benchmark="robot"):
    from experiments.cem_integration import train_cem_model
    from concept_benchmark.utils import determine_device
    device = determine_device()
    _macos = platform.system() == "Darwin"
    model = train_cem_model(
        train_dataset=dataset.train, valid_dataset=dataset.val,
        benchmark=benchmark, config=config, device=device,
        num_workers=0 if _macos else 4, pin_memory=not _macos,
    )
    logger.info("CEM trained")
    return model


def train_probcbm(dataset, config, benchmark="robot"):
    from experiments.cem_integration import train_probcbm_model
    from concept_benchmark.utils import determine_device
    device = determine_device()
    _macos = platform.system() == "Darwin"
    model = train_probcbm_model(
        train_dataset=dataset.train, valid_dataset=dataset.val,
        benchmark=benchmark, config=config, device=device,
        num_workers=0 if _macos else 4, pin_memory=not _macos,
    )
    logger.info("ProbCBM trained")
    return model


def train_ecbm(dataset, config, benchmark="robot"):
    from experiments.cem_integration import train_ecbm_model
    from concept_benchmark.utils import determine_device
    device = determine_device()
    _macos = platform.system() == "Darwin"
    model = train_ecbm_model(
        train_dataset=dataset.train, valid_dataset=dataset.val,
        benchmark=benchmark, config=config, device=device,
        num_workers=0 if _macos else 4, pin_memory=not _macos,
    )
    logger.info("ECBM trained")
    return model


# ── Selective classification evaluation ──────────────────────────────


def evaluate_selective_all_thresholds(y_pred, y_true, confidence, target_accs):
    unique_confs = np.unique(confidence)
    grid = np.linspace(0.5, 1.0, 500)
    thresholds = np.unique(np.concatenate([unique_confs, grid]))
    thresholds.sort()

    results = []
    for target in target_accs:
        best = {"target_acc": target, "threshold": np.nan,
                "selective_acc": np.nan, "coverage": 0.0}
        for t in thresholds:
            mask = confidence >= t
            if mask.sum() == 0:
                continue
            sa = (y_pred[mask] == y_true[mask]).mean()
            cov = mask.mean()
            if sa >= target and cov > best["coverage"]:
                best = {"target_acc": target, "threshold": t,
                        "selective_acc": sa, "coverage": cov}
        results.append(best)
    return results


# ── Main pipeline ────────────────────────────────────────────────────


def run_pipeline(dataset: ConceptDataset, config: SimpleNamespace, skip_training: bool = False):
    logger.info("=" * 60)
    logger.info("Running PPE automation pipeline")
    logger.info("=" * 60)

    benchmark = "robot"  # Use robot backbone spec (image-based)
    target_accs = [0.90, 0.95, 0.99]
    model_dir = RESULTS_DIR / "ppe_models"
    model_dir.mkdir(exist_ok=True)

    import pickle

    def save_model(name, model):
        path = model_dir / f"{name}.pkl"
        if name == "DNN":
            torch.save({"state_dict": model.state_dict(), "n_cls": len(dataset.meta["classes"])}, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(model, f)
        logger.info("  Saved %s to %s", name, path)

    def load_model(name):
        path = model_dir / f"{name}.pkl"
        if not path.exists():
            return None
        if name == "DNN":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = ViTDNN(ckpt["n_cls"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
        else:
            with open(path, "rb") as f:
                model = pickle.load(f)
        logger.info("  Loaded %s from %s", name, path)
        return model

    models = {}
    train_fns = {
        "DNN": lambda: train_dnn(dataset, config),
        "CBM": lambda: train_cbm(dataset, config),
        "CEM": lambda: train_cem(dataset, config, benchmark),
        "ProbCBM": lambda: train_probcbm(dataset, config, benchmark),
        "ECBM": lambda: train_ecbm(dataset, config, benchmark),
    }

    for name in ["DNN", "CBM", "CEM", "ProbCBM", "ECBM"]:
        if skip_training:
            model = load_model(name)
            if model is not None:
                models[name] = model
                continue
            logger.warning("  No saved model for %s, training from scratch", name)

        set_deterministic_seed(config.seed)
        logger.info("Training %s...", name)
        models[name] = train_fns[name]()
        save_model(name, models[name])

    # Evaluate
    test = dataset.test
    y_true = np.array(test.y)
    all_results = []

    for name, model in models.items():
        logger.info("Evaluating %s...", name)

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

        acc = accuracy(y_pred, y_true)
        logger.info("  %s raw accuracy: %.4f", name, acc)

        for res in evaluate_selective_all_thresholds(y_pred, y_true, conf, target_accs):
            all_results.append({
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
        # Logic: at k=0, find threshold t0 for each target. Automated set = conf >= t0.
        # At k>0, only intervene on ABSTAINED samples, re-predict those only.
        # Automated predictions are FIXED. Coverage can only grow.
        intervention_budgets = [0, 1, 3, dataset.train.n_concepts]
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

            # Find k=0 thresholds for each target
            k0_results = evaluate_selective_all_thresholds(y_pred, y_true, conf, target_accs)

            for k in intervention_budgets:
                if k == 0 or C_pred_proba is None:
                    # k=0: just record the selective results
                    for res in k0_results:
                        all_results.append({
                            "model": name,
                            "metric_type": "intervention_selective",
                            "target_acc": res["target_acc"],
                            "threshold": res["threshold"],
                            "selective_acc": res["selective_acc"],
                            "coverage": res["coverage"],
                            "budget": k,
                            "accuracy": acc,
                        })
                    if k == 0:
                        logger.info("  %s k=%d accuracy: %.4f", name, k, acc)
                    continue

                # For each target, use the k=0 threshold and only re-predict abstained
                for res0 in k0_results:
                    t0 = res0["threshold"]
                    if np.isnan(t0):
                        all_results.append({
                            "model": name, "metric_type": "intervention_selective",
                            "target_acc": res0["target_acc"], "threshold": t0,
                            "selective_acc": np.nan, "coverage": 0.0,
                            "budget": k, "accuracy": acc,
                        })
                        continue

                    # Automated set (FIXED, never re-predicted)
                    automated = conf >= t0
                    abstained = ~automated

                    # Intervene on abstained samples only
                    abstained_idx = np.where(abstained)[0]
                    if len(abstained_idx) == 0:
                        # Everything already automated
                        all_results.append({
                            "model": name, "metric_type": "intervention_selective",
                            "target_acc": res0["target_acc"], "threshold": t0,
                            "selective_acc": res0["selective_acc"],
                            "coverage": res0["coverage"],
                            "budget": k, "accuracy": acc,
                        })
                        continue

                    try:
                        C_int = C_pred_proba[abstained_idx].copy()
                        uncertainty = np.abs(C_int - 0.5)
                        int_mask = np.zeros_like(C_int, dtype=bool)
                        for i in range(len(C_int)):
                            most_uncertain = np.argsort(uncertainty[i])[:k]
                            C_int[i, most_uncertain] = C_true[abstained_idx[i], most_uncertain]
                            int_mask[i, most_uncertain] = True

                        y_prob_abs = predict_label_proba_from_concepts(
                            model, C_int,
                            row_indices=abstained_idx.astype(int),
                            baseline_concepts=C_pred_proba[abstained_idx],
                            intervention_mask=int_mask,
                        )
                        if y_prob_abs.ndim == 1:
                            conf_abs = np.maximum(y_prob_abs, 1 - y_prob_abs)
                            y_pred_abs = (y_prob_abs >= 0.5).astype(int)
                        else:
                            conf_abs = y_prob_abs.max(axis=1)
                            y_pred_abs = y_prob_abs.argmax(axis=1)

                        # Newly automated: abstained samples now confident enough
                        newly_automated = conf_abs >= t0

                        # Combined automated set
                        final_pred = y_pred.copy()
                        final_pred[abstained_idx[newly_automated]] = y_pred_abs[newly_automated]
                        final_automated = automated.copy()
                        final_automated[abstained_idx[newly_automated]] = True

                        n_total = len(y_true)
                        cov_after = final_automated.sum() / n_total
                        sa_after = (final_pred[final_automated] == y_true[final_automated]).mean() if final_automated.any() else np.nan
                        acc_int = (final_pred == y_true).mean()

                    except Exception as e:
                        logger.warning("  Intervention failed for %s k=%d: %s", name, k, e)
                        sa_after = res0["selective_acc"]
                        cov_after = res0["coverage"]
                        acc_int = acc

                    all_results.append({
                        "model": name, "metric_type": "intervention_selective",
                        "target_acc": res0["target_acc"], "threshold": t0,
                        "selective_acc": sa_after, "coverage": cov_after,
                        "budget": k, "accuracy": acc_int,
                    })

                logger.info("  %s k=%d accuracy: %.4f", name, k, acc_int)

    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "ppe_automation_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # Print summary
    logger.info("\nPPE Automation Summary:")
    for target in target_accs:
        subset = results_df[(results_df["target_acc"] == target) & (results_df["metric_type"] == "selective")]
        logger.info("\nSelective classification (target=%.2f):", target)
        for _, row in subset.iterrows():
            logger.info("  %s: sel_acc=%.4f, coverage=%.4f",
                        row["model"], row["selective_acc"], row["coverage"])

    int_rows = results_df[results_df["metric_type"] == "intervention"]
    if not int_rows.empty:
        pivot = int_rows.pivot_table(index="model", columns="budget", values="accuracy", aggfunc="first")
        logger.info("\nIntervention accuracy:\n%s", pivot.to_string(float_format="%.4f"))

    return results_df


def main():
    parser = argparse.ArgumentParser(description="PPE Compliance automation experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--skip-training", action="store_true",
                        help="Load saved models instead of retraining")
    args = parser.parse_args()

    config = make_config(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    ds = load_ppe(seed=args.seed)
    run_pipeline(ds, config, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
