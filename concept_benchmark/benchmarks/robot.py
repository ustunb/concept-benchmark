"""Robot classification benchmark pipeline.

Provides functions to run each stage of the robot benchmark programmatically.
Extracted from scripts/robot_demo/*.py — same logic, no subprocess calls.
"""
from __future__ import annotations

import copy
import platform
from itertools import product
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from concept_benchmark.benchmarks._common import (
    compute_accuracy,
    create_skewed_splits_full,
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
)
from concept_benchmark.config import (
    INPUT_MAP,
    MISSING_PROP,
    SUBCONCEPT_DROP,
    RobotBenchmarkConfig,
)
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.models import (
    ConceptBasedModel,
    ConceptDetector,
    RobotConceptClassifier,
)
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_synthetic_dataset

# Lazy import to avoid circular deps — intervention modules
_intervention_imported = False


def _ensure_intervention_imports():
    global _intervention_imported
    if not _intervention_imported:
        global ConceptInterventionRunner, InterventionConfig, ScoreIntervention
        from concept_benchmark.intervention import (
            ConceptInterventionRunner,
            InterventionConfig,
        )
        from concept_benchmark.kflip import KFlipInterventionStrategy as ScoreIntervention

        _intervention_imported = True


# Re-export the CNN for backward compat
from scripts.robot_demo.utils import RobotClassifierCNN  # noqa: E402


# ── Stage: setup_dataset ──────────────────────────────────────────────

def setup_dataset(config: RobotBenchmarkConfig):
    """Generate robot dataset, apply skewed splits, and save.

    Returns the saved ConceptDataset.
    """
    settings = config.to_dict()
    data = create_synthetic_dataset(**settings)
    tf = transforms.Compose([transforms.ToTensor()])
    data.transform = tf
    data.generate_cvindices(seed=config.seed)

    rng = np.random.default_rng(config.seed)
    sk_data = create_skewed_splits_full(dataset=data, rng=rng, **settings)
    save(sk_data, config.get_dataset_path(), overwrite=True)
    return sk_data


# ── Stage: train_cbm ──────────────────────────────────────────────────

def train_cbm(
    config: RobotBenchmarkConfig,
    data=None,
) -> ConceptBasedModel:
    """Train a ConceptBasedModel (concept detector + frontend).

    Returns the trained CBM.
    """
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    # Apply concept missingness if configured
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

    _macos = platform.system() == "Darwin"
    loader_config = {
        "device": device,
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }
    torch.manual_seed(config.seed)

    cd = ConceptDetector(
        model=RobotConceptClassifier(
            num_concepts=data.training.n_concepts,
            input_size=config.input_size,
        )
    )
    cbm = ConceptBasedModel(concept_detector=cd)
    cbm.fit(
        train_dataset=data.training,
        valid_dataset=data.validation,
        freeze=False,
        concept_embed_params={"shuffle": False, **loader_config},
        fit_params={
            "epochs": config.epochs,
            "lr": config.lr,
            "patience": config.patience,
            **loader_config,
        },
    )

    test_pred = cbm.predict(data.test)
    print("Test Accuracy:", np.mean(test_pred == data.test.y))

    save(cbm, config.get_model_path("cbm"), overwrite=True)
    return cbm


# ── Stage: train_dnn ──────────────────────────────────────────────────

def train_dnn(
    config: RobotBenchmarkConfig,
    data=None,
) -> dict:
    """Train an end-to-end DNN baseline.

    Returns the best state_dict.
    """
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    torch.manual_seed(config.seed)
    model = RobotClassifierCNN(input_size=config.input_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"with best val loss {best_val_loss:.6f}"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_acc = compute_accuracy(model, train_loader, device=device)
    valid_acc = compute_accuracy(model, valid_loader, device=device)
    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {valid_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    weights = best_state_dict if best_state_dict is not None else model.state_dict()
    save(weights, config.get_model_path("dnn"), overwrite=True)
    return weights


# ── Stage: run_interventions ──────────────────────────────────────────

def run_interventions(
    config: RobotBenchmarkConfig,
    model: Optional[ConceptBasedModel] = None,
    data=None,
) -> pd.DataFrame:
    """Run interventions on the trained CBM and return a results DataFrame.

    Mirrors the logic in scripts/robot_demo/intervene.py.
    """
    _ensure_intervention_imports()
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    budgets = list(config.intervention_budgets) + [data.n_concepts]
    thresholds = config.intervention_thresholds

    sttngs = {
        "seed": config.seed,
        "budget": budgets,
        "intervention_accuracy": config.intervention_accuracy,
        "intervention_threshold": thresholds[0] if thresholds else 0.2,
    }

    c_preds = model.concept_detector.predict(data.test)
    acc = (model.predict(data.test) == data.test.y).mean().item()

    METRIC_COLS = [
        "accuracy",
        "predictions_intervened_on",
        "predictions_changed",
        "total_concept_confirmations",
        "total_concept_edits_made",
    ]
    COLS = ["budget", "threshold"] + METRIC_COLS

    # Import test_interventions from the image pipeline for now
    from scripts.robot_image_pipeline import test_interventions

    df_lst = []
    for t in thresholds:
        sttngs["intervention_threshold"] = t
        b, a, r = test_interventions(
            prob_test=c_preds,
            sttngs=sttngs,
            acc_det=acc,
            fe=model.front_end_model,
            test=data.test,
        )
        df_lst.append(
            pd.DataFrame(r)
            .T.assign(budget=budgets)
            .assign(threshold=t)
            .reset_index(drop=True)[COLS]
        )

    results_df = pd.concat(df_lst, axis=0).reset_index(drop=True)
    results_df["data_name"] = "subconcept" if config.subconcept else "ideal"
    results_df["n"] = data.test.n
    results_df["concept_missing"] = config.concept_missing
    results_df["concept_missing_mech"] = config.concept_missing_mech
    results_df.to_csv(config.get_results_path("cbm"), index=False)
    return results_df


# ── Stage: collect_results ────────────────────────────────────────────

def collect_results(
    configs: Optional[List[RobotBenchmarkConfig]] = None,
) -> pd.DataFrame:
    """Aggregate all robot results into a single DataFrame.

    Mirrors scripts/robot_demo/calc_metrics.py.
    """
    if configs is None:
        # Default: all combinations from the paper
        configs = []
        for subconcept in [False, True]:
            for missing, missing_mech in [
                (0.0, "none"),
                (MISSING_PROP, "mcar"),
                (MISSING_PROP, "mnar"),
            ]:
                cfg = RobotBenchmarkConfig(
                    subconcept=subconcept,
                    concept_missing=missing,
                    concept_missing_mech=missing_mech,
                )
                if subconcept:
                    cfg.drop_concepts = list(SUBCONCEPT_DROP)
                configs.append(cfg)

    ideal_config = RobotBenchmarkConfig.default_ideal()
    data = load(ideal_config.get_dataset_path())
    device = determine_device()
    loader_config = get_loader_config(device)

    acc_rows = []

    # DNN accuracy
    dnn_weights = load(ideal_config.get_model_path("dnn"))
    dnn = RobotClassifierCNN(input_size=ideal_config.input_size).to(device)
    dnn.load_state_dict(dnn_weights)
    test_loader = data.test.loader(shuffle=False, **loader_config)
    dnn_accuracy = compute_accuracy(dnn, test_loader, device)
    acc_rows.append(["ideal", 0.0, "none", "dnn", "accuracy", dnn_accuracy])
    acc_rows.append(["subconcept", 0.0, "none", "dnn", "accuracy", dnn_accuracy])

    COLS = [
        "accuracy",
        "predictions_intervened_on",
        "predictions_changed",
        "total_concept_confirmations",
    ]

    interv_lst = []
    for cfg in configs:
        data_name = "subconcept" if cfg.subconcept else "ideal"
        cbm = load(cfg.get_model_path("cbm"))
        cbm_acc = (cbm.predict(data.test) == data.test.y).mean().item()
        acc_rows.append([
            data_name, cfg.concept_missing, cfg.concept_missing_mech,
            "cbm_no_int", "accuracy", cbm_acc,
        ])

        results_path = cfg.get_results_path("cbm")
        if results_path.exists():
            metrics = pd.read_csv(results_path)
            metrics = metrics.melt(
                id_vars=[
                    "data_name", "concept_missing", "concept_missing_mech",
                    "budget", "threshold",
                ],
                value_vars=COLS,
                var_name="metric",
                value_name="value",
            )
            metrics["model"] = "cbm_with_int_" + metrics["budget"].astype(str)
            interv_lst.append(metrics)

    acc_df = pd.DataFrame(
        acc_rows,
        columns=[
            "data_name", "concept_missing", "concept_missing_mech",
            "model", "metric", "value",
        ],
    )
    if interv_lst:
        interv_df = pd.concat(interv_lst, ignore_index=True).reset_index(drop=True)
        final_df = pd.concat([acc_df, interv_df], ignore_index=True).reset_index(drop=True)
    else:
        final_df = acc_df

    final_df.to_csv(results_dir / "robot_demo_results.csv", index=False)
    return final_df


# ── Stage: run (orchestrator) ─────────────────────────────────────────

def run(
    config: Optional[RobotBenchmarkConfig] = None,
    stages: Optional[List[str]] = None,
    missing: bool = True,
) -> None:
    """Run the full robot benchmark pipeline.

    Args:
        config: Benchmark configuration. Defaults to ideal.
        stages: List of stages to run. Default: all.
        missing: Whether to also run MCAR/MNAR variants.
    """
    patch_macos_dataloader()

    if config is None:
        config = RobotBenchmarkConfig.default_ideal()
    if stages is None:
        stages = ["setup", "cbm", "dnn", "intervene"]

    # Setup
    if "setup" in stages:
        # Ideal
        ideal_cfg = RobotBenchmarkConfig.default_ideal()
        ideal_cfg.seed = config.seed
        setup_dataset(ideal_cfg)

        # Subconcept
        sub_cfg = RobotBenchmarkConfig.default_subconcept()
        sub_cfg.seed = config.seed
        setup_dataset(sub_cfg)

    # CBM training
    if "cbm" in stages:
        ideal_cfg = RobotBenchmarkConfig.default_ideal()
        ideal_cfg.seed = config.seed
        train_cbm(ideal_cfg)

        sub_cfg = RobotBenchmarkConfig.default_subconcept()
        sub_cfg.seed = config.seed
        train_cbm(sub_cfg)

        if missing:
            for mech in ["mcar", "mnar"]:
                for subconcept in [False, True]:
                    cfg = (
                        RobotBenchmarkConfig.default_subconcept()
                        if subconcept
                        else RobotBenchmarkConfig.default_ideal()
                    )
                    cfg.seed = config.seed
                    cfg.concept_missing = MISSING_PROP
                    cfg.concept_missing_mech = mech
                    train_cbm(cfg)

    # DNN training
    if "dnn" in stages:
        ideal_cfg = RobotBenchmarkConfig.default_ideal()
        ideal_cfg.seed = config.seed
        train_dnn(ideal_cfg)

    # Interventions
    if "intervene" in stages:
        ideal_cfg = RobotBenchmarkConfig.default_ideal()
        ideal_cfg.seed = config.seed
        run_interventions(ideal_cfg)

        sub_cfg = RobotBenchmarkConfig.default_subconcept()
        sub_cfg.seed = config.seed
        run_interventions(sub_cfg)

        if missing:
            for mech in ["mcar", "mnar"]:
                for subconcept in [False, True]:
                    cfg = (
                        RobotBenchmarkConfig.default_subconcept()
                        if subconcept
                        else RobotBenchmarkConfig.default_ideal()
                    )
                    cfg.seed = config.seed
                    cfg.concept_missing = MISSING_PROP
                    cfg.concept_missing_mech = mech
                    run_interventions(cfg)
