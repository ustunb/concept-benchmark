"""Robot classification benchmark pipeline.

Provides functions to run each stage of the robot benchmark programmatically.
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
    run_alignment,
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
    RobotClassifierCNN,
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


# ── Intervention helper ───────────────────────────────────────────────

def _test_interventions(prob_test, sttngs, acc_det, fe, test):
    """Run interventions for each budget and return results dict.

    Moved from ``scripts/robot_image_pipeline.test_interventions`` —
    duplicate lines removed, uses lazy-imported globals.
    """
    _ensure_intervention_imports()

    intervention_results = {}
    rng = np.random.default_rng(int(sttngs["seed"]))
    budgets = sttngs.get("budget", [1])
    human_acc = sttngs.get("intervention_accuracy", 0.9)
    err_prob = 1.0 - human_acc

    # Create a CBM wrapper for the intervention framework
    cbm = ConceptBasedModel(concept_detector=None, front_end_model=fe)
    runner = ConceptInterventionRunner(cbm)

    for budget in budgets:
        config = InterventionConfig(
            max_concepts_per_instance=budget,
            random_state=int(sttngs["seed"]),
            score_threshold=sttngs.get("intervention_threshold", 1.0),
            noise=1.0 - human_acc,
        )

        strategy = ScoreIntervention()

        # Run intervention
        result = runner.run(
            strategy=strategy,
            config=config,
            dataset=test,
            concept_proba=prob_test,
            labels=test.y.astype(int),
        )

        mask = result.mask
        C_gt = test.C.astype(np.float32)
        C_after = result.C_intervened.copy()

        mistake_draw = rng.random(C_after.shape) < err_prob
        mistakes = mask & mistake_draw
        C_after[mistakes] = 1.0 - C_gt[mistakes]
        result.C_intervened = C_after

        # Extract intervention statistics
        n_intervened = np.sum(result.mask)
        n_samples = prob_test.shape[0]

        intervened_concepts = np.any(result.mask, axis=0)

        C_pred_binary = (result.C_pred >= 0.5).astype(int)
        C_final_binary = (result.C_intervened >= 0.5).astype(int)
        actual_edits_mask = C_pred_binary != C_final_binary
        result.y_prob_after = fe.predict_proba(C_final_binary)
        result.y_pred_after = np.argmax(result.y_prob_after, axis=1)

        y_pred_before = np.argmax(result.y_prob_before, axis=1)

        num_preds_change = int(np.sum(result.y_pred_after != y_pred_before))
        acc_intervened = float((result.y_pred_after == test.y.astype(int)).mean())

        prediction_num_concepts_intervened_on = {
            int(i): int(np.sum(actual_edits_mask[i])) for i in range(n_samples)
        }
        concept_intervention_counts = {
            c: f"{int(np.sum(result.mask[:, i]))} ({int(np.sum(actual_edits_mask[:, i]))})"
            for i, c in enumerate(test.concepts)
            if intervened_concepts[i]
        }

        key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
        intervention_results[key] = {
            "accuracy": acc_intervened,
            "accuracy_gain": acc_intervened - acc_det,
            "predictions_intervened_on": int(np.sum(np.any(result.mask, axis=1))),
            "interventions_rate": float(np.sum(np.any(result.mask, axis=1)) / n_samples),
            "predictions_changed": num_preds_change,
            "avg_edits_per_intervention": float(
                sum(prediction_num_concepts_intervened_on.values())
            )
            / n_samples,
            "total_concept_confirmations": int(n_intervened),
            "total_concept_edits_made": sum(prediction_num_concepts_intervened_on.values()),
            "concept_interventions": concept_intervention_counts,
            "human_accuracy": human_acc,
        }

    return budgets, human_acc, intervention_results


# ── Stage: run_interventions ──────────────────────────────────────────

def run_interventions(
    config: RobotBenchmarkConfig,
    model: Optional[ConceptBasedModel] = None,
    data=None,
) -> pd.DataFrame:
    """Run interventions on the trained CBM and return a results DataFrame.

    Run interventions on the trained CBM and return a results DataFrame.
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

    df_lst = []
    for t in thresholds:
        sttngs["intervention_threshold"] = t
        b, a, r = _test_interventions(
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


# ── Stage: align ─────────────────────────────────────────────────────

def align(
    config: RobotBenchmarkConfig,
    model: Optional[ConceptBasedModel] = None,
    data=None,
) -> dict:
    """Run alignment test on the trained CBM.

    Retrains the frontend with monotonicity (sign) constraints and
    compares original vs constrained accuracy.

    Returns dict with original_accuracy, aligned_accuracy, accuracy_change,
    predictions_changed, aligned_weights.
    """
    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    return run_alignment(
        cbm=model,
        train_dataset=data.training,
        test_dataset=data.test,
        monotonicity_constraints=config.get_alignment_constraints(),
        save_path=config.get_alignment_results_path(),
    )


# ── Stage: collect_results ────────────────────────────────────────────

def _dataset_label(cfg: RobotBenchmarkConfig) -> str:
    """Return a human-readable dataset label for a config."""
    base = "subconcept" if cfg.subconcept else "ideal"
    if cfg.concept_missing_mech != "none":
        return f"{base}_{cfg.concept_missing_mech}"
    return base


def collect_results(
    configs: Optional[List[RobotBenchmarkConfig]] = None,
) -> pd.DataFrame:
    """Aggregate all robot results into a single flat CSV.

    Produces one row per (dataset, model, budget) combination with columns:
      dataset, model, budget, threshold, accuracy, gain,
      predictions_intervened_on, avg_concepts_per_sample, predictions_changed

    Reads saved artifacts only — no model retraining.
    """
    import json

    if configs is None:
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

    rows = []

    # ── DNN accuracy (shared baseline) ───────────────────────────────
    dnn_weights = load(ideal_config.get_model_path("dnn"))
    dnn = RobotClassifierCNN(input_size=ideal_config.input_size).to(device)
    dnn.load_state_dict(dnn_weights)
    test_loader = data.test.loader(shuffle=False, **loader_config)
    dnn_accuracy = compute_accuracy(dnn, test_loader, device)

    # Emit one DNN row per dataset
    dataset_labels_seen = set()
    for cfg in configs:
        label = _dataset_label(cfg)
        if label not in dataset_labels_seen:
            dataset_labels_seen.add(label)
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

    # ── Per-config: CBM, interventions, alignment ────────────────────
    for cfg in configs:
        label = _dataset_label(cfg)

        # CBM no-intervention (k=0)
        cbm = load(cfg.get_model_path("cbm"))
        cbm_acc = float((cbm.predict(data.test) == data.test.y).mean())
        rows.append({
            "dataset": label,
            "model": "cbm",
            "budget": 0,
            "threshold": "",
            "accuracy": round(cbm_acc, 4),
            "gain": round(cbm_acc - dnn_accuracy, 4),
            "predictions_intervened_on": "",
            "avg_concepts_per_sample": "",
            "predictions_changed": "",
        })

        # CBM with interventions (k>0)
        results_path = cfg.get_results_path("cbm")
        if results_path.exists():
            interv_df = pd.read_csv(results_path)
            # Use threshold=0.2 as the canonical threshold for the summary
            t02 = interv_df[interv_df["threshold"] == 0.2]
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
                    "gain": round(acc - dnn_accuracy, 4),
                    "predictions_intervened_on": pio,
                    "avg_concepts_per_sample": avg_cps,
                    "predictions_changed": int(row["predictions_changed"]),
                })

        # Aligned CBM (only for non-missingness configs)
        if cfg.concept_missing_mech == "none":
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
                    "gain": round(aligned_acc - dnn_accuracy, 4),
                    "predictions_intervened_on": "",
                    "avg_concepts_per_sample": "",
                    "predictions_changed": "",
                })

                # Aligned CBM with intervention at k=3
                aligned_weights = align_data.get("aligned_weights")
                if aligned_weights is not None:
                    from concept_benchmark.alignment import align_frontend_weights
                    import copy as _copy

                    # Load the config's own dataset so concept shapes match
                    cfg_data = load(cfg.get_dataset_path())
                    aligned_fe = _copy.deepcopy(cbm.front_end_model)
                    aligned_fe = align_frontend_weights(
                        aligned_fe, list(cfg_data.test.concepts), aligned_weights,
                    )
                    c_preds = cbm.concept_detector.predict(cfg_data.test)
                    sttngs = {
                        "seed": cfg.seed,
                        "budget": [3],
                        "intervention_accuracy": cfg.intervention_accuracy,
                        "intervention_threshold": 0.2,
                    }
                    _, _, int_results = _test_interventions(
                        prob_test=c_preds,
                        sttngs=sttngs,
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
                            "gain": round(float(res["accuracy"]) - dnn_accuracy, 4),
                            "predictions_intervened_on": pio,
                            "avg_concepts_per_sample": avg_cps,
                            "predictions_changed": int(res["predictions_changed"]),
                        })

    final_df = pd.DataFrame(rows)
    out_path = results_dir / "robot_demo_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"Saved {len(final_df)} rows to {out_path}")
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
        stages = ["setup", "cbm", "dnn", "intervene", "align", "collect"]

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

    # Alignment
    if "align" in stages:
        ideal_cfg = RobotBenchmarkConfig.default_ideal()
        ideal_cfg.seed = config.seed
        align(ideal_cfg)

        sub_cfg = RobotBenchmarkConfig.default_subconcept()
        sub_cfg.seed = config.seed
        align(sub_cfg)

    # Collect results
    if "collect" in stages:
        collect_results()
