"""Sudoku validation benchmark pipeline.

Provides functions to run each stage of the sudoku benchmark programmatically.

Usage:
    python scripts/sudoku_pipeline.py --seed 171
    python scripts/sudoku_pipeline.py --seed 171 --stages cs dnn selective
    python scripts/sudoku_pipeline.py --config my_config.yaml
"""

from __future__ import annotations

import copy
import logging
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from concept_benchmark.utils import (
    compute_accuracy,
    determine_device,
    get_loader_config,
    parse_budgets,
    patch_macos_dataloader,
    set_deterministic_seed,
)
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.ext.fileutils import load, save
from experiments.cem_integration import train_cem_model, train_probcbm_model
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
)
from experiments.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionConfig,
)

logger = logging.getLogger(__name__)


def _selected_cs_key(config: SudokuBenchmarkConfig) -> str:
    family = str(getattr(config, "cbm_family", "cbm"))
    return "cs" if family == "cbm" else family


# ── Stage: setup_dataset ──────────────────────────────────────────────


def setup_dataset(config: SudokuBenchmarkConfig) -> None:
    """Generate sudoku dataset (image + tabular + OCR sidecar)."""
    from concept_benchmark.synthetic.sudoku_ocr.make_ocr_dataset import (
        generate_sudoku_pipeline_data,
    )

    generate_sudoku_pipeline_data(config)


# ── Stage: train_ocr ──────────────────────────────────────────────────


def train_ocr(config: SudokuBenchmarkConfig) -> None:
    """Train OCR digit recognizer."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "concept_benchmark.synthetic.sudoku_ocr.train_ocr_fast",
        "--seed",
        str(config.seed),
        "--max-corrupt",
        str(config.max_cell_swaps),
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
    set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        tab_dir = config.get_dataset_path(data_type="tabular")
        data = load(tab_dir / "sudoku_dataset.pkl")
        data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=config.seed)

    model_key = _selected_cs_key(config)
    if model_key != "cs":
        if config.data_type != "tabular":
            raise NotImplementedError(
                "Sudoku CEM/ProbCBM support is implemented only for the tabular variant."
            )
        loader_config = get_loader_config()
        trainer_fn = train_cem_model if model_key == "cem" else train_probcbm_model
        model = trainer_fn(
            train_dataset=data.train,
            valid_dataset=data.validation,
            benchmark="sudoku",
            config=config,
            device=device,
            num_workers=loader_config["num_workers"],
            pin_memory=loader_config["pin_memory"],
        )
        test_pred = model.predict(data.test)
        logger.info("Test Accuracy: %s", np.mean(test_pred == data.test.y))
        save(model, config.get_model_path(model_key, data_type="tabular"), overwrite=True)
        return model

    # Missingness can be applied here if needed:
    # data.sample_concept_missingness(p=0.2, mechanism="mcar", rng=config.seed)

    _macos = platform.system() == "Darwin"
    loader_config = {
        "device": device,
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }

    from experiments.models import (
        GroupPoolingConceptSudokuCNN as SudokuConceptModel,
    )

    model = SudokuConceptModel()
    cd = ConceptDetector(model=model)
    cbm = ConceptBasedModel(concept_detector=cd, should_propagate=True)
    cbm.fit(
        train_dataset=data.train,
        valid_dataset=data.validation,
        freeze_backbone=False,
        concept_embed_params={"shuffle": False, **loader_config},
        concept_fit_params={
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
    set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        tab_dir = config.get_dataset_path(data_type="tabular")
        data = load(tab_dir / "sudoku_dataset.pkl")
        data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=config.seed)

    from experiments.models import SudokuValidatorCNN as DNNSudokuModel

    model = DNNSudokuModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader_config = get_loader_config()
    train_loader = data.train.loader(shuffle=True, **loader_config)
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
                    epoch + 1,
                    best_val_loss,
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
    cs_model: ConceptBasedModel | None = None,
    data=None,
) -> pd.DataFrame:
    """Run conceptual safeguards interventions on the sudoku CS model.

    Returns the intervention results DataFrame.
    """
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        if config.data_type == "image":
            img_dir = config.get_dataset_path(data_type="image")
            data = load(img_dir / "ocr_inferred_full_dataset.pkl")
        else:
            tab_dir = config.get_dataset_path(data_type="tabular")
            data = load(tab_dir / "sudoku_dataset.pkl")
        data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=config.seed)

    if cs_model is None:
        cs_model = load(config.get_model_path(_selected_cs_key(config), data_type="tabular"))
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
    budgets = [data.n_concepts if b == -1 else b for b in config.intervention_budgets]
    for budget in budgets:
        interv_cfg = InterventionConfig(
            abstention_threshold=cs_t,
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
        selective_acc_after = result.strategy_metrics.get("selective_acc_after", None)
        coverage_after = result.strategy_metrics.get("coverage_after", None)
        rows.append(
            {
                "budget": budget,
                "accuracy": acc_intervened,
                "predictions_intervened_on": predictions_intervened_on,
                "total_concept_checks": total_concept_checks,
                "total_concept_edits_made": total_concept_edits_made,
                "selective_accuracy_after": selective_acc_after,
                "coverage_after": coverage_after,
            }
        )

    cs_intervention_df = pd.DataFrame(rows)

    results_key = (
        f"{_selected_cs_key(config)}_interventions"
        if _selected_cs_key(config) != "cs"
        else "interventions"
    )
    csv_path = config.get_results_path(results_key, data_type="tabular").with_suffix(
        ".csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cs_intervention_df.to_csv(csv_path, index=False)
    logger.info("Saved intervention results to %s", csv_path)

    return cs_intervention_df


# ── Stage: align ─────────────────────────────────────────────────────


def align(
    config: SudokuBenchmarkConfig,
    cs_model: ConceptBasedModel | None = None,
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
        data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=config.seed)

    if cs_model is None:
        cs_model = load(config.get_model_path("cs", data_type="tabular"))

    from experiments.alignment import test_alignment

    concept_preds_test = cs_model.concept_detector.predict(data.test).astype(np.float32)
    stats = test_alignment(
        concept_preds_test=concept_preds_test,
        alignment_params=config.get_alignment_weights(),
        label_predictor=cs_model.label_predictor,
        test_dataset=data.test,
    )

    logger.info("=== Alignment Results ===")
    logger.info("  Original accuracy: %.4f", stats["original_accuracy"])
    logger.info("  Aligned accuracy:  %.4f", stats["aligned_accuracy"])
    logger.info("  Accuracy change:   %+.4f", stats["accuracy_change"])
    logger.info("  Predictions changed: %s", stats["predictions_changed"])

    save_path = config.get_alignment_results_path()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        _json.dump(stats, f, indent=2)
    logger.info("  Saved to %s", save_path)

    return stats


# ── Stage: collect_results ────────────────────────────────────────────


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
    import json

    if configs is None:
        configs = [SudokuBenchmarkConfig.default()]

    rows: list[dict] = []

    for cfg in configs:
        label = _dataset_label(cfg)
        target = cfg.target_accuracy

        # ── Selective CSV: DNN + CS at the default target_accuracy ────
        sel_csv = cfg.get_results_path("selective", data_type="tabular").with_suffix(
            ".csv"
        )
        if sel_csv.exists():
            sel_df = pd.read_csv(sel_csv)
            # Normalise column name (mc9 uses "tau", mc21 uses "target_accuracy")
            if "tau" in sel_df.columns:
                sel_df = sel_df.rename(columns={"tau": "target_accuracy"})

            for model in ["dnn", "cs"]:
                model_df = sel_df[
                    (sel_df["model"] == model) & (sel_df["target_accuracy"] == target)
                ]
                if model_df.empty:
                    continue
                r = model_df.iloc[0]
                sel_acc = r["selective_acc"]
                rows.append(
                    {
                        "dataset": label,
                        "model": model,
                        "budget": 0 if model == "cs" else "",
                        "target_accuracy": target,
                        "raw_test_acc": round(float(r["raw_test_acc"]), 4),
                        "selective_acc": round(float(sel_acc), 4)
                        if pd.notna(sel_acc)
                        else "",
                        "selective_cov": round(float(r["selective_cov"]), 4),
                        "predictions_intervened_on": "",
                        "avg_concepts_per_sample": "",
                        "predictions_changed": "",
                    }
                )

        # ── Intervention CSV: CS at k > 0 ────────────────────────────
        interv_csv = cfg.get_results_path(
            "interventions", data_type="tabular"
        ).with_suffix(".csv")
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
                rows.append(
                    {
                        "dataset": label,
                        "model": "cs",
                        "budget": budget,
                        "target_accuracy": target,
                        "raw_test_acc": "",
                        "selective_acc": round(float(sel_acc), 4)
                        if pd.notna(sel_acc)
                        else "",
                        "selective_cov": round(float(cov), 4) if pd.notna(cov) else "",
                        "predictions_intervened_on": pio,
                        "avg_concepts_per_sample": avg_cps,
                        "predictions_changed": int(r["total_concept_edits_made"]),
                    }
                )

        # ── Alignment JSON ───────────────────────────────────────────
        align_path = cfg.get_alignment_results_path(data_type="tabular")
        if align_path.exists():
            with open(align_path) as f:
                align_data = json.load(f)
            rows.append(
                {
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
                }
            )

    final_df = pd.DataFrame(rows)
    cfg0 = configs[0]
    out_path = cfg0.get_collect_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(final_df), out_path)
    return final_df


# ── Stage: run (orchestrator) ─────────────────────────────────────────


def run(
    config: SudokuBenchmarkConfig | None = None,
    stages: list[str] | None = None,
    force_setup: bool = False,
) -> None:
    """Run the full sudoku benchmark pipeline.

    Args:
        config: Benchmark configuration. Defaults to default().
        stages: List of stages to run. Default: all.
        force_setup: If True, delete cached data before regenerating.
    """
    from concept_benchmark._logging import setup_logging

    setup_logging()
    patch_macos_dataloader()

    if config is None:
        config = SudokuBenchmarkConfig.default()
    if stages is None:
        stages = [
            "setup",
            "ocr",
            "cs",
            "dnn",
            "intervene",
            "selective",
            "align",
            "collect",
        ]

    if _selected_cs_key(config) != "cs" and config.data_type != "tabular":
        raise ValueError(
            "Sudoku CEM/ProbCBM support currently requires --data-type tabular."
        )

    # Early validation: check that dataset directory exists if we need it
    _needs_data = {"cs", "dnn", "intervene", "selective", "align", "collect"}
    if _needs_data & set(stages) and "setup" not in stages:
        tab_dir = config.get_dataset_path(data_type="tabular")
        ds_path = tab_dir / "sudoku_dataset.pkl"
        if not ds_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {ds_path}\n"
                f"Run with --stages setup ocr first, or include 'setup' and 'ocr' in --stages."
            )

    device = determine_device()
    n_stages = len(stages)
    _si = {s: i for i, s in enumerate(stages, 1)}
    logger.info(
        "=== Sudoku Benchmark === seed=%d, cbm_family=%s, stages=%s, device=%s",
        config.seed,
        _selected_cs_key(config),
        stages,
        device,
    )

    if "setup" in stages:
        logger.info("=== [%d/%d] Setup ===", _si["setup"], n_stages)
        import shutil

        fp_path = config.get_dataset_path(data_type="tabular") / ".fingerprint"
        current_fp = config.setup_fingerprint()
        cached_fp = fp_path.read_text().strip() if fp_path.exists() else None

        if force_setup or cached_fp != current_fp:
            if force_setup:
                logger.info("--force-setup: regenerating data from scratch")
            elif cached_fp is None:
                logger.info(
                    "No cached data found — generating sudoku boards (this may take a few minutes)"
                )
            else:
                logger.info("Config changed since last setup — regenerating data")
            for dt in ("tabular", "image"):
                ds_dir = config.get_dataset_path(data_type=dt)
                if ds_dir.exists():
                    shutil.rmtree(ds_dir)
            setup_dataset(config)
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            fp_path.write_text(current_fp)
        else:
            logger.info("Setup data is up to date (fingerprint matches), skipping")

    # Model fingerprint: retrain if config changed since last training
    model_fp_path = config.get_model_path(_selected_cs_key(config)).with_suffix(
        ".fingerprint"
    )
    current_model_fp = config.model_fingerprint()
    cached_model_fp = (
        model_fp_path.read_text().strip() if model_fp_path.exists() else None
    )
    model_stale = cached_model_fp != current_model_fp

    if "ocr" in stages:
        if config.data_type == "tabular":
            logger.info(
                "=== [%d/%d] Train OCR === (skipped — data_type is tabular)",
                _si["ocr"],
                n_stages,
            )
        else:
            logger.info("=== [%d/%d] Train OCR ===", _si["ocr"], n_stages)
            if model_stale or not config.get_model_path("ocr").exists():
                train_ocr(config)
            else:
                logger.info(
                    "Using existing OCR model: %s", config.get_model_path("ocr")
                )

    if "cs" in stages:
        logger.info("=== [%d/%d] Train CS ===", _si["cs"], n_stages)
        selected_cs_key = _selected_cs_key(config)
        if model_stale or not config.get_model_path(selected_cs_key).exists():
            train_cs(config)
        else:
            logger.info(
                "Using existing %s model: %s",
                selected_cs_key,
                config.get_model_path(selected_cs_key),
            )

    if "dnn" in stages:
        logger.info("=== [%d/%d] Train DNN ===", _si["dnn"], n_stages)
        if model_stale or not config.get_model_path("dnn").exists():
            train_dnn(config)
        else:
            logger.info("Using existing DNN: %s", config.get_model_path("dnn"))

    # Save model fingerprint after training stages
    if any(s in stages for s in ("ocr", "cs", "dnn")) and model_stale:
        model_fp_path.parent.mkdir(parents=True, exist_ok=True)
        model_fp_path.write_text(current_model_fp)

    # Pre-load shared data and models for intervene/selective/align stages
    _eval_stages = {"intervene", "selective", "align"}
    _shared_data = None
    _shared_cs = None
    _shared_dnn = None
    if _eval_stages & set(stages):
        if config.data_type == "image":
            img_dir = config.get_dataset_path(data_type="image")
            _shared_data = load(img_dir / "ocr_inferred_full_dataset.pkl")
        else:
            tab_dir = config.get_dataset_path(data_type="tabular")
            _shared_data = load(tab_dir / "sudoku_dataset.pkl")
        _shared_data.sample(
            test_size=0.2, val_size=0.2, stratify=_shared_data.y, seed=config.seed
        )

        cs_path = config.get_model_path(_selected_cs_key(config), data_type="tabular")
        if cs_path.exists():
            _shared_cs = load(cs_path)
            _shared_cs._random_state = config.seed

        dnn_path = config.get_model_path("dnn", data_type="tabular")
        if dnn_path.exists():
            _shared_dnn = load(dnn_path)

    if "intervene" in stages:
        logger.info("=== [%d/%d] Intervene ===", _si["intervene"], n_stages)
        df = run_interventions(config, cs_model=_shared_cs, data=_shared_data)
        logger.info("=== Intervention Results ===\n%s", df.to_string(index=False))

    if "selective" in stages:
        logger.info("=== [%d/%d] Selective ===", _si["selective"], n_stages)
        if _selected_cs_key(config) != "cs":
            logger.info(
                "Selective evaluation is only implemented for cbm_family='cbm'; skipping %s.",
                _selected_cs_key(config),
            )
        else:
            sel_df = compute_selective_results(
                config, cs_model=_shared_cs, dnn_weights=_shared_dnn, data=_shared_data
            )
            logger.info("=== Selective Metrics ===\n%s", sel_df.to_string(index=False))

    if "align" in stages:
        logger.info("=== [%d/%d] Align ===", _si["align"], n_stages)
        if _selected_cs_key(config) != "cs":
            logger.info(
                "Alignment is only implemented for cbm_family='cbm'; skipping %s.",
                _selected_cs_key(config),
            )
        else:
            align(config, cs_model=_shared_cs, data=_shared_data)

    if "collect" in stages:
        logger.info("=== [%d/%d] Collect ===", _si["collect"], n_stages)
        if _selected_cs_key(config) != "cs":
            logger.info(
                "Collect is only implemented for cbm_family='cbm'; skipping %s.",
                _selected_cs_key(config),
            )
        else:
            collect_results([config])

    if "plot" in stages:
        logger.info("=== [%d/%d] Plot ===", _si.get("plot", n_stages), n_stages)
        if _selected_cs_key(config) != "cs":
            logger.info(
                "Plotting is only implemented for cbm_family='cbm'; skipping %s.",
                _selected_cs_key(config),
            )
        else:
            _plot_results(config)

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
    best_t = (
        0.5 * (min(best_thresholds) + max(best_thresholds)) if best_thresholds else 0.5
    )
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
        target_accuracies = [
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            0.99,
            1.00,
        ]

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
        rows.append(
            {
                "model": "dnn",
                "target_accuracy": tau,
                "raw_test_acc": dnn_raw_acc,
                "selective_acc": sel_acc,
                "selective_cov": sel_cov,
            }
        )

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
        rows.append(
            {
                "model": "cs",
                "target_accuracy": tau,
                "raw_test_acc": cs_raw_acc,
                "selective_acc": sel_acc,
                "selective_cov": sel_cov,
            }
        )

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = config.get_results_path("selective", data_type="tabular").with_suffix(
        ".csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved selective metrics to %s", csv_path)

    return df


# ── CLI entry point ──────────────────────────────────────────────────


def _plot_results(config) -> None:
    """Generate figures from collected results."""
    from concept_benchmark.evaluation.plots import plot_selective_classification
    from concept_benchmark.paths import results_dir

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load collect CSV for selective metrics
    collect_candidates = sorted(
        results_dir.glob(f"sudoku_seed{config.seed}_*_results.csv")
    )
    if collect_candidates:
        import pandas as pd

        df = pd.read_csv(collect_candidates[-1])
        dnn_row = df[df["model"] == "dnn"]
        cs_row = df[(df["model"] == "cs") & (df["budget"] == 0)]
        if len(dnn_row) and len(cs_row):
            dnn_metrics = {
                "selective_acc": float(dnn_row.iloc[0].get("selective_acc", 0)),
                "coverage": float(dnn_row.iloc[0].get("selective_cov", 0)),
            }
            cbm_metrics = {
                "selective_acc": float(cs_row.iloc[0].get("selective_acc", 0)),
                "coverage": float(cs_row.iloc[0].get("selective_cov", 0)),
            }
            fig, _ = plot_selective_classification(dnn_metrics, cbm_metrics)
            fig.savefig(
                out_dir / "sudoku_selective_classification.png",
                dpi=150,
                bbox_inches="tight",
            )
            logger.info("Saved %s", out_dir / "sudoku_selective_classification.png")


SUDOKU_STAGES = (
    "setup",
    "ocr",
    "cs",
    "dnn",
    "intervene",
    "selective",
    "align",
    "collect",
    "plot",
)


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the sudoku validation benchmark pipeline.",
    )
    parser.add_argument("--seed", type=int, default=171)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(SUDOKU_STAGES),
        help=f"Pipeline stages to run (default: all). Valid: {' -> '.join(SUDOKU_STAGES)}",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help="Intervention budgets (e.g. 1 3 5 max).",
    )
    parser.add_argument(
        "--data-type", type=str, default=None, choices=["tabular", "image"]
    )
    parser.add_argument(
        "--cbm-family",
        type=str,
        default=None,
        choices=["cbm", "cem", "probcbm"],
        help="Concept-model family to train/evaluate (tabular support only for cem/probcbm).",
    )
    parser.add_argument("--handwriting", action="store_true", default=None)
    parser.add_argument("--no-handwriting", action="store_true")
    parser.add_argument("--force-setup", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    unknown = set(args.stages) - set(SUDOKU_STAGES)
    if unknown:
        raise ValueError(
            f"unknown stages: {sorted(unknown)}. Valid: {list(SUDOKU_STAGES)}"
        )

    if args.config:
        config = SudokuBenchmarkConfig.from_yaml(args.config)
    else:
        config = SudokuBenchmarkConfig(seed=args.seed)

    if args.budgets:
        config.intervention_budgets = parse_budgets(args.budgets)
    if args.no_handwriting:
        config.font_style = "printed"
    elif args.handwriting:
        config.font_style = "handwritten"
    if args.data_type is not None:
        config.data_type = args.data_type
    if args.cbm_family is not None:
        config.cbm_family = args.cbm_family

    run(config, stages=args.stages, force_setup=args.force_setup)


if __name__ == "__main__":
    main()
