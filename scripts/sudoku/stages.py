"""Sudoku pipeline — intervention, alignment, and run orchestrator."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from concept_benchmark.utils import (
    determine_device,
    patch_macos_dataloader,
)
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.ext.fileutils import load
from experiments.models import ConceptBasedModel
from experiments.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionConfig,
)

from scripts.sudoku.training import (
    setup_dataset,
    train_cs,
    train_dnn,
    train_ocr,
)
from scripts.sudoku.selective import (
    _cs_val_probs,
    _decision_threshold_sweep,
    _selective_accuracy_threshold,
    _selective_metrics,
    compute_selective_results,
)
from scripts.sudoku.collect import collect_results

logger = logging.getLogger(__name__)


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

    # Threshold at 0.5 to match cbm.predict() binarisation
    concept_preds_test = (cs_model.concept_detector.predict(data.test) > 0.5).astype(np.float32)
    stats = test_alignment(
        concept_preds_test=concept_preds_test,
        alignment_params=config.get_alignment_weights(),
        label_predictor=cs_model.label_predictor,
        test_dataset=data.test,
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
        stages = ["setup", "ocr", "cs", "dnn", "intervene", "selective", "align", "collect"]

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

    # If image mode, evaluation stages need OCR-inferred data
    _needs_ocr = {"intervene", "selective", "align"}
    if (
        config.data_type == "image"
        and _needs_ocr & set(stages)
        and "ocr" not in stages
    ):
        img_dir = config.get_dataset_path(data_type="image")
        ocr_path = img_dir / "ocr_inferred_full_dataset.pkl"
        if not ocr_path.exists():
            raise FileNotFoundError(
                f"OCR-inferred dataset not found: {ocr_path}\n"
                f"Include 'ocr' in --stages, or use --data-type tabular."
            )

    device = determine_device()
    n_stages = len(stages)
    _si = {s: i for i, s in enumerate(stages, 1)}
    logger.info(
        "=== Sudoku Benchmark === seed=%d, stages=%s, device=%s",
        config.seed, stages, device,
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
                logger.info("No cached data found — generating sudoku boards (this may take a few minutes)")
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
    model_fp_path = config.get_model_path("cs").with_suffix(".fingerprint")
    current_model_fp = config.model_fingerprint()
    cached_model_fp = model_fp_path.read_text().strip() if model_fp_path.exists() else None
    model_stale = cached_model_fp != current_model_fp

    if "ocr" in stages:
        if config.data_type == "tabular":
            logger.info("=== [%d/%d] Train OCR === (skipped — data_type is tabular)", _si["ocr"], n_stages)
        else:
            logger.info("=== [%d/%d] Train OCR ===", _si["ocr"], n_stages)
            if model_stale or not config.get_model_path("ocr").exists():
                train_ocr(config)
            else:
                logger.info("Using existing OCR model: %s", config.get_model_path("ocr"))

    if "cs" in stages:
        logger.info("=== [%d/%d] Train CS ===", _si["cs"], n_stages)
        if model_stale or not config.get_model_path("cs").exists():
            train_cs(config)
        else:
            logger.info("Using existing CS model: %s", config.get_model_path("cs"))

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

        cs_path = config.get_model_path("cs", data_type="tabular")
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
        sel_df = compute_selective_results(
            config, cs_model=_shared_cs, dnn_weights=_shared_dnn, data=_shared_data
        )
        logger.info("=== Selective Metrics ===\n%s", sel_df.to_string(index=False))

    if "align" in stages:
        logger.info("=== [%d/%d] Align ===", _si["align"], n_stages)
        align(config, cs_model=_shared_cs, data=_shared_data)

    if "collect" in stages:
        logger.info("=== [%d/%d] Collect ===", _si["collect"], n_stages)
        collect_results([config])

    logger.info("Pipeline complete!")
