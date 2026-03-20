"""Robot text pipeline — stage orchestration (run_interventions, align, run)."""
from __future__ import annotations

import logging

import pandas as pd

from concept_benchmark.utils import determine_device
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.ext.fileutils import load
from experiments.models import ConceptBasedModel
from experiments.utils import run_alignment

from .training import (
    setup_dataset,
    train_cbm,
    train_dnn,
    train_lfcbm,
)
from .regimes import _ensure_intervention_imports, _run_text_regime
from .collect import collect_results

logger = logging.getLogger(__name__)


# ── Stage: run_interventions ─────────────────────────────────────────


def run_interventions(
    config: RobotBenchmarkConfig,
    model: ConceptBasedModel | None = None,
    data: ConceptDatasetSample | None = None,
) -> pd.DataFrame:
    """Run k-flip interventions on the trained CBM.

    Loops over ``config.intervention_regimes`` (default: ``["baseline"]``).
    """
    _ensure_intervention_imports()
    determine_device()

    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    budgets = [data.n_concepts if b == -1 else b for b in config.intervention_budgets]
    budgets = [b for b in budgets if b > 0]
    threshold = config.concept_uncertainty_threshold

    all_dfs = []
    for regime in config.intervention_regimes:
        try:
            regime_df = _run_text_regime(
                config, regime, model, data, budgets, threshold
            )
            all_dfs.append(regime_df)
        except (FileNotFoundError, NotImplementedError) as e:
            logger.warning("Skipping regime %r: %s", regime, e)

    if not all_dfs:
        logger.warning("No regimes produced results.")
        return pd.DataFrame()

    results_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    results_df["seed"] = config.seed
    results_df["human_accuracy"] = config.intervention_accuracy
    results_df.to_csv(config.get_results_path("cbm"), index=False)
    logger.info("Saved intervention results to %s", config.get_results_path("cbm"))
    return results_df


# ── Stage: align ─────────────────────────────────────────────────────


def align(
    config: RobotBenchmarkConfig,
    model: ConceptBasedModel | None = None,
    data: ConceptDatasetSample | None = None,
) -> dict:
    """Run alignment test on the trained CBM.

    Retrains the frontend with monotonicity constraints and compares
    original vs constrained accuracy.
    """
    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    return run_alignment(
        concept_based_model=model,
        train_dataset=data.training,
        test_dataset=data.test,
        monotonicity_constraints=config.get_alignment_constraints(),
        save_path=config.get_alignment_results_path(),
    )


# ── Stage: run (orchestrator) ────────────────────────────────────────


def run(
    config: RobotBenchmarkConfig | None = None,
    stages: list[str] | None = None,
    force_setup: bool = False,
) -> None:
    """Run the full robot text benchmark pipeline.

    Args:
        config: Benchmark configuration.
        stages: List of stages to run. Default: all.
        force_setup: If True, delete cached data before regenerating.
    """
    from concept_benchmark._logging import setup_logging

    setup_logging()
    if config is None:
        config = RobotBenchmarkConfig(data_type="text")
    if stages is None:
        stages = ["setup", "cbm", "dnn", "intervene", "align", "collect"]

    # Early validation: check that dataset exists if we need it
    _needs_data = {"cbm", "dnn", "intervene", "align", "collect"}
    if _needs_data & set(stages) and "setup" not in stages:
        ds_path = config.get_dataset_path()
        if not ds_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {ds_path}\n"
                f"Run with --stages setup first, or include 'setup' in --stages."
            )

    device = determine_device()
    n_stages = len(stages)
    _si = {s: i for i, s in enumerate(stages, 1)}
    logger.info(
        "=== Robot Text Benchmark === seed=%d, stages=%s, device=%s",
        config.seed,
        stages,
        device,
    )

    if "setup" in stages:
        logger.info("=== [%d/%d] Setup ===", _si["setup"], n_stages)
        fp_path = config.get_dataset_path().with_suffix(".fingerprint")
        current_fp = config.setup_fingerprint()
        cached_fp = fp_path.read_text().strip() if fp_path.exists() else None

        if force_setup or cached_fp != current_fp:
            if force_setup:
                logger.info("--force-setup: regenerating data from scratch")
            elif cached_fp is None:
                logger.info("No cached data found — generating text dataset")
            else:
                logger.info("Config changed since last setup — regenerating data")
            ds_path = config.get_dataset_path()
            if ds_path.exists():
                ds_path.unlink()
            setup_dataset(config)
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            fp_path.write_text(current_fp)
        else:
            logger.info("Setup data is up to date (fingerprint matches), skipping")

    # Model fingerprint: retrain if config changed since last training
    model_fp_path = config.get_model_path("cbm").with_suffix(".fingerprint")
    current_model_fp = config.model_fingerprint()
    cached_model_fp = (
        model_fp_path.read_text().strip() if model_fp_path.exists() else None
    )
    model_stale = cached_model_fp != current_model_fp

    if "cbm" in stages:
        logger.info("=== [%d/%d] Train CBM ===", _si["cbm"], n_stages)
        if (
            model_stale
            or config.force_retrain
            or not config.get_model_path("cbm").exists()
        ):
            train_cbm(config)
        else:
            logger.info("Using existing CBM: %s", config.get_model_path("cbm"))

    if "dnn" in stages:
        logger.info("=== [%d/%d] Train DNN ===", _si["dnn"], n_stages)
        if (
            model_stale
            or config.force_retrain
            or not config.get_model_path("dnn").exists()
        ):
            train_dnn(config)
        else:
            logger.info("Using existing DNN: %s", config.get_model_path("dnn"))

    if "lfcbm" in stages and config.use_label_free_concepts:
        logger.info("=== [%d/%d] Train LFCBM ===", _si["lfcbm"], n_stages)
        if (
            model_stale
            or config.force_retrain
            or not config.get_model_path("lfcbm").exists()
        ):
            train_lfcbm(config)
        else:
            logger.info("Using existing LFCBM: %s", config.get_model_path("lfcbm"))

    # Save model fingerprint after training stages
    if any(s in stages for s in ("cbm", "dnn", "lfcbm")) and model_stale:
        model_fp_path.parent.mkdir(parents=True, exist_ok=True)
        model_fp_path.write_text(current_model_fp)

    if "intervene" in stages:
        logger.info("=== [%d/%d] Intervene ===", _si["intervene"], n_stages)
        run_interventions(config)

    if "align" in stages:
        logger.info("=== [%d/%d] Align ===", _si["align"], n_stages)
        align(config)

    if "collect" in stages:
        logger.info("=== [%d/%d] Collect ===", _si["collect"], n_stages)
        collect_results([config])

    logger.info("Robot text pipeline complete!")
