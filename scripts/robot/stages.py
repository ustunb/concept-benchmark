"""Robot pipeline — stage orchestration (run_interventions, align, run)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from concept_benchmark.utils import (
    determine_device,
    patch_macos_dataloader,
    set_deterministic_seed,
)
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load
from experiments.utils import run_alignment

from .training import (
    setup_dataset,
    train_cbm,
    train_cbm_subjective,
    train_dnn,
    train_lfcbm,
)
from .regimes import _ensure_intervention_imports, _run_regime
from .collect import collect_results

logger = logging.getLogger(__name__)


# ── Stage: run_interventions ──────────────────────────────────────────

def run_interventions(
    config: RobotBenchmarkConfig,
    model=None,
    data=None,
    missing_fraction: float = 0.0,
    missing_mechanism: str = "none",
) -> pd.DataFrame:
    """Run interventions on the trained CBM and return a results DataFrame.

    Loops over ``config.intervention_regimes`` (default: ``["baseline"]``).
    """
    set_deterministic_seed(config.seed)
    _ensure_intervention_imports()
    patch_macos_dataloader()
    determine_device()

    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    budgets = sorted(set(
        [0] + [data.n_concepts if b == -1 else b for b in config.intervention_budgets]
    ))
    thresholds = config.intervention_thresholds

    all_dfs = []
    for regime in config.intervention_regimes:
        try:
            regime_df = _run_regime(config, regime, model, data, budgets, thresholds)
            all_dfs.append(regime_df)
        except (FileNotFoundError, NotImplementedError) as e:
            logger.warning("Skipping regime %r: %s", regime, e)

    if not all_dfs:
        logger.warning("No regimes produced results.")
        return pd.DataFrame()

    results_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    results_df["data_name"] = "subconcept" if config.concept_preset == "foot_subtypes" else "ideal"
    results_df["n"] = data.test.n
    results_df["missing_fraction"] = missing_fraction
    results_df["missing_mechanism"] = missing_mechanism
    results_df.to_csv(config.get_results_path("cbm"), index=False)
    return results_df


# ── Stage: align ─────────────────────────────────────────────────────

def align(
    config: RobotBenchmarkConfig,
    model=None,
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
        concept_based_model=model,
        train_dataset=data.training,
        test_dataset=data.test,
        monotonicity_constraints=config.get_alignment_constraints(),
        save_path=config.get_alignment_results_path(),
    )


# ── Stage: run (orchestrator) ─────────────────────────────────────────

def run(
    config: RobotBenchmarkConfig | None = None,
    stages: list[str] | None = None,
    force_setup: bool = False,
    missing_fraction: float = 0.0,
    missing_mechanism: str = "mcar",
) -> None:
    """Run the robot benchmark pipeline for a single configuration.

    Args:
        config: Benchmark configuration. Defaults to ideal.
        stages: List of stages to run. Default: all.
        force_setup: If True, delete cached images/data before regenerating.
        missing_fraction: Fraction of concept labels to mask.
        missing_mechanism: Missingness mechanism ("mcar" or "mnar").
    """
    from concept_benchmark._logging import setup_logging
    setup_logging()
    patch_macos_dataloader()

    if config is None:
        config = RobotBenchmarkConfig.default_ideal()
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
    variant = "subconcept" if config.concept_preset == "foot_subtypes" else "ideal"
    n_stages = len(stages)
    _si = {s: i for i, s in enumerate(stages, 1)}
    logger.info(
        "=== Robot Benchmark === seed=%d, variant=%s, stages=%s, device=%s",
        config.seed, variant, stages, device,
    )

    if "setup" in stages:
        logger.info("=== [%d/%d] Setup ===", _si["setup"], n_stages)
        import shutil
        fp_path = config.get_dataset_path().with_suffix(".fingerprint")
        current_fp = config.setup_fingerprint()
        cached_fp = fp_path.read_text().strip() if fp_path.exists() else None

        if force_setup or cached_fp != current_fp:
            if force_setup:
                logger.info("--force-setup: regenerating data from scratch")
            elif cached_fp is None:
                logger.info("No cached data found — generating dataset and robot images (this may take a minute)")
            else:
                logger.info("Config changed since last setup — regenerating data")
            # Clear cached images and dataset
            img_dir = config.to_dict()["output_directory"]
            if Path(img_dir).exists():
                shutil.rmtree(img_dir)
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
    cached_model_fp = model_fp_path.read_text().strip() if model_fp_path.exists() else None
    model_stale = cached_model_fp != current_model_fp

    def _should_train(model_key: str) -> bool:
        model_path = config.get_model_path(model_key)
        if config.force_retrain:
            return True
        if not model_path.exists():
            return True
        if model_stale:
            logger.info("Config changed since last training — retraining %s", model_key)
            return True
        return False

    if "cbm" in stages:
        logger.info("=== [%d/%d] Train CBM ===", _si["cbm"], n_stages)
        if _should_train("cbm"):
            train_cbm(config, missing_fraction=missing_fraction, missing_mechanism=missing_mechanism)
        else:
            logger.info("Using existing CBM: %s", config.get_model_path("cbm"))
        if "subjective" in config.intervention_regimes:
            if _should_train("cbm_subjective"):
                train_cbm_subjective(config)
            else:
                logger.info("Using existing subjective CBM: %s", config.get_model_path("cbm_subjective"))
        if "machine" in config.intervention_regimes:
            if _should_train("lfcbm"):
                train_lfcbm(config)
            else:
                logger.info("Using existing LFCBM: %s", config.get_model_path("lfcbm"))

    if "dnn" in stages:
        logger.info("=== [%d/%d] Train DNN ===", _si["dnn"], n_stages)
        if _should_train("dnn"):
            train_dnn(config)
        else:
            logger.info("Using existing DNN: %s", config.get_model_path("dnn"))

    # Save model fingerprint after training stages
    if ("cbm" in stages or "dnn" in stages) and model_stale:
        model_fp_path.parent.mkdir(parents=True, exist_ok=True)
        model_fp_path.write_text(current_model_fp)

    if "intervene" in stages:
        logger.info("=== [%d/%d] Intervene ===", _si["intervene"], n_stages)
        run_interventions(config, missing_fraction=missing_fraction,
                          missing_mechanism=missing_mechanism)

    if "align" in stages:
        logger.info("=== [%d/%d] Align ===", _si["align"], n_stages)
        align(config)

    if "collect" in stages:
        logger.info("=== [%d/%d] Collect ===", _si["collect"], n_stages)
        collect_results([config])
