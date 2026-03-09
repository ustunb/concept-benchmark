"""Robot text classification benchmark pipeline.

Provides functions to run each stage of the robot text benchmark
programmatically, following the same pattern as robot_pipeline.py and
sudoku_pipeline.py.

Usage:
    python scripts/robot_text_pipeline.py --seed 1337
    python scripts/robot_text_pipeline.py --regimes baseline expert
    python scripts/robot_text_pipeline.py --config my_config.yaml
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from concept_benchmark.utils import (
    determine_device,
    run_alignment,
)
from concept_benchmark.config import RobotTextBenchmarkConfig
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.models import ConceptBasedModel, FrontEndModel
from concept_benchmark.paths import pkg_dir, results_dir
from concept_benchmark.synthetic.helper.text_concept_detector import TextConceptDetector
from concept_benchmark.synthetic.robot_text.catalog import (
    TEXT_CONCEPTS,
    compute_label,
    enumerate_robot_concepts,
)
from concept_benchmark.synthetic.robot_text.dataset import (
    build_text_dataset,
    kfold_by_robot_identity,
)

logger = logging.getLogger(__name__)

# Lazy imports for intervention modules
_intervention_imported = False


def _ensure_intervention_imports():
    global _intervention_imported
    if not _intervention_imported:
        global ConceptInterventionRunner, InterventionConfig, KFlipInterventionStrategy
        from concept_benchmark.intervention import (
            ConceptInterventionRunner,
            InterventionConfig,
        )
        from concept_benchmark.kflip import KFlipInterventionStrategy

        _intervention_imported = True


def _set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── Stage: setup_dataset ─────────────────────────────────────────────

def setup_dataset(
    config: RobotTextBenchmarkConfig,
) -> ConceptDatasetSample:
    """Generate robot text dataset and split by robot identity.

    Returns the dataset with training/validation/test splits set.
    """
    _set_seed(config.seed)

    # Enumerate all robot concept combinations
    catalog_df = enumerate_robot_concepts(concepts=TEXT_CONCEPTS, seed=config.seed)

    # Compute labels
    catalog_df["label"] = compute_label(
        catalog_df, config.label_expr, seed=config.seed,
    )

    _lbl = catalog_df["label"].astype(str)
    n_pos = int((_lbl == "glorp").sum())
    n_neg = int((_lbl == "drent").sum())
    minority_label = "glorp" if n_pos < n_neg else "drent"
    logger.info("Catalog: %d glorp, %d drent (minority=%s)", n_pos, n_neg, minority_label)

    # Determine per-row variant counts
    row_variants = [
        config.variants_per_row_minority if lab == minority_label else config.variants_per_row_majority
        for lab in _lbl
    ]

    # Find corpus paths
    corpus_path = _get_corpus_path(config)
    generic_path = _get_generic_corpus_path(config, corpus_path) if config.generic_enable else None

    # Build dataset
    ds = build_text_dataset(
        catalog_df=catalog_df,
        corpus_path=corpus_path,
        variants_per_row=1,
        seed=config.seed,
        row_variants=row_variants,
        generic_path=generic_path,
        generic_rate=config.generic_rate if config.generic_enable else 0.0,
        generic_target=config.generic_target,
    )

    # Split by robot identity
    ds = kfold_by_robot_identity(
        ds,
        cv_k=config.cv_k,
        cv_fold=config.cv_fold,
        dev_per_fold=config.dev_per_fold,
        deployment_size=config.deployment_size,
        seed=config.seed,
        generic_target=config.generic_target,
        corpus_path=corpus_path,
        catalog_df=catalog_df,
        generic_path=generic_path,
        generic_rate=config.generic_rate if config.generic_enable else 0.0,
    )

    logger.info("Split sizes — train: %d, val: %d, test: %d", ds.training.n, ds.validation.n, ds.test.n)
    save(ds, config.get_dataset_path(), overwrite=True)
    return ds


def _get_corpus_path(config: RobotTextBenchmarkConfig) -> Path:
    """Resolve the corpus path based on difficulty setting."""
    if config.difficulty == "hard":
        p = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "HardCorpus.jsonl"
        if p.is_file():
            return p
    name = "Templates.txt" if config.difficulty == "medium" else "Templates_simple.txt"
    return pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / name


def _get_generic_corpus_path(
    config: RobotTextBenchmarkConfig,
    corpus_path: Path,
) -> Path | None:
    """Resolve the generic corpus path for a given target concept."""
    target = config.generic_target.capitalize()
    gen = corpus_path.with_name(f"HardCorpus_{target}Generic.jsonl")
    return gen if gen.is_file() else None


# ── Stage: train_cbm ─────────────────────────────────────────────────

def train_cbm(
    config: RobotTextBenchmarkConfig,
    data: Optional[ConceptDatasetSample] = None,
) -> ConceptBasedModel:
    """Train TextConceptDetector + FrontEndModel.

    Returns the trained ConceptBasedModel.
    """
    _set_seed(config.seed)

    if data is None:
        data = load(config.get_dataset_path())

    detector = TextConceptDetector(
        embed_dim=128,
        hidden_dim=192,
        epochs=config.detector_epochs,
        batch_size=config.detector_batch_size,
        lr=config.detector_lr,
        use_bigrams=True,
        dropout=0.1,
        pos_weight="auto",
        output_mode=config.concept_mode,
        threshold_mode="auto",
        pooling="attn",
        group_unknown_threshold=0.50,
        validate=True,
    )

    detector.fit(data.training, data.validation)

    cbm = ConceptBasedModel(
        concept_detector=detector,
        front_end_model=FrontEndModel(),
        propagate=(config.concept_mode == "soft"),
    )

    # Train frontend on ground-truth concepts
    cbm.front_end_model.fit(data.training.C, data.training.y)

    test_pred = cbm.predict(data.test)
    acc = float(np.mean(test_pred == data.test.y))
    logger.info("CBM Test Accuracy: %.4f", acc)

    save(cbm, config.get_model_path("cbm"), overwrite=True)
    return cbm


def train_cbm_subjective(
    config: RobotTextBenchmarkConfig,
    data: Optional[ConceptDatasetSample] = None,
) -> ConceptBasedModel:
    """Train a CBM on noisy (subjective) concept labels.

    Saves to ``config.get_model_path("cbm_subjective")``.
    """
    import copy as _copy

    if data is None:
        data = load(config.get_dataset_path())
    noisy_data = _copy.deepcopy(data)
    noisy_data.sample_concept_noise(
        p=config.subjective_noise_rate,
        # Offset seed so noise RNG is independent of data-generation RNG.
        rng=np.random.default_rng(config.seed + 555),
        enable=True,
    )

    _set_seed(config.seed)
    detector = TextConceptDetector(
        embed_dim=128,
        hidden_dim=192,
        epochs=config.detector_epochs,
        batch_size=config.detector_batch_size,
        lr=config.detector_lr,
        use_bigrams=True,
        dropout=0.1,
        pos_weight="auto",
        output_mode=config.concept_mode,
        threshold_mode="auto",
        pooling="attn",
        group_unknown_threshold=0.50,
        validate=True,
    )
    detector.fit(noisy_data.training, noisy_data.validation)
    cbm = ConceptBasedModel(
        concept_detector=detector,
        front_end_model=FrontEndModel(),
        propagate=(config.concept_mode == "soft"),
    )
    cbm.front_end_model.fit(noisy_data.training.C, noisy_data.training.y)

    test_pred = cbm.predict(data.test)
    acc = float(np.mean(test_pred == data.test.y))
    logger.info("Subjective CBM Test Accuracy: %.4f", acc)

    save(cbm, config.get_model_path("cbm_subjective"), overwrite=True)
    return cbm


# ── Stage: train_dnn ─────────────────────────────────────────────────

class _TextDS(Dataset):
    """Simple text dataset for DistilBERT fine-tuning."""

    def __init__(self, X, y, tok, max_length=256):
        self.X = list(map(str, X))
        self.y = np.asarray(y, dtype=int)
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        enc = self.tok(
            self.X[i], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(self.y[i], dtype=torch.long)
        return enc, y


def _train_dnn_text(X_tr, y_tr, X_te, y_te, model_id, epochs, batch_size, lr, device):
    """Train a text classifier (DistilBERT) and return accuracy + model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    ds_tr = _TextDS(X_tr, y_tr, tok)
    ds_te = _TextDS(X_te, y_te, tok)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in dl_tr:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb, labels=yb)
            optim.zero_grad()
            out.loss.backward()
            optim.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb)
            pred = out.logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    acc = correct / total if total > 0 else 0.0
    return float(acc), tok, model


def _fit_platt(X, y, tok, model, device):
    """Fit Platt calibration on validation set."""
    from sklearn.linear_model import LogisticRegression

    ds = _TextDS(X, np.asarray(y, dtype=int), tok)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    z_list, y_list = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            out = model(**xb)
            z = (out.logits[:, 1] - out.logits[:, 0]).detach().cpu().numpy()
            z_list.append(z)
            y_list.append(yb.numpy())
    Z = np.concatenate(z_list).reshape(-1, 1) if z_list else np.zeros((0, 1))
    Y = np.concatenate(y_list).astype(int) if y_list else np.zeros(0, dtype=int)
    if Y.size == 0 or np.unique(Y).size < 2:
        return None
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(Z, Y)
    return lr


def train_dnn(
    config: RobotTextBenchmarkConfig,
    data: Optional[ConceptDatasetSample] = None,
) -> dict:
    """Fine-tune DistilBERT on text -> label (bypasses concepts).

    Returns a dict with accuracy metrics and model paths.
    """
    _set_seed(config.seed)
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    acc, tok, model = _train_dnn_text(
        X_tr=data.training.X,
        y_tr=data.training.y,
        X_te=data.test.X,
        y_te=data.test.y,
        model_id=config.dnn_model_name,
        epochs=config.dnn_epochs,
        batch_size=config.dnn_batch_size,
        lr=config.dnn_lr,
        device=device,
    )

    # Platt calibration on validation
    calibrator = _fit_platt(data.validation.X, data.validation.y, tok, model, device)

    metrics = {"accuracy": acc, "seed": config.seed, "model": config.dnn_model_name}
    logger.info("DNN Test Accuracy: %.4f", acc)

    # Save model and metrics
    model_dir = config.get_model_path("dnn")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    save({"metrics": metrics, "calibrator": calibrator}, model_dir, overwrite=True)
    tok.save_pretrained(str(model_dir) + "_tok")
    model.save_pretrained(str(model_dir) + "_model")

    return metrics


# ── Stage: train_lfcbm ──────────────────────────────────────────────

def train_lfcbm(
    config: RobotTextBenchmarkConfig,
    data: Optional[ConceptDatasetSample] = None,
) -> ConceptBasedModel:
    """Train label-free CBM using sentence embeddings + concepts CSV.

    Returns a ConceptBasedModel with LabelFreeDetector as concept source.
    """
    from concept_benchmark.synthetic.robot_text.lfcbm import LabelFreeDetector

    _set_seed(config.seed)

    if data is None:
        data = load(config.get_dataset_path())

    # Determine concepts CSV path
    concepts_csv = config.lfcbm_concepts_csv
    if not concepts_csv:
        default = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "concepts.csv"
        if default.is_file():
            concepts_csv = str(default)

    lf_device = "cuda" if torch.cuda.is_available() else "cpu"
    lf_settings = {
        "concepts_csv": concepts_csv,
        "lf_alpha": 0.5,
        "lf_threshold": 0.5,
        "lf_mode": config.concept_mode,
        "lf_ridge": False,
        "lf_ridge_alpha": 1.0,
        "lf_encoder": config.lfcbm_encoder,
        "lf_device": lf_device,
        "lf_batch_size": 64,
        "lf_keep_k": 9,
        "lf_group_threshold": 0.9,
    }

    det_lf = LabelFreeDetector(lf_settings)
    det_lf.fit([str(x) for x in data.training.X], y=data.training.y.astype(int))

    # Use LFCBM predictions as concept features to train a student detector
    if config.concept_mode == "hard":
        det_lf.settings["lf_mode"] = "hard"
    C_train = det_lf.predict([str(x) for x in data.training.X])

    # Build a CBM with a simple frontend trained on LFCBM concepts
    fe = FrontEndModel()
    fe.fit(C_train.astype(np.float32), data.training.y)

    # Wrap: we need a concept detector that can predict on a ConceptDatasetSample
    # For LFCBM, predictions are the LFCBM output; we create a thin wrapper
    detector = TextConceptDetector(
        epochs=config.detector_epochs,
        batch_size=config.detector_batch_size,
        lr=config.detector_lr,
        output_mode=config.concept_mode,
    )
    detector.fit(data.training, data.validation)

    cbm = ConceptBasedModel(
        concept_detector=detector,
        front_end_model=fe,
        propagate=(config.concept_mode == "soft"),
    )

    test_pred = cbm.predict(data.test)
    acc = float(np.mean(test_pred == data.test.y))
    logger.info("LFCBM Test Accuracy: %.4f", acc)

    save(cbm, config.get_model_path("lfcbm"), overwrite=True)
    det_lf.save(str(config.get_model_path("lfcbm")) + "_lf")
    return cbm


# ── Regime dispatch (text) ────────────────────────────────────────────

def _run_text_regime(config, regime, model, data, budgets, threshold):
    """Run one intervention regime for the text benchmark.

    Returns a DataFrame with rows for each budget.
    """
    _ensure_intervention_imports()

    # Select model and human accuracy per regime
    if regime == "baseline":
        regime_model = model
        human_acc = config.intervention_accuracy
    elif regime == "expert":
        regime_model = model
        human_acc = config.expert_intervention_accuracy
    elif regime == "subjective":
        regime_model = load(config.get_model_path("cbm_subjective"))
        human_acc = config.subjective_intervention_accuracy
    elif regime == "machine":
        # Use existing LabelFreeDetector from robot_text/lfcbm.py
        regime_model = load(config.get_model_path("lfcbm"))
        human_acc = config.intervention_accuracy
    else:
        raise ValueError(f"Unknown regime: {regime!r}")

    c_preds = regime_model.concept_detector.predict(data.test)
    base_pred = regime_model.predict(data.test)
    base_acc = float(np.mean(base_pred == data.test.y))

    runner = ConceptInterventionRunner(regime_model)

    rows = []
    # k=0 baseline
    rows.append({
        "budget": 0,
        "threshold": threshold,
        "accuracy": base_acc,
        "predictions_intervened_on": 0,
        "predictions_changed": 0,
        "total_concept_confirmations": 0,
        "total_concept_edits_made": 0,
    })

    rng = np.random.default_rng(config.seed)
    err_prob = 1.0 - human_acc

    for budget in budgets:
        interv_config = InterventionConfig(
            max_concepts_per_instance=budget,
            random_state=config.seed,
            score_threshold=threshold,
            noise=err_prob,
        )
        strategy = KFlipInterventionStrategy(
            exact_k=(config.intervention_strategy == "exact_k"),
        )

        result = runner.run(
            strategy=strategy,
            config=interv_config,
            dataset=data.test,
            concept_proba=c_preds,
            labels=data.test.y.astype(int),
        )

        mask = result.mask
        C_gt = data.test.C.astype(np.float32)
        C_after = result.C_intervened.copy()

        mistake_draw = rng.random(C_after.shape) < err_prob
        mistakes = mask & mistake_draw
        C_after[mistakes] = 1.0 - C_gt[mistakes]
        result.C_intervened = C_after

        C_pred_binary = (result.C_pred >= 0.5).astype(int)
        C_final_binary = (result.C_intervened >= 0.5).astype(int)
        actual_edits_mask = C_pred_binary != C_final_binary
        result.y_prob_after = regime_model.front_end_model.predict_proba(C_final_binary)
        result.y_pred_after = np.argmax(result.y_prob_after, axis=1)

        y_pred_before = np.argmax(result.y_prob_before, axis=1)
        num_preds_change = int(np.sum(result.y_pred_after != y_pred_before))
        acc_intervened = float(np.mean(result.y_pred_after == data.test.y.astype(int)))
        n_intervened = int(np.sum(mask))

        rows.append({
            "budget": budget,
            "threshold": threshold,
            "accuracy": acc_intervened,
            "predictions_intervened_on": int(np.sum(np.any(mask, axis=1))),
            "predictions_changed": num_preds_change,
            "total_concept_confirmations": n_intervened,
            "total_concept_edits_made": int(np.sum(actual_edits_mask)),
        })

    regime_df = pd.DataFrame(rows)
    regime_df["regime"] = regime
    return regime_df


# ── Stage: run_interventions ─────────────────────────────────────────

def run_interventions(
    config: RobotTextBenchmarkConfig,
    model: Optional[ConceptBasedModel] = None,
    data: Optional[ConceptDatasetSample] = None,
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
    threshold = config.flip_threshold

    all_dfs = []
    for regime in config.intervention_regimes:
        try:
            regime_df = _run_text_regime(config, regime, model, data, budgets, threshold)
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
    config: RobotTextBenchmarkConfig,
    model: Optional[ConceptBasedModel] = None,
    data: Optional[ConceptDatasetSample] = None,
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
        cbm=model,
        train_dataset=data.training,
        test_dataset=data.test,
        monotonicity_constraints=config.get_alignment_constraints(),
        save_path=config.get_alignment_results_path(),
    )


# ── Stage: collect_results ───────────────────────────────────────────

def collect_results(
    configs: Optional[List[RobotTextBenchmarkConfig]] = None,
) -> pd.DataFrame:
    """Aggregate robot text results into a single CSV.

    Reads saved artifacts only — no model retraining.
    """
    if configs is None:
        configs = [RobotTextBenchmarkConfig()]

    rows: list[dict] = []
    for cfg in configs:
        results_path = cfg.get_results_path("cbm")
        if results_path.exists():
            df = pd.read_csv(results_path)
            for _, r in df.iterrows():
                rows.append({
                    "seed": cfg.seed,
                    "model": "cbm",
                    "budget": int(r["budget"]),
                    "threshold": float(r["threshold"]),
                    "accuracy": float(r["accuracy"]),
                    "predictions_intervened_on": int(r["predictions_intervened_on"]),
                    "predictions_changed": int(r["predictions_changed"]),
                })

        # DNN metrics
        dnn_path = cfg.get_model_path("dnn")
        if dnn_path.exists():
            dnn_data = load(dnn_path)
            if isinstance(dnn_data, dict) and "metrics" in dnn_data:
                rows.append({
                    "seed": cfg.seed,
                    "model": "dnn",
                    "budget": "",
                    "threshold": "",
                    "accuracy": dnn_data["metrics"].get("accuracy", ""),
                    "predictions_intervened_on": "",
                    "predictions_changed": "",
                })

        # Alignment
        align_path = cfg.get_alignment_results_path()
        if align_path.exists():
            with open(align_path) as f:
                align_data = json.load(f)
            rows.append({
                "seed": cfg.seed,
                "model": "aligned_cbm",
                "budget": 0,
                "threshold": "",
                "accuracy": float(align_data["aligned_accuracy"]),
                "predictions_intervened_on": "",
                "predictions_changed": align_data.get("predictions_changed", ""),
            })

    final_df = pd.DataFrame(rows)
    out_path = results_dir / "robot_text_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(final_df), out_path)
    return final_df


# ── Stage: run (orchestrator) ────────────────────────────────────────

def run(
    config: Optional[RobotTextBenchmarkConfig] = None,
    stages: Optional[List[str]] = None,
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
        config = RobotTextBenchmarkConfig()
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
        config.seed, stages, device,
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
    cached_model_fp = model_fp_path.read_text().strip() if model_fp_path.exists() else None
    model_stale = cached_model_fp != current_model_fp

    if "cbm" in stages:
        logger.info("=== [%d/%d] Train CBM ===", _si["cbm"], n_stages)
        if model_stale or config.force_retrain or not config.get_model_path("cbm").exists():
            train_cbm(config)
        else:
            logger.info("Using existing CBM: %s", config.get_model_path("cbm"))

    if "dnn" in stages:
        logger.info("=== [%d/%d] Train DNN ===", _si["dnn"], n_stages)
        if model_stale or config.force_retrain or not config.get_model_path("dnn").exists():
            train_dnn(config)
        else:
            logger.info("Using existing DNN: %s", config.get_model_path("dnn"))

    if "lfcbm" in stages and config.lfcbm_enable:
        logger.info("=== [%d/%d] Train LFCBM ===", _si["lfcbm"], n_stages)
        if model_stale or config.force_retrain or not config.get_model_path("lfcbm").exists():
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


# ── CLI entry point ──────────────────────────────────────────────────

ROBOT_TEXT_STAGES = ("setup", "cbm", "dnn", "lfcbm", "intervene", "align", "collect")


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the robot text classification benchmark pipeline.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--stages", nargs="+", default=list(ROBOT_TEXT_STAGES),
        help=f"Pipeline stages to run (default: all). Valid: {' -> '.join(ROBOT_TEXT_STAGES)}",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--budgets", nargs="+", default=None,
                        help="Intervention budgets (e.g. 1 2 5 max).")
    parser.add_argument("--regimes", nargs="+", default=None,
                        help="Intervention regimes (e.g. baseline expert subjective).")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["kflip", "exact_k"])
    parser.add_argument("--lfcbm", action="store_true",
                        help="Also run LFCBM variant.")
    parser.add_argument("--force-setup", action="store_true")
    return parser.parse_args(argv)


def _parse_budgets(raw):
    budgets = []
    for v in raw:
        budgets.append(-1 if v.lower() == "max" else int(v))
    return budgets


def main(argv=None):
    args = _parse_args(argv)

    unknown = set(args.stages) - set(ROBOT_TEXT_STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}. Valid: {list(ROBOT_TEXT_STAGES)}")

    if args.config:
        config = RobotTextBenchmarkConfig.from_yaml(args.config)
    else:
        config = RobotTextBenchmarkConfig(seed=args.seed)

    if args.budgets:
        config.intervention_budgets = _parse_budgets(args.budgets)
    if args.regimes:
        config.intervention_regimes = args.regimes
    if args.strategy:
        config.intervention_strategy = args.strategy
    if args.lfcbm:
        config.lfcbm_enable = True
        if "lfcbm" not in args.stages:
            args.stages.append("lfcbm")

    run(config, stages=args.stages, force_setup=args.force_setup)


if __name__ == "__main__":
    main()
