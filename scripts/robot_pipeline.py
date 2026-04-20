"""Robot classification benchmark pipeline.

Provides functions to run each stage of the robot benchmark programmatically.

Usage:
    python scripts/robot_pipeline.py --seed 1014
    python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes
    python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --regimes baseline expert
    python scripts/robot_pipeline.py --config my_config.yaml
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.generators import DatasetGenerator
from concept_benchmark.ext.fileutils import load, save
from experiments.cem_integration import (
    CEMDependencyError,
    compute_ecbm_interpretation_summary,
    train_cem_model,
    train_ecbm_model,
    train_probcbm_model,
)
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
    FrontEndModel,
    RobotClassifierCNN,
    RobotConceptClassifier,
)
from experiments.utils import run_alignment
from concept_benchmark.paths import data_dir, results_dir


@dataclass
class InterventionSettings:
    """Typed config for _test_interventions, replacing the old ``sttngs`` dict."""

    seed: int
    budgets: list[int]
    intervention_accuracy: float = 0.9
    intervention_threshold: float = 1.0
    intervention_strategy: str = "up_to_k"
    intervention_expert: str = ""  # "" for standard path, "llm" for inline LLM
    intervention_llm: dict[str, Any] | None = None
    run_dir: str = "."


class FEOnProbs(FrontEndModel):
    """Wrap an LFCBM sklearn classifier so it works as a FrontEndModel.

    Applies logit transform before predict_proba, matching how the
    LFCBM classifier was trained (on z-scored, then logit-transformed features).

    Exposes effective LR weights that fold the logit transform into the
    coefficients so KFlip can use the vectorized fast path.
    """

    _kflip_fast_path = True

    def __init__(self, clf):
        from types import SimpleNamespace

        super().__init__()
        self._clf = clf

        # Precompute effective weights: logit(binary) is affine in Z
        _eps = 1e-6
        lo = np.log(_eps / (1.0 - _eps))
        hi = np.log((1.0 - _eps) / _eps)
        scale = hi - lo
        coef = np.asarray(clf.coef_)
        intercept = np.asarray(clf.intercept_)
        self.model = SimpleNamespace(
            coef_=coef * scale,
            intercept_=lo * coef.sum(axis=1) + intercept,
        )

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        P = np.clip(P, 1e-6, 1 - 1e-6)
        Z = np.log(P / (1.0 - P))
        return self._clf.predict_proba(Z)


def _selected_cbm_key(config: RobotBenchmarkConfig) -> str:
    return str(getattr(config, "cbm_family", "cbm"))


AUTOMATED_REGIME_SPECS = {
    "llm": {
        "config_attr": "llm_concepts_file",
        "default_filename": "llm.jsonl",
    },
    "clip": {
        "config_attr": "clip_concepts_file",
        "default_filename": "clip.jsonl",
    },
    "placeholder3": {
        "config_attr": "placeholder3_concepts_file",
        "default_filename": None,
    },
}

# Machine uses the same LFCBM infrastructure but with human (not LLM)
# interventions, so it's kept separate from AUTOMATED_REGIME_SPECS to
# avoid unnecessary LLM cache prefills.
_LFCBM_REGIME_SPECS = {
    "machine": {
        "config_attr": "label_free_concepts_file",
        "default_filename": None,  # resolved dynamically based on concept_preset
    },
    **AUTOMATED_REGIME_SPECS,
}
AUTOMATED_REGIMES = frozenset(AUTOMATED_REGIME_SPECS)

# ── Concept source × intervention source (decoupled axes) ────────────
CONCEPT_SOURCES = [
    "ground_truth",       # 7 ideal concepts, human-annotated
    "human_concepts",     # 12 subconcepts, human-annotated
    "machine_annotation", # human descriptions, CLIP-labeled
    "llm_concepts",       # LLM descriptions, CLIP-labeled
    "clip_concepts",      # CLIP-Dissect keywords, CLIP-labeled
]
INTERVENTION_SOURCES = ["perfect", "expert", "llm"]

# Map concept source names to LFCBM regime keys (for auto-discovered sources)
_CONCEPT_SOURCE_TO_LFCBM = {
    "machine_annotation": "machine",
    "llm_concepts": "llm",
    "clip_concepts": "clip",
}

# Map old regime names to (concept_source, intervention_source) pairs
_REGIME_TO_CELL = {
    "baseline": ("human_concepts", "perfect"),
    "expert": ("human_concepts", "expert"),
    "machine": ("machine_annotation", "expert"),
    "llm": ("llm_concepts", "llm"),
    "clip": ("clip_concepts", "llm"),
}


def _resolve_automated_regime_concepts_file(
    config: RobotBenchmarkConfig, regime: str
) -> Path:
    """Resolve the concept-description file for an automated regime."""
    from concept_benchmark.paths import package_dir

    spec = _LFCBM_REGIME_SPECS.get(regime)
    if spec is None:
        raise ValueError(f"Unknown automated regime: {regime!r}")

    concepts_file = str(getattr(config, spec["config_attr"], "") or "").strip()
    if not concepts_file:
        default_filename = spec["default_filename"]
        if default_filename is None:
            # Machine regime: resolve dynamically like train_lfcbm()
            if regime == "machine":
                suffix = "_subconcept" if config.concept_preset == "foot_subtypes" else ""
                default_filename = f"gt_concepts{suffix}.jsonl"
            else:
                raise ValueError(
                    f"Regime {regime!r} requires config.{spec['config_attr']} to point "
                    "to a concepts JSONL file."
                )
        concepts_file = str(package_dir / "concept_descriptions" / default_filename)

    path = Path(concepts_file)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        if spec["default_filename"] is None:
            raise ValueError(
                f"Regime {regime!r} concepts file does not exist: {path}"
            )
        raise FileNotFoundError(f"concepts file not found: {path}")
    return path


def _automated_intervention_llm_settings(config: RobotBenchmarkConfig) -> dict[str, Any]:
    """Return the LLM config passed to automated intervention judgments."""
    provider = str(config.llm_provider).strip().lower()
    auto_batch_size = 4 if provider in {
        "codex",
        "codex_exec",
        "claude_exec",
    } else 100
    batch_size = int(config.llm_batch_size) if int(config.llm_batch_size) > 0 else auto_batch_size
    batch_sleep = (
        float(config.llm_batch_sleep)
        if float(config.llm_batch_sleep) >= 0.0
        else (0.0 if provider in {"codex", "codex_exec", "claude_exec"} else 5.0)
    )
    workers = int(config.llm_workers) if int(config.llm_workers) > 0 else 1
    return {
        "provider": config.llm_provider,
        "model": config.llm_model,
        "reasoning_effort": config.llm_reasoning_effort,
        "api_key": config.llm_api_key,
        "api_key_env": config.llm_api_key_env,
        "batch_size": batch_size,
        "batch_sleep": batch_sleep,
        "workers": workers,
        "cache_all_concepts": bool(config.llm_cache_all_concepts),
    }


# Lazy import to avoid circular deps — intervention modules
_intervention_imported = False


def _ensure_intervention_imports():
    global _intervention_imported
    if not _intervention_imported:
        global ConceptInterventionRunner
        global InterventionConfig
        global KFlipInterventionStrategy
        global predict_label_proba_from_concepts
        from experiments.intervention import (
            ConceptInterventionRunner,
            InterventionConfig,
            predict_label_proba_from_concepts,
        )
        from experiments.kflip import KFlipInterventionStrategy

        _intervention_imported = True


# ── Stage: setup_dataset ──────────────────────────────────────────────


def setup_dataset(config: RobotBenchmarkConfig):
    """Generate robot dataset, split, and save.

    Returns the saved ConceptDataset.
    """
    from concept_benchmark.config import (
        PRESET_EXCLUDED_CONCEPTS,
        ROBOT_SAMPLING_CONSTRAINTS,
    )

    logger.info("Generating robot dataset...")
    data = DatasetGenerator.from_config(config).generate()

    # Split with skewing constraints BEFORE dropping concepts.
    # The constraints reference subconcept names (e.g. foot_shape_pointy_square)
    # that exist in the full concept set.  Dropping concepts first would remove
    # them and disable skewing, changing the training set composition.
    test_size = 10000
    train_size = 3800
    remaining = data.n - test_size
    n_val = int((remaining - train_size) * 0.2)
    data.sample(
        test_size=test_size,
        val_size=n_val,
        train_size=train_size,
        sampling_constraints=ROBOT_SAMPLING_CONSTRAINTS,
        seed=config.seed,
    )

    data.drop_concepts(PRESET_EXCLUDED_CONCEPTS[config.concept_preset])
    save(data, config.get_dataset_path(), overwrite=True)
    return data


# ── Stage: train_cbm ──────────────────────────────────────────────────


def train_cbm(
    config: RobotBenchmarkConfig,
    data=None,
    save_key: str | None = "cbm",
    missing_fraction: float = 0.0,
    missing_mechanism: str = "mcar",
) -> ConceptBasedModel:
    """Train a ConceptBasedModel (concept detector + frontend).

    Args:
        save_key: Model save key. Use None to skip saving.
        missing_fraction: Fraction of concept labels to mask.
        missing_mechanism: Missingness mechanism ("mcar" or "mnar").

    Returns the trained CBM.
    """
    set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    # Apply concept missingness if configured
    if missing_fraction > 0:
        data.sample_concept_missingness(
            p=missing_fraction,
            mechanism=missing_mechanism,
            rng=np.random.default_rng(config.seed),
        )
        data.train.has_concept_missing = True

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
            num_concepts=data.train.n_concepts,
            input_size=config.input_size,
        )
    )
    cbm = ConceptBasedModel(concept_detector=cd)
    cbm.fit(
        train_dataset=data.train,
        valid_dataset=data.validation,
        freeze_backbone=False,
        concept_embed_params={"shuffle": False, **loader_config},
        concept_fit_params={
            "epochs": config.epochs,
            "lr": config.learning_rate,
            "patience": config.patience,
            **loader_config,
        },
    )

    test_pred = cbm.predict(data.test)
    logger.info("Test Accuracy: %s", np.mean(test_pred == data.test.y))

    if save_key is not None:
        save(cbm, config.get_model_path(save_key), overwrite=True)
    return cbm


def _train_wrapped_cbm_family(
    config: RobotBenchmarkConfig,
    *,
    family: str,
    data=None,
    save_key: str | None = None,
) -> ConceptBasedModel:
    set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    loader_config = get_loader_config()
    trainer_fn = {
        "cem": train_cem_model,
        "probcbm": train_probcbm_model,
        "ecbm": train_ecbm_model,
    }[family]
    model = trainer_fn(
        train_dataset=data.train,
        valid_dataset=data.validation,
        benchmark="robot",
        config=config,
        device=device,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )

    test_pred = model.predict(data.test)
    logger.info("Test Accuracy: %s", np.mean(test_pred == data.test.y))

    if save_key is not None:
        save(model, config.get_model_path(save_key), overwrite=True)
    return model


def _build_concept_groups(concept_names, concept_spec):
    """Build groups of concept column indices from the concept spec.

    For each key in ``concept_spec``, find all columns in ``concept_names``
    whose name equals or starts with ``key_``.  Any ungrouped columns get
    their own singleton group.

    Returns dict mapping group name → list of column indices.
    """
    groups = {}
    used = set()
    for base in list(concept_spec.keys()):
        idxs = [
            i
            for i, c in enumerate(concept_names)
            if c == base or c.startswith(f"{base}_")
        ]
        if idxs:
            groups[base] = idxs
            used.update(idxs)
    for i, c in enumerate(concept_names):
        if i not in used:
            groups[c] = [i]
    return groups


def _flip_onehot_row(row, idxs, rng):
    """Flip one active bit in a one-hot group to a random other option."""
    vals = row[idxs]
    if len(idxs) == 1:
        row[idxs[0]] = 1.0 - row[idxs[0]]
        return
    s = int(vals.sum())
    if s != 1:
        return
    active = int(np.argmax(vals))
    choices = [j for j in range(len(idxs)) if j != active]
    if not choices:
        return
    new_j = int(rng.choice(choices))
    row[idxs] = 0.0
    row[idxs[new_j]] = 1.0


def _apply_concept_noise_grouped(C, concept_names, concept_spec, rate, rng):
    """Apply group-level one-hot flip noise (original paper method).

    For each sample and each concept group, with probability ``rate``,
    randomly switch the active category to a different one within the group.
    """
    C_out = C.astype(np.float32).copy()
    groups = _build_concept_groups(concept_names, concept_spec)
    for r in range(C_out.shape[0]):
        for _, idxs in groups.items():
            if rng.random() < float(rate):
                _flip_onehot_row(C_out[r], idxs, rng)
    return C_out


def _clone_sample_with_C(sample, C_new, concept_names=None):
    """Clone a dataset sample with replaced concept matrix (and optionally concept names)."""
    meta = dict(sample.meta)
    if concept_names is not None:
        meta["concepts"] = list(concept_names)
    return sample.__class__(
        parent=sample.parent,
        X=sample.X,
        C=C_new.astype(np.float32),
        y=sample.y,
        meta=meta,
        transform=sample.transform,
        concept_transform=sample.concept_transform,
        target_transform=sample.target_transform,
        base_dir=getattr(sample, "base_dir", None),
    )


def train_cbm_subjective(
    config: RobotBenchmarkConfig,
    data=None,
) -> ConceptBasedModel:
    """Train a CBM on noisy (subjective) concept labels.

    Uses group-level one-hot flip noise (matching the original paper):
    for each sample and each concept group, with probability
    ``config.subjective_noise_rate``, randomly switch the active
    category to a different one within the group.

    The noisy CBM is saved to ``config.get_model_path("cbm_subjective")``.
    """
    if data is None:
        data = load(config.get_dataset_path())
    noisy_data = copy.deepcopy(data)

    # Offset seed so noise RNG is independent of data-generation RNG.
    # This specific offset (+555) reproduces the paper's noise patterns.
    rng = np.random.default_rng(config.seed + 555)
    concept_names = list(noisy_data.train.concepts)
    concept_spec = config.concepts

    # Apply group-level noise to training and validation splits
    noisy_data.train = _clone_sample_with_C(
        noisy_data.train,
        _apply_concept_noise_grouped(
            noisy_data.train.C,
            concept_names,
            concept_spec,
            config.subjective_noise_rate,
            rng,
        ),
    )
    if hasattr(noisy_data, "validation") and noisy_data.validation is not None:
        noisy_data.validation = _clone_sample_with_C(
            noisy_data.validation,
            _apply_concept_noise_grouped(
                noisy_data.validation.C,
                concept_names,
                concept_spec,
                config.subjective_noise_rate,
                rng,
            ),
        )

    cbm = train_cbm(config, data=noisy_data, save_key=None)
    save(cbm, config.get_model_path("cbm_subjective"), overwrite=True)
    return cbm


def _train_family_subjective(
    config: RobotBenchmarkConfig,
    family: str,
    data=None,
):
    """Train a non-CBM family (CEM/ProbCBM/ECBM) on noisy concept labels."""
    if data is None:
        data = load(config.get_dataset_path())
    noisy_data = copy.deepcopy(data)

    rng = np.random.default_rng(config.seed + 555)
    concept_names = list(noisy_data.train.concepts)
    concept_spec = config.concepts

    noisy_data.train = _clone_sample_with_C(
        noisy_data.train,
        _apply_concept_noise_grouped(
            noisy_data.train.C, concept_names, concept_spec,
            config.subjective_noise_rate, rng,
        ),
    )
    if hasattr(noisy_data, "validation") and noisy_data.validation is not None:
        noisy_data.validation = _clone_sample_with_C(
            noisy_data.validation,
            _apply_concept_noise_grouped(
                noisy_data.validation.C, concept_names, concept_spec,
                config.subjective_noise_rate, rng,
            ),
        )

    model = _train_wrapped_cbm_family(
        config, family=family, data=noisy_data, save_key=None,
    )
    save_key = f"{family}_subjective"
    save(model, config.get_model_path(save_key), overwrite=True)
    return model


def train_lfcbm(
    config: RobotBenchmarkConfig,
    data=None,
) -> dict:
    """Train a Label-Free CBM using CLIP embeddings.

    Requires ``open-clip-torch`` and a concepts JSONL file.
    Saves to ``config.get_model_path("lfcbm")``.
    """
    from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig

    set_deterministic_seed(config.seed)

    if data is None:
        data = load(config.get_dataset_path())

    concepts_file = config.label_free_concepts_file
    if not concepts_file:
        from concept_benchmark.paths import package_dir

        suffix = "_subconcept" if config.concept_preset == "foot_subtypes" else ""
        default = package_dir / "concept_descriptions" / f"gt_concepts{suffix}.jsonl"
        if default.exists():
            concepts_file = str(default)
        else:
            raise FileNotFoundError(
                f"No label-free concepts file: set config.label_free_concepts_file or create {default}"
            )

    concept_set = LFConceptSet.from_file(concepts_file)
    device_str = str(determine_device())

    cfg = LFTrainingConfig(
        device=device_str,
        seed=config.seed,
        cache_dir=config.get_model_path("lfcbm").parent / "lfcbm_cache",
    )
    lfcbm = LabelFreeCBM(cfg)

    # Extract image paths from dataset — X contains bare filenames like
    # "robot_003.png", so we prepend the image directory.
    image_dir = data_dir / "robot_images"
    train_paths = [str(image_dir / p) for p in data.train.X]
    valid_paths = [str(image_dir / p) for p in data.validation.X]

    stats = lfcbm.fit(
        train_X=train_paths,
        train_y=data.train.y.astype(int),
        valid_X=valid_paths,
        valid_y=data.validation.y.astype(int),
        concept_set=concept_set,
    )
    logger.info(
        "LFCBM stats: %s/%s concepts kept",
        stats.get("kept_concepts"),
        stats.get("total_concepts"),
    )

    out_dir = str(config.get_model_path("lfcbm")) + "_bundle"
    lfcbm.save(out_dir)
    bundle = {"lfcbm": lfcbm, "frontend": FEOnProbs(lfcbm.classifier)}
    save(bundle, config.get_model_path("lfcbm"), overwrite=True)
    return stats


# ── Stage: train_dnn ──────────────────────────────────────────────────


def train_dnn(
    config: RobotBenchmarkConfig,
    data=None,
) -> dict:
    """Train an end-to-end DNN baseline.

    Returns the best state_dict.
    """
    set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    torch.manual_seed(config.seed)
    model = RobotClassifierCNN(input_size=config.input_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

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
    # Move tensors to CPU before saving for cross-device portability
    weights = {k: v.cpu() for k, v in weights.items()}
    save(weights, config.get_model_path("dnn"), overwrite=True)
    return weights


# ── Intervention helper ───────────────────────────────────────────────


def _test_interventions(
    prob_test,
    settings: InterventionSettings,
    acc_det,
    fe,
    test,
    concept_names=None,
    model=None,
    *,
    cache_only: bool = False,
):
    """Run interventions for each budget and return results dict.

    Parameters
    ----------
    model : ConceptBasedModel, optional
        Full model (e.g. CEM/ProbCBM) for aligned concept replay.
        When provided, interventions replay through the model's learned
        embeddings instead of binarizing concepts.
    """
    import hashlib
    import json
    import time
    from pathlib import Path
    from types import SimpleNamespace

    _ensure_intervention_imports()

    intervention_results = {}
    rng = np.random.default_rng(settings.seed)
    budgets = list(settings.budgets)
    human_acc = settings.intervention_accuracy
    err_prob = 1.0 - human_acc

    _coerce_to_gt = concept_names is None
    if concept_names is None:
        concept_names = list(getattr(test, "concepts", []))
    else:
        concept_names = list(concept_names)

    # Coerce concept_proba to match dataset concept ground truth shape,
    # but only when using the bare CBM path (not aligned replay).
    # Aligned replay models operate in their own concept space.
    if _coerce_to_gt and hasattr(test, "C") and model is None:
        n_gt = int(test.C.shape[1])
        n_pred = int(prob_test.shape[1])
        if n_pred != n_gt:
            if n_pred > n_gt:
                prob_test = prob_test[:, :n_gt]
            else:
                pad = np.zeros(
                    (prob_test.shape[0], n_gt - n_pred), dtype=prob_test.dtype
                )
                prob_test = np.concatenate([prob_test, pad], axis=1)
        if len(concept_names) != prob_test.shape[1]:
            concept_names = concept_names[: prob_test.shape[1]]

    # Use the full model for aligned replay, or create a bare wrapper
    if model is not None:
        cbm = model
        # Slice test dataset concepts to match model's concept space.
        # The dataset may have more concepts (e.g., 19) than the model
        # was trained on (e.g., 7). The runner needs matching shapes.
        n_model = prob_test.shape[1]
        if hasattr(test, "C") and test.C.shape[1] != n_model:
            import copy

            test = copy.copy(test)
            test.C = test.C[:, :n_model]
    else:
        cbm = ConceptBasedModel(concept_detector=None, label_predictor=fe)
    runner = ConceptInterventionRunner(cbm)
    supports_aligned = bool(
        getattr(cbm, "supports_aligned_concept_replay", False)
        or getattr(fe, "supports_aligned_concept_replay", False)
    )
    llm_cache = None
    _intervention_cache = None

    print(f"Starting interventions: budgets={budgets}, strategy={settings.intervention_strategy}, expert={settings.intervention_expert}", flush=True)
    for b_idx, budget in enumerate(budgets, 1):
        print(f"Budget {b_idx}/{len(budgets)} (k={budget})...", flush=True)
        if int(budget) <= 0:
            key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
            n_samples = prob_test.shape[0]
            intervention_results[key] = {
                "accuracy": float(acc_det),
                "accuracy_gain": 0.0,
                "predictions_intervened_on": 0,
                "predictions_changed": 0,
                "interventions_rate": 0.0,
                "intervention_rate": 0.0,
                "avg_edits_per_intervention": 0.0,
                "total_concept_checks": 0,
                "total_concept_confirmations": 0,
                "total_concept_edits_made": 0,
                "concepts_intervened": {},
                "concepts_edits": {},
            }
            continue

        n_concepts = prob_test.shape[1]
        if int(budget) >= n_concepts:
            # k=max: intervene on ALL concepts → just replace with ground truth
            C_gt = test.C.astype(np.float32)
            C_pred = prob_test.copy()
            # Apply intervention noise
            if err_prob > 0:
                mistake_draw = rng.random(C_gt.shape) < err_prob
                C_noisy = C_gt.copy()
                C_noisy[mistake_draw] = 1.0 - C_gt[mistake_draw]
                C_intervened = C_noisy
            else:
                C_intervened = C_gt.copy()

            mask = np.ones_like(C_pred, dtype=bool)

            if supports_aligned and model is not None:
                y_prob_after = predict_label_proba_from_concepts(
                    model,
                    C_intervened,
                    row_indices=np.arange(C_intervened.shape[0], dtype=int),
                    baseline_concepts=(C_pred >= 0.5).astype(np.float32),
                    intervention_mask=mask,
                )
            else:
                C_binary = (C_intervened >= 0.5).astype(int)
                y_prob_after = fe.predict_proba(C_binary)
            y_pred_after = np.argmax(y_prob_after, axis=1)

            acc_after = float((y_pred_after == test.y.astype(int)).mean())
            C_pred_binary = (C_pred >= 0.5).astype(int)
            C_final_binary = (C_intervened >= 0.5).astype(int)
            edits = int(np.sum(C_pred_binary != C_final_binary))
            n_samples = prob_test.shape[0]
            y_pred_before = np.argmax(fe.predict_proba((C_pred >= 0.5).astype(int)), axis=1) if not supports_aligned else np.argmax(y_prob_after, axis=1)

            key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
            intervention_results[key] = {
                "accuracy": acc_after,
                "accuracy_gain": acc_after - acc_det,
                "predictions_intervened_on": n_samples,
                "predictions_changed": int(np.sum(y_pred_after != y_pred_before)),
                "interventions_rate": 1.0,
                "intervention_rate": 1.0,
                "avg_edits_per_intervention": edits / n_samples,
                "total_concept_checks": n_samples * n_concepts,
                "total_concept_confirmations": n_samples * n_concepts,
                "total_concept_edits_made": edits,
                "concepts_intervened": {},
                "concepts_edits": {},
            }
            print(f"  k=max short-circuit: acc={acc_after:.4f}", flush=True)
            continue

        config = InterventionConfig(
            max_concepts_per_instance=budget,
            random_state=settings.seed,
            score_threshold=settings.intervention_threshold,
            intervention_noise_rate=1.0 - human_acc,
        )

        strategy = KFlipInterventionStrategy(
            use_exact_k=(settings.intervention_strategy == "exactly_k"),
        )

        if settings.intervention_expert.lower() == "llm":
            # ── Inline LLM path (matching original robot_concept_regimes.py) ──
            from experiments.llm_client import (
                is_local_exec_provider,
                is_retryable_llm_error,
                make_llm_client,
            )

            llm_cfg = settings.intervention_llm or {}
            provider = str(llm_cfg.get("provider", "gemini"))
            model_name = str(llm_cfg.get("model", "gemini-3-flash-preview"))
            api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))

            import os

            api_key = str(llm_cfg.get("api_key", "")) or os.environ.get(api_key_env, "")

            # When cache_only=True, skip API key validation entirely —
            # the cache provides all LLM votes, no live calls needed.
            if cache_only:
                api_key = api_key or "cache-only-no-key-needed"
            elif not api_key and not is_local_exec_provider(provider):
                raise SystemExit(
                    f"missing API key: set llm_api_key in config or {api_key_env} in env"
                )

            reasoning_effort = str(llm_cfg.get("reasoning_effort", "") or "")
            cache_all_concepts = bool(llm_cfg.get("cache_all_concepts") or False)
            client = make_llm_client(
                provider,
                model_name,
                api_key,
                reasoning_effort=reasoning_effort,
            )

            def _llm_call_with_retry(fn, *, max_retries=5, backoff=30.0, label="LLM"):
                """Call *fn* and retry on transient provider failures."""
                for attempt in range(1, max_retries + 1):
                    try:
                        return fn()
                    except Exception as e:
                        if is_retryable_llm_error(provider, e):
                            logger.warning(
                                "%s transient %s attempt %d/%d: %s",
                                label,
                                e.__class__.__name__,
                                attempt,
                                max_retries,
                                e,
                            )
                            if attempt >= max_retries:
                                logger.error(
                                    "%s giving up after %d attempts.",
                                    label,
                                    max_retries,
                                )
                                raise
                            time.sleep(backoff)
                        else:
                            raise

            def _resolve_img_path(i: int) -> str:
                p = Path(str(test.X[i]))
                if p.is_absolute():
                    return str(p)
                return str((data_dir / "robot_images" / p).resolve())

            def _llm_judge(image_path: str, names: list) -> dict:
                prompt = (
                    "You will be shown one robot image. "
                    "For each concept below, output 0 or 1 indicating ABSENT(0) or PRESENT(1). "
                    "Return ONLY one JSON object with string keys and 0/1 integer values.\n\n"
                    "concepts:\n- " + "\n- ".join(names) + "\n\n"
                    'Respond like: {"conceptA":1,"conceptB":0}'
                )
                logger.debug(
                    "LLM fallback judge start image=%s, concepts=%d",
                    image_path,
                    len(names),
                )
                raw = _llm_call_with_retry(
                    lambda: (client.generate(prompt, [image_path]) or "").strip(),
                    label="LLM single-image",
                )
                parsed: dict = {}
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, bool):
                                parsed[str(k)] = 1 if v else 0
                            elif isinstance(v, (int, float, str)):
                                s = str(v).strip().lower()
                                parsed[str(k)] = (
                                    1 if s in {"1", "true", "yes", "present"} else 0
                                )
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.debug("JSON parse failure for LLM response: %s", e)
                return parsed

            def _llm_judge_batch(image_paths: list, per_image_names: list) -> list:
                N = len(image_paths)
                lines = []
                for i, names in enumerate(per_image_names):
                    lines.append(f"Image {i}: " + ", ".join(names))

                prompt = (
                    f"You will be shown {N} robot image(s) in order. "
                    f"For each image i (0..{N - 1}), output ONLY a JSON array of length {N} "
                    "where array[i] is a JSON object mapping the listed concepts to 0/1 integers "
                    "(ABSENT=0, PRESENT=1). No Markdown code fences, no extra keys, "
                    "and no text outside the JSON.\n\n"
                    "Per-image concepts:\n- " + "\n- ".join(lines)
                )

                raw = (client.generate(prompt, image_paths) or "").strip()

                # Strip Markdown ``` fences if the model ignores instructions
                if raw.startswith("```"):
                    fence_lines = raw.splitlines()
                    fence_lines = fence_lines[1:]
                    if fence_lines and fence_lines[-1].strip().startswith("```"):
                        fence_lines = fence_lines[:-1]
                    raw_clean = "\n".join(fence_lines).strip()
                else:
                    raw_clean = raw

                def _to01(v):
                    if isinstance(v, bool):
                        return 1 if v else 0
                    s = str(v).strip().lower()
                    return 1 if s in {"1", "true", "yes", "present"} else 0

                out: list = [dict() for _ in range(N)]

                try:
                    obj = json.loads(raw_clean)
                except Exception as e:
                    logger.debug("json.loads failed in _llm_judge_batch: %r", e)
                    logger.debug(
                        "raw_clean (first 400 chars): %s",
                        raw_clean[:400].replace("\n", "\\n"),
                    )
                    return out

                if isinstance(obj, list):
                    if len(obj) == 1 and isinstance(obj[0], list):
                        arr = obj[0]
                    else:
                        arr = obj

                    for i in range(min(N, len(arr))):
                        d = arr[i]
                        if not isinstance(d, dict):
                            continue
                        allow = set(per_image_names[i])
                        for k, v in d.items():
                            if k in allow:
                                out[i][str(k)] = _to01(v)

                elif isinstance(obj, dict):
                    for i_str, d in obj.items():
                        try:
                            i = int(i_str)
                        except Exception:
                            continue
                        if not (0 <= i < N and isinstance(d, dict)):
                            continue
                        allow = set(per_image_names[i])
                        for k, v in d.items():
                            if k in allow:
                                out[i][str(k)] = _to01(v)

                return out

            # compute-once at K=max(budgets), batch LLM once, reuse for smaller budgets
            if llm_cache is None:
                batch = runner._build_batch(
                    dataset=test,
                    concept_proba=prob_test,
                    concept_true=np.full_like(prob_test, np.nan, dtype=np.float32),
                    labels=test.y.astype(int),
                    instance_ids=None,
                )
                C_before = batch.C_pred
                if supports_aligned and model is not None:
                    y_prob_before = predict_label_proba_from_concepts(
                        cbm, C_before,
                        row_indices=np.arange(C_before.shape[0], dtype=int),
                        baseline_concepts=C_before,
                    )
                else:
                    y_prob_before = fe.predict_proba((C_before >= 0.5).astype(int))
                if cache_all_concepts and cache_only:
                    mask_max = np.ones_like(C_before, dtype=bool)
                else:
                    max_budget = max(int(b) for b in budgets)
                    maxK = int(min(max_budget, batch.C_pred.shape[1]))
                    config_max = InterventionConfig(
                        max_concepts_per_instance=maxK,
                        random_state=settings.seed,
                        score_threshold=settings.intervention_threshold,
                        intervention_noise_rate=1.0 - human_acc,
                    )
                    proposal_max = strategy.propose(cbm, batch, config_max)
                    mask_max = proposal_max.mask

                C_true_llm = np.full_like(C_before, np.nan, dtype=float)

                # JSONL on-disk cache
                run_root = Path(settings.run_dir)
                cache_dir = run_root / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

                def _concepts_sig():
                    h = hashlib.sha1()
                    for name in map(str, concept_names):
                        h.update(name.encode("utf-8"))
                        h.update(b"\x00")
                    return h.hexdigest()

                def _dataset_sig():
                    h = hashlib.sha1()
                    for pth in map(str, test.X):
                        h.update(pth.encode("utf-8"))
                        h.update(b"\x00")
                    return h.hexdigest()

                cache_path = (
                    cache_dir
                    / f"llm_interventions_{_concepts_sig()}_{_dataset_sig()}.jsonl"
                )

                def _load_cache():
                    d = {}
                    if cache_path.exists():
                        with open(cache_path, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    rec = json.loads(line)
                                    i0 = int(rec["i"])
                                    votes_idx = {
                                        int(k): int(v)
                                        for k, v in rec.get("votes_idx", {}).items()
                                    }
                                    d[i0] = votes_idx
                                except Exception:
                                    continue
                    return d

                def _append_cache_rows(row_ids):
                    if not row_ids:
                        return
                    with open(cache_path, "a", encoding="utf-8") as f:
                        for i0 in sorted(row_ids):
                            votes_idx = _intervention_cache.get(i0)
                            if votes_idx is None:
                                continue
                            f.write(
                                json.dumps(
                                    {
                                        "i": int(i0),
                                        "votes_idx": {
                                            str(k): int(v) for k, v in votes_idx.items()
                                        },
                                    }
                                )
                                + "\n"
                            )

                if _intervention_cache is None:
                    _intervention_cache = _load_cache()

                tasks = []
                total_pairs = 0
                missing_pairs = 0
                all_idxs = np.arange(C_before.shape[1], dtype=int)
                for i in range(C_before.shape[0]):
                    idxs = all_idxs if cache_all_concepts else np.where(mask_max[i])[0]
                    if idxs.size == 0:
                        continue
                    total_pairs += int(idxs.size)
                    known = _intervention_cache.get(i, {})
                    for j in idxs:
                        if j in known:
                            C_true_llm[i, j] = float(known[j])
                    missing = [j for j in idxs if j not in known]
                    missing_pairs += len(missing)
                    if not missing:
                        continue
                    image_path = _resolve_img_path(i)
                    names = [str(concept_names[j]) for j in missing]
                    tasks.append((i, image_path, names, missing))

                cached_pairs = total_pairs - missing_pairs
                logger.info(
                    "LLM %sselection: total=%d, from_cache=%d, "
                    "to_query=%d, images_needing_llm=%d",
                    "exhaustive " if cache_all_concepts else "intervention ",
                    total_pairs,
                    cached_pairs,
                    missing_pairs,
                    len(tasks),
                )

                if cache_only and tasks:
                    logger.warning(
                        "cache_only=True but %d images have %d missing concept pairs. "
                        "Using NaN for missing entries.",
                        len(tasks), missing_pairs,
                    )
                    tasks = []  # skip LLM calls, proceed with what the cache has

                bs = int(llm_cfg.get("batch_size") or 32)
                if bs < 1:
                    bs = 1
                n_batches = (len(tasks) + bs - 1) // bs
                if n_batches > 0:
                    logger.info(
                        "LLM starting batched calls: %d images, batch_size=%d, n_batches=%d",
                        len(tasks),
                        bs,
                        n_batches,
                    )

                retry_backoff = float(llm_cfg.get("retry_backoff") or 30.0)
                max_retries = int(llm_cfg.get("max_retries") or 5)
                sleep_time = float(llm_cfg.get("batch_sleep") or 5.0)
                workers = int(llm_cfg.get("workers") or 1)
                if workers < 1:
                    workers = 1

                def _apply_llm_votes(chunk, votes_list):
                    changed_rows = set()
                    for (i_idx, _pth, names, idxs), votes in zip(chunk, votes_list):
                        if i_idx not in _intervention_cache:
                            _intervention_cache[i_idx] = {}

                        for j, name in zip(idxs, names):
                            if name in votes:
                                v = 1 if votes[name] else 0
                                C_true_llm[i_idx, j] = float(v)
                                _intervention_cache[i_idx][j] = v
                                changed_rows.add(i_idx)
                    return changed_rows
                def _run_llm_chunk(batch_idx, chunk):
                    image_paths = [p for (_i, p, _n, _j) in chunk]
                    per_image_names = [names for (_i, _p, names, _idxs) in chunk]
                    votes_list = _llm_call_with_retry(
                        lambda: _llm_judge_batch(image_paths, per_image_names),
                        max_retries=max_retries,
                        backoff=retry_backoff,
                        label=f"LLM batch {batch_idx}/{n_batches}",
                    )
                    if sleep_time > 0.0:
                        logger.debug(
                            "LLM batch %d/%d ok; sleeping %.1fs to respect rate limits",
                            batch_idx,
                            n_batches,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                    return batch_idx, chunk, votes_list

                if workers == 1:
                    for batch_idx, s in enumerate(range(0, len(tasks), bs), start=1):
                        chunk = tasks[s : s + bs]
                        _, chunk_done, votes_list = _run_llm_chunk(batch_idx, chunk)
                        changed_rows = _apply_llm_votes(chunk_done, votes_list)
                        _append_cache_rows(changed_rows)
                        logger.debug(
                            "LLM batch %d/%d complete; cache rows appended.",
                            batch_idx,
                            n_batches,
                        )
                else:
                    logger.info(
                        "LLM using %d concurrent worker(s) across %d batches.",
                        workers,
                        n_batches,
                    )
                    batch_cursor = enumerate(range(0, len(tasks), bs), start=1)
                    pending: dict[cf.Future, int] = {}

                    def _submit_next(executor: cf.ThreadPoolExecutor) -> bool:
                        try:
                            batch_idx, s = next(batch_cursor)
                        except StopIteration:
                            return False
                        chunk = tasks[s : s + bs]
                        fut = executor.submit(_run_llm_chunk, batch_idx, chunk)
                        pending[fut] = batch_idx
                        return True

                    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
                        for _ in range(min(workers, n_batches)):
                            _submit_next(executor)

                        while pending:
                            done, _ = cf.wait(
                                pending,
                                return_when=cf.FIRST_COMPLETED,
                            )
                            for fut in done:
                                batch_idx = pending.pop(fut)
                                chunk_done = None
                                votes_list = None
                                try:
                                    _, chunk_done, votes_list = fut.result()
                                except Exception:
                                    for other in pending:
                                        other.cancel()
                                    raise

                                changed_rows = _apply_llm_votes(chunk_done, votes_list)
                                _append_cache_rows(changed_rows)
                                logger.debug(
                                    "LLM batch %d/%d complete; cache rows appended.",
                                    batch_idx,
                                    n_batches,
                                )
                                _submit_next(executor)

                if n_batches > 0:
                    logger.info("LLM all %d batches complete.", n_batches)
                if cache_only and cache_all_concepts:
                    logger.info(
                        "LLM exhaustive cache-only mode complete; "
                        "skipping intervention scoring."
                    )
                    return [budget], human_acc, {}

                # rank concepts per instance by flip effect (reuse for budgets < K)
                order = [np.array([], dtype=int)] * C_before.shape[0]
                if supports_aligned and model is not None:
                    # Aligned models (CEM/ProbCBM/ECBM) can't do per-row predict_proba.
                    # Rank by concept uncertainty: concepts closest to 0.5 are most
                    # likely to benefit from intervention.
                    for i in range(C_before.shape[0]):
                        sel = np.where(mask_max[i])[0]
                        if sel.size == 0:
                            order[i] = np.array([], dtype=int)
                            continue
                        uncertainty = 0.5 - np.abs(C_before[i, sel] - 0.5)
                        ranked = sel[np.argsort(-uncertainty)]
                        order[i] = ranked.astype(int)
                else:
                    for i in range(C_before.shape[0]):
                        sel = np.where(mask_max[i])[0]
                        if sel.size == 0:
                            order[i] = np.array([], dtype=int)
                            continue
                        base_vec = (C_before[i] >= 0.5).astype(int)
                        base_prob = fe.predict_proba(base_vec[None, :])[0]
                        pairs = []
                        for j in sel:
                            flipped = base_vec.copy()
                            flipped[j] = 1 - flipped[j]
                            p_after = fe.predict_proba(flipped[None, :])[0]
                            score = float(np.max(np.abs(p_after - base_prob)))
                            pairs.append((j, score))
                        order[i] = np.asarray(
                            [j for (j, _) in sorted(pairs, key=lambda t: t[1], reverse=True)],
                            dtype=int,
                        )

                llm_cache = {
                    "mask_max": mask_max,
                    "C_true_llm": C_true_llm,
                    "C_before": C_before,
                    "y_prob_before": y_prob_before,
                    "order": order,
                }

            # derive current-budget mask from cached K=max selection
            mask_max = llm_cache["mask_max"]
            C_true_llm = llm_cache["C_true_llm"]
            C_before = llm_cache["C_before"]
            y_prob_before = llm_cache["y_prob_before"]
            order = llm_cache["order"]

            mask = np.zeros_like(mask_max, dtype=bool)
            for i in range(mask.shape[0]):
                if order[i].size:
                    k_take = int(min(budget, order[i].size))
                    if k_take > 0:
                        mask[i, order[i][:k_take]] = True

            overwrite_mask = mask & ~np.isnan(C_true_llm)
            C_after = np.where(overwrite_mask, C_true_llm, C_before)
            if supports_aligned:
                y_prob_after = predict_label_proba_from_concepts(
                    cbm,
                    C_after,
                    row_indices=np.arange(C_after.shape[0], dtype=int),
                    baseline_concepts=C_before,
                    intervention_mask=overwrite_mask,
                )
            else:
                C_final_binary = (C_after >= 0.5).astype(int)
                y_prob_after = fe.predict_proba(C_final_binary)
            y_pred_after = np.argmax(y_prob_after, axis=1)

            result = SimpleNamespace(
                C_pred=C_before,
                C_intervened=C_after,
                mask=overwrite_mask,
                y_prob_before=y_prob_before,
                y_prob_after=y_prob_after,
                y_pred_after=y_pred_after,
            )

        else:
            # ── Standard (non-LLM) path ──
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

            # Recompute downstream prediction after error injection
            if supports_aligned:
                result.y_prob_after = predict_label_proba_from_concepts(
                    cbm,
                    result.C_intervened,
                    row_indices=np.arange(result.C_intervened.shape[0], dtype=int),
                    baseline_concepts=result.C_pred,
                    intervention_mask=result.mask,
                )
            else:
                C_final_binary = (result.C_intervened >= 0.5).astype(int)
                result.y_prob_after = fe.predict_proba(C_final_binary)
            result.y_pred_after = np.argmax(result.y_prob_after, axis=1)

        # Extract intervention statistics
        acc_intervened = float((result.y_pred_after == test.y.astype(int)).mean())

        n_intervened = int(np.sum(result.mask))
        n_samples = prob_test.shape[0]

        intervened_concepts = np.any(result.mask, axis=0)
        C_pred_binary = (result.C_pred >= 0.5).astype(int)
        C_final_binary = (result.C_intervened >= 0.5).astype(int)
        actual_edits_mask = C_pred_binary != C_final_binary
        prediction_num_concepts_intervened_on = {
            int(i): int(np.sum(actual_edits_mask[i])) for i in range(n_samples)
        }

        y_pred_before = np.argmax(result.y_prob_before, axis=1)
        num_preds_change = int(np.sum(result.y_pred_after != y_pred_before))

        concept_intervention_counts = {
            c: f"{int(np.sum(result.mask[:, i]))} ({int(np.sum(actual_edits_mask[:, i]))})"
            for i, c in enumerate(concept_names)
            if i < intervened_concepts.shape[0] and intervened_concepts[i]
        }

        key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
        intervention_results[key] = {
            "accuracy": acc_intervened,
            "accuracy_gain": acc_intervened - acc_det,
            "predictions_intervened_on": int(np.sum(np.any(result.mask, axis=1))),
            "interventions_rate": float(
                np.sum(np.any(result.mask, axis=1)) / n_samples
            ),
            "predictions_changed": num_preds_change,
            "avg_edits_per_intervention": float(
                sum(prediction_num_concepts_intervened_on.values())
            )
            / n_samples,
            "total_concept_confirmations": int(n_intervened),
            "total_concept_edits_made": int(
                sum(prediction_num_concepts_intervened_on.values())
            ),
            "concept_interventions": concept_intervention_counts,
            "human_accuracy": human_acc,
        }

    return budgets, human_acc, intervention_results


# ── Automated regime helper ───────────────────────────────────────────


def _prepare_automated_regime_backend(config, regime, data):
    """Load or train the automated regime backend and return test-time inputs."""
    lf = _load_or_train_regime_lfcbm(config, regime, data)

    image_dir = data_dir / "robot_images"
    test_paths = [str(image_dir / p) for p in data.test.X]
    print(f"Computing concept_proba for {len(test_paths)} test images (regime={regime})...", flush=True)
    P_te = lf.concept_proba(test_paths)
    print(f"concept_proba done, shape={P_te.shape}", flush=True)

    fe = FEOnProbs(lf.classifier)
    y_pred_det = fe.predict_proba(P_te)
    acc_det = float((y_pred_det.argmax(1) == data.test.y.astype(int)).mean())
    return {
        "concept_names": list(lf.concept_set.keys),
        "prob_test": P_te,
        "fe": fe,
        "acc_det": acc_det,
    }


def _load_or_train_regime_lfcbm(config, regime, data):
    """Load or train the LFCBM for an automated regime. Returns the LabelFreeCBM object."""
    from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig

    concepts_path = _resolve_automated_regime_concepts_file(config, regime)
    concept_set = LFConceptSet.from_file(str(concepts_path))
    if not getattr(concept_set, "texts", None):
        raise ValueError(f"concepts file parsed empty: {concepts_path}")

    lfcbm_key = f"lfcbm_{regime}"
    lfcbm_path = config.get_model_path(lfcbm_key)

    if lfcbm_path.exists() and not config.force_retrain:
        logger.info("Loading existing LFCBM for %s: %s", regime, lfcbm_path)
        return load(lfcbm_path)

    device_str = str(determine_device())
    cfg = LFTrainingConfig(
        device=device_str,
        seed=config.seed,
        cache_dir=config.get_model_path("lfcbm").parent / f"lfcbm_{regime}_cache",
    )
    lf = LabelFreeCBM(cfg)
    image_dir = data_dir / "robot_images"
    train_paths = [str(image_dir / p) for p in data.train.X]
    valid_paths = [str(image_dir / p) for p in data.validation.X]
    stats = lf.fit(
        train_X=train_paths,
        train_y=data.train.y.astype(int),
        valid_X=valid_paths,
        valid_y=data.validation.y.astype(int),
        concept_set=concept_set,
        cache_dir=cfg.cache_dir,
    )
    logger.info(
        "LFCBM (%s) stats: %s/%s concepts kept",
        regime, stats.get("kept_concepts"), stats.get("total_concepts"),
    )
    save(lf, lfcbm_path, overwrite=True)
    return lf


def _prepare_lfcbm_labeled_data(lf, data):
    """Create a copy of data with concept labels replaced by LFCBM predictions."""
    image_dir = data_dir / "robot_images"
    concept_names = list(lf.concept_set.keys)

    splits = {}
    for name in ("train", "validation", "test"):
        sample = getattr(data, name, None)
        if sample is None:
            continue
        paths = [str(image_dir / p) for p in sample.X]
        print(f"LFCBM concept_proba for {name} ({len(paths)} images)...", flush=True)
        C_new = (lf.concept_proba(paths) >= 0.5).astype(np.float32)
        splits[name] = _clone_sample_with_C(sample, C_new, concept_names=concept_names)

    data_lf = copy.deepcopy(data)
    for name, sample in splits.items():
        setattr(data_lf, name, sample)
    return data_lf


def _run_automated_regime_with_family(
    config, regime, family, data, budgets, thresholds
):
    """Train a CBM family on LFCBM-derived concept labels and run interventions."""
    _ensure_intervention_imports()

    lf = _load_or_train_regime_lfcbm(config, regime, data)
    data_lf = _prepare_lfcbm_labeled_data(lf, data)

    # Train family model on LFCBM-labeled data (or load cached)
    model_key = f"{family}_{regime}"
    model_path = config.get_model_path(model_key)
    if model_path.exists() and not config.force_retrain:
        logger.info("Loading cached %s for regime %s: %s", family, regime, model_path)
        regime_model = load(model_path)
    else:
        print(f"Training {family} on LFCBM concepts (regime={regime})...", flush=True)
        regime_model = _train_wrapped_cbm_family(
            config, family=family, data=data_lf, save_key=None,
        )
        save(regime_model, model_path, overwrite=True)
        print(f"{family} trained and saved to {model_path}", flush=True)

    # Get concept predictions + accuracy
    c_preds = regime_model.concept_detector.predict_proba(data_lf.test)
    acc_det = float(
        (regime_model.predict(data_lf.test) == data_lf.test.y.astype(int)).mean()
    )
    print(f"{family}/{regime} acc_det={acc_det:.4f}", flush=True)

    # Run interventions — use regime model's own mechanism
    # Ground truth for corrections = LFCBM concept labels (data_lf.test.C)
    supports_aligned = bool(
        getattr(regime_model, "supports_aligned_concept_replay", False)
    )

    METRIC_COLS = [
        "accuracy", "predictions_intervened_on", "predictions_changed",
        "total_concept_confirmations", "total_concept_edits_made",
    ]
    COLS = ["budget", "threshold"] + METRIC_COLS
    df_lst = []
    for t in thresholds:
        isettings = InterventionSettings(
            seed=config.seed,
            budgets=budgets,
            intervention_accuracy=config.expert_intervention_accuracy,
            intervention_threshold=t,
            intervention_strategy=config.intervention_strategy,
        )
        _, _, r = _test_interventions(
            prob_test=c_preds,
            settings=isettings,
            acc_det=acc_det,
            fe=regime_model.label_predictor,
            test=data_lf.test,
            model=regime_model if supports_aligned else None,
        )
        df_lst.append(
            pd.DataFrame(r).T
            .assign(budget=budgets)
            .assign(threshold=t)
            .reset_index(drop=True)[COLS]
        )

    regime_df = pd.concat(df_lst, axis=0).reset_index(drop=True)
    regime_df["regime"] = regime
    return regime_df


# ── Decoupled concept source × intervention source ───────────────────


def _prepare_model_for_concept_source(config, concept_source, family, data):
    """Load or train the model for a given concept source and architecture.

    Returns (model, c_preds, concept_names, test_data) where test_data.C
    contains the ground truth concept values for this concept source.
    """
    lfcbm_key = _CONCEPT_SOURCE_TO_LFCBM.get(concept_source)

    if lfcbm_key is not None:
        # Auto-discovered concepts: load LFCBM, get concept labels, train family
        lf = _load_or_train_regime_lfcbm(config, lfcbm_key, data)
        data_lf = _prepare_lfcbm_labeled_data(lf, data)

        if family == "cbm":
            # Use LFCBM end-to-end (FEOnProbs)
            image_dir = data_dir / "robot_images"
            test_paths = [str(image_dir / p) for p in data.test.X]
            c_preds = lf.concept_proba(test_paths)
            model = ConceptBasedModel(
                concept_detector=None,
                label_predictor=FEOnProbs(lf.classifier),
            )
            return model, c_preds, list(lf.concept_set.keys), data_lf.test
        else:
            # Train family model on LFCBM-labeled data
            model_key = f"{family}_{lfcbm_key}"
            model_path = config.get_model_path(model_key)
            if model_path.exists() and not config.force_retrain:
                logger.info("Loading cached %s for %s: %s", family, concept_source, model_path)
                model = load(model_path)
            else:
                print(f"Training {family} on {concept_source} concepts...", flush=True)
                model = _train_wrapped_cbm_family(
                    config, family=family, data=data_lf, save_key=None,
                )
                save(model, model_path, overwrite=True)
            c_preds = model.concept_detector.predict_proba(data_lf.test)
            return model, c_preds, list(lf.concept_set.keys), data_lf.test
    else:
        # GT concepts: load the baseline model for this family
        if concept_source == "ground_truth":
            # Use ideal preset (7 concepts) — need a separate config
            gt_config = RobotBenchmarkConfig(seed=config.seed)
            gt_config.rng_seed = getattr(config, "rng_seed", 12345)
            gt_config.cbm_family = family
            gt_data = load(gt_config.get_dataset_path())
            model = load(gt_config.get_model_path(family))
            c_preds = model.concept_detector.predict_proba(gt_data.test)
            return model, c_preds, list(gt_data.test.concepts), gt_data.test
        else:
            # human_concepts — uses subconcept preset (already in config)
            model = load(config.get_model_path(family))
            c_preds = model.concept_detector.predict_proba(data.test)
            return model, c_preds, list(data.test.concepts), data.test


def _run_cell(config, concept_source, intervention_source, family, data,
              budgets, thresholds):
    """Run one cell of the concept_source × intervention_source matrix."""
    _ensure_intervention_imports()

    print(f"=== Cell: {concept_source} × {intervention_source} × {family} ===", flush=True)

    model, c_preds, concept_names, test_data = _prepare_model_for_concept_source(
        config, concept_source, family, data,
    )

    # Determine intervention parameters from intervention_source
    if intervention_source == "perfect":
        human_acc = 1.0
        expert_type = ""
    elif intervention_source == "expert":
        human_acc = config.expert_intervention_accuracy
        expert_type = ""
    elif intervention_source == "llm":
        human_acc = config.expert_intervention_accuracy
        expert_type = "llm"
    else:
        raise ValueError(f"Unknown intervention_source: {intervention_source!r}")

    # Compute baseline accuracy
    supports_aligned = bool(
        getattr(model, "supports_aligned_concept_replay", False)
    )
    if family == "cbm" and concept_source in _CONCEPT_SOURCE_TO_LFCBM:
        # LFCBM path: use continuous probs for accuracy
        acc_det = float(
            (np.argmax(model.label_predictor.predict_proba(c_preds), axis=1)
             == test_data.y.astype(int)).mean()
        )
    else:
        acc_det = float(
            (model.predict(test_data) == test_data.y.astype(int)).mean()
        )

    METRIC_COLS = [
        "accuracy", "predictions_intervened_on", "predictions_changed",
        "total_concept_confirmations", "total_concept_edits_made",
    ]
    COLS = ["budget", "threshold"] + METRIC_COLS
    df_lst = []
    for t in thresholds:
        isettings = InterventionSettings(
            seed=config.seed,
            budgets=budgets,
            intervention_accuracy=human_acc,
            intervention_threshold=t,
            intervention_strategy=config.intervention_strategy,
        )
        if expert_type == "llm":
            isettings.intervention_expert = "llm"
            isettings.intervention_llm = _automated_intervention_llm_settings(config)
            isettings.run_dir = str(results_dir)

        _, _, r = _test_interventions(
            prob_test=c_preds,
            settings=isettings,
            acc_det=acc_det,
            fe=model.label_predictor,
            test=test_data,
            concept_names=concept_names,
            model=model if supports_aligned else None,
            cache_only=bool(
                expert_type == "llm" and getattr(config, "llm_cache_only", False)
            ),
        )
        df_lst.append(
            pd.DataFrame(r).T
            .assign(budget=budgets)
            .assign(threshold=t)
            .reset_index(drop=True)[COLS]
        )

    cell_df = pd.concat(df_lst, axis=0).reset_index(drop=True)
    cell_df["concept_source"] = concept_source
    cell_df["intervention_source"] = intervention_source
    print(f"Cell {concept_source}×{intervention_source}×{family} DONE. Rows: {len(cell_df)}", flush=True)
    return cell_df


def _run_automated_regime(config, regime, model, data, budgets, thresholds):
    """Run an automated intervention regime backed by LFCBM + LLM judgments."""

    del model
    backend = _prepare_automated_regime_backend(config, regime, data)

    # Matching original: human_annotation_accuracy = 0.8
    ia_val = config.expert_intervention_accuracy

    METRIC_COLS = [
        "accuracy",
        "predictions_intervened_on",
        "predictions_changed",
        "total_concept_confirmations",
        "total_concept_edits_made",
    ]

    COLS = ["budget", "threshold"] + METRIC_COLS
    all_dfs = []

    for t in thresholds:
        isettings = InterventionSettings(
            seed=config.seed,
            budgets=budgets,
            intervention_accuracy=ia_val,
            intervention_threshold=t,
            intervention_strategy=config.intervention_strategy,
            intervention_expert="llm",
            intervention_llm=_automated_intervention_llm_settings(config),
            run_dir=str(results_dir),
        )

        _, _, r = _test_interventions(
            prob_test=backend["prob_test"],
            settings=isettings,
            acc_det=backend["acc_det"],
            fe=backend["fe"],
            test=data.test,
            concept_names=backend["concept_names"],
            cache_only=bool(config.llm_cache_only),
        )
        df = (
            pd.DataFrame(r)
            .T.assign(budget=budgets)
            .assign(threshold=t)
            .reset_index(drop=True)[COLS]
        )
        all_dfs.append(df)

    regime_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    regime_df["regime"] = regime
    return regime_df


def prefill_automated_intervention_caches(
    config: RobotBenchmarkConfig,
    data=None,
) -> dict[str, int]:
    """Populate JSONL caches for automated regimes without scoring interventions."""
    if data is None:
        data = load(config.get_dataset_path())

    automated_regimes = [
        regime for regime in config.intervention_regimes if regime in AUTOMATED_REGIMES
    ]
    if not automated_regimes:
        raise ValueError(
            "llm_cache_only requires at least one automated regime "
            f"from {sorted(AUTOMATED_REGIMES)}."
        )

    budgets = sorted(
        set(
            [0]
            + [data.n_concepts if b == -1 else b for b in config.intervention_budgets]
        )
    )
    max_budget = max(int(b) for b in budgets)
    cache_all_concepts = bool(config.llm_cache_all_concepts)
    thresholds = [0.0] if cache_all_concepts else list(config.intervention_thresholds)
    summary: dict[str, int] = {}

    for regime in automated_regimes:
        backend = _prepare_automated_regime_backend(config, regime, data)
        for t in thresholds:
            isettings = InterventionSettings(
                seed=config.seed,
                budgets=[max_budget],
                intervention_accuracy=config.expert_intervention_accuracy,
                intervention_threshold=t,
                intervention_strategy=config.intervention_strategy,
                intervention_expert="llm",
                intervention_llm=_automated_intervention_llm_settings(config),
                run_dir=str(results_dir),
            )
            _test_interventions(
                prob_test=backend["prob_test"],
                settings=isettings,
                acc_det=backend["acc_det"],
                fe=backend["fe"],
                test=data.test,
                concept_names=backend["concept_names"],
                cache_only=True,
            )
            summary[f"{regime}@{t}"] = max_budget

    return summary


# ── Regime dispatch ───────────────────────────────────────────────────


def _run_regime(config, regime, model, data, budgets, thresholds):
    """Run one intervention regime. Returns list of result row dicts.

    ``model`` is always the *baseline* CBM (loaded once by the caller).
    For regimes that use a different CBM (e.g. "subjective"), this
    function loads the regime-specific model internally.
    """
    _ensure_intervention_imports()

    METRIC_COLS = [
        "accuracy",
        "predictions_intervened_on",
        "predictions_changed",
        "total_concept_confirmations",
        "total_concept_edits_made",
    ]

    # Select model, concept predictions, and human accuracy per regime
    c_preds = None  # set below; None means use regime_model.concept_detector
    regime_concept_names = None  # set for LFCBM regimes; None → use GT concepts
    if regime == "baseline":
        regime_model = model
        human_acc = config.intervention_accuracy
    elif regime == "expert":
        regime_model = model
        human_acc = config.expert_intervention_accuracy
    elif regime == "subjective":
        family = _selected_cbm_key(config)
        subj_key = f"{family}_subjective" if family != "cbm" else "cbm_subjective"
        regime_model = load(config.get_model_path(subj_key))
        human_acc = config.subjective_intervention_accuracy
    elif regime == "machine":
        family = _selected_cbm_key(config)
        if family != "cbm":
            return _run_automated_regime_with_family(
                config, regime, family, data, budgets, thresholds,
            )
        # CBM path: use LFCBM end-to-end (original behavior)
        lfcbm_bundle = load(config.get_model_path("lfcbm"))
        lfcbm_obj = lfcbm_bundle["lfcbm"]
        fe_machine = lfcbm_bundle["frontend"]
        image_dir = data_dir / "robot_images"
        test_paths = [str(image_dir / p) for p in data.test.X]
        c_preds = lfcbm_obj.concept_proba(test_paths)
        regime_concept_names = list(lfcbm_obj.concept_set.keys)
        regime_model = ConceptBasedModel(
            concept_detector=None, label_predictor=fe_machine
        )
        human_acc = config.expert_intervention_accuracy
    elif regime in AUTOMATED_REGIMES:
        family = _selected_cbm_key(config)
        if family == "cbm":
            return _run_automated_regime(config, regime, model, data, budgets, thresholds)
        return _run_automated_regime_with_family(
            config, regime, family, data, budgets, thresholds,
        )
    else:
        raise ValueError(f"Unknown regime: {regime!r}")

    if c_preds is None:
        c_preds = regime_model.concept_detector.predict_proba(data.test)
    # For machine regime (FEOnProbs), pass continuous probs directly;
    # for other regimes, binarize first (matching original code).
    if regime == "machine":
        acc_det = float(
            (
                np.argmax(regime_model.label_predictor.predict_proba(c_preds), axis=1)
                == data.test.y.astype(int)
            ).mean()
        )
    else:
        acc_det = float(
            (regime_model.predict(data.test) == data.test.y.astype(int)).mean()
        )

    COLS = ["budget", "threshold"] + METRIC_COLS
    df_lst = []
    for t in thresholds:
        isettings = InterventionSettings(
            seed=config.seed,
            budgets=budgets,
            intervention_accuracy=human_acc,
            intervention_threshold=t,
            intervention_strategy=config.intervention_strategy,
        )
        # Pass the full model for CEM/ProbCBM so aligned concept replay
        # is used instead of bare binarization.
        use_full_model = getattr(
            regime_model, "supports_aligned_concept_replay", False
        )
        _, _, r = _test_interventions(
            prob_test=c_preds,
            settings=isettings,
            acc_det=acc_det,
            fe=regime_model.label_predictor,
            test=data.test,
            concept_names=regime_concept_names,
            model=regime_model if use_full_model else None,
        )
        df_lst.append(
            pd.DataFrame(r)
            .T.assign(budget=budgets)
            .assign(threshold=t)
            .reset_index(drop=True)[COLS]
        )

    regime_df = pd.concat(df_lst, axis=0).reset_index(drop=True)
    regime_df["regime"] = regime
    return regime_df


# ── Stage: run_interventions ──────────────────────────────────────────


def run_interventions(
    config: RobotBenchmarkConfig,
    data=None,
    missing_fraction: float = 0.0,
    missing_mechanism: str = "none",
) -> pd.DataFrame:
    """Run interventions across concept_sources × intervention_sources."""
    set_deterministic_seed(config.seed)
    _ensure_intervention_imports()
    patch_macos_dataloader()
    determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    family = _selected_cbm_key(config)
    concept_sources = getattr(config, "concept_sources", None) or ["human_concepts"]
    intervention_sources = getattr(config, "intervention_sources", None) or ["perfect"]

    budgets = sorted(
        set(
            [0]
            + [data.n_concepts if b == -1 else b for b in config.intervention_budgets]
        )
    )
    thresholds = config.intervention_thresholds

    all_dfs = []
    total = len(concept_sources) * len(intervention_sources)
    idx = 0
    out_path = config.get_results_path(family)
    for cs in concept_sources:
        for isrc in intervention_sources:
            idx += 1
            print(f"[{idx}/{total}] {cs} × {isrc} × {family}", flush=True)
            try:
                cell_df = _run_cell(
                    config, cs, isrc, family, data, budgets, thresholds,
                )
                all_dfs.append(cell_df)
                # Save incrementally after each cell
                partial = pd.concat(all_dfs, axis=0).reset_index(drop=True)
                partial["model_family"] = family
                partial["n"] = data.test.n
                partial["missing_fraction"] = missing_fraction
                partial["missing_mechanism"] = missing_mechanism
                partial.to_csv(out_path, index=False)
                print(f"  Saved {len(partial)} rows to {out_path}", flush=True)
            except (FileNotFoundError, NotImplementedError) as e:
                logger.warning("Skipping %s × %s: %s", cs, isrc, e)

    if not all_dfs:
        logger.warning("No cells produced results.")
        return pd.DataFrame()

    results_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    results_df["model_family"] = family
    results_df["n"] = data.test.n
    results_df["missing_fraction"] = missing_fraction
    results_df["missing_mechanism"] = missing_mechanism
    results_df.to_csv(config.get_results_path(family), index=False)
    return results_df


# ── Stage: align ─────────────────────────────────────────────────────


def align(
    config: RobotBenchmarkConfig,
    model: ConceptBasedModel | None = None,
    data=None,
) -> dict:
    """Run alignment test on the trained CBM.

    Retrains the frontend with monotonicity (sign) constraints and
    compares original vs constrained accuracy.

    Returns dict with original_accuracy, aligned_accuracy, accuracy_change,
    predictions_changed, aligned_weights.
    """
    if _selected_cbm_key(config) != "cbm":
        logger.info(
            "Alignment is only supported for cbm_family='cbm'; skipping %s.",
            _selected_cbm_key(config),
        )
        return {}
    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    return run_alignment(
        concept_based_model=model,
        train_dataset=data.train,
        test_dataset=data.test,
        monotonicity_constraints=config.get_alignment_constraints(),
        save_path=config.get_alignment_results_path(),
    )


# ── Stage: collect_results ────────────────────────────────────────────


def _dataset_label(cfg: RobotBenchmarkConfig) -> str:
    """Return a human-readable dataset label for a config."""
    return "subconcept" if cfg.concept_preset == "foot_subtypes" else "ideal"


def collect_results(
    configs: list[RobotBenchmarkConfig] | None = None,
) -> pd.DataFrame:
    """Aggregate all robot results into a single flat CSV.

    Produces one row per (dataset, model, budget) combination with columns:
      dataset, model, budget, threshold, accuracy, gain,
      predictions_intervened_on, avg_concepts_per_sample, predictions_changed

    Reads saved artifacts only — no model retraining.
    """
    import json

    if configs is None:
        configs = [RobotBenchmarkConfig.default_ideal()]

    device = determine_device()
    loader_config = get_loader_config()
    rows = []

    # ── Per-config: DNN, CBM, interventions, alignment ───────────────
    for cfg in configs:
        label = _dataset_label(cfg)
        model_key = _selected_cbm_key(cfg)

        # Load this config's dataset
        data = load(cfg.get_dataset_path())

        # DNN accuracy
        dnn_path = cfg.get_model_path("dnn")
        dnn_accuracy = None
        if dnn_path.exists():
            dnn_weights = load(dnn_path)
            dnn = RobotClassifierCNN(input_size=cfg.input_size).to(device)
            dnn.load_state_dict(dnn_weights)
            test_loader = data.test.loader(shuffle=False, **loader_config)
            dnn_accuracy = compute_accuracy(dnn, test_loader, device)
            rows.append(
                {
                    "dataset": label,
                    "model": "dnn",
                    "budget": "",
                    "threshold": "",
                    "accuracy": round(dnn_accuracy, 4),
                    "gain": 0.0,
                    "predictions_intervened_on": "",
                    "avg_concepts_per_sample": "",
                    "predictions_changed": "",
                }
            )

        # CBM no-intervention (k=0)
        cbm_path = cfg.get_model_path(model_key)
        if not cbm_path.exists():
            logger.warning(
                "%s model not found for %s, skipping: %s", model_key, label, cbm_path
            )
            continue
        cbm = load(cbm_path)
        cbm_acc = float((cbm.predict(data.test) == data.test.y).mean())
        if model_key == "ecbm":
            interpretation_summary = compute_ecbm_interpretation_summary(cbm, data.test)
            interpretation_path = cfg.get_interpretation_path(model_key)
            interpretation_path.parent.mkdir(parents=True, exist_ok=True)
            interpretation_path.write_text(
                json.dumps(interpretation_summary, indent=2, sort_keys=True)
            )
            pd.DataFrame(interpretation_summary["rows"]).to_csv(
                interpretation_path.with_suffix(".csv"),
                index=False,
            )
        gain_ref = dnn_accuracy if dnn_accuracy is not None else cbm_acc
        rows.append(
            {
                "dataset": label,
                "model": model_key,
                "budget": 0,
                "threshold": "",
                "accuracy": round(cbm_acc, 4),
                "gain": round(cbm_acc - gain_ref, 4),
                "predictions_intervened_on": "",
                "avg_concepts_per_sample": "",
                "predictions_changed": "",
            }
        )

        # CBM with interventions (k>0)
        results_path = cfg.get_results_path(model_key)
        if results_path.exists():
            interv_df = pd.read_csv(results_path)
            # Filter to baseline regime if column present
            if "regime" in interv_df.columns:
                interv_df = interv_df[interv_df["regime"] == "baseline"]
            # Use threshold=0.2 as the canonical threshold for the summary
            t02 = interv_df[(interv_df["threshold"] == 0.2) & (interv_df["budget"] > 0)]
            for _, row in t02.iterrows():
                budget = int(row["budget"])
                acc = float(row["accuracy"])
                pio = int(row["predictions_intervened_on"])
                tcc = int(row["total_concept_confirmations"])
                avg_cps = round(tcc / pio, 2) if pio > 0 else 0.0
                rows.append(
                    {
                        "dataset": label,
                        "model": model_key,
                        "budget": budget,
                        "threshold": 0.2,
                        "accuracy": round(acc, 4),
                        "gain": round(acc - gain_ref, 4),
                        "predictions_intervened_on": pio,
                        "avg_concepts_per_sample": avg_cps,
                        "predictions_changed": int(row["predictions_changed"]),
                    }
                )

        # Aligned CBM
        align_path = cfg.get_alignment_results_path()
        if model_key == "cbm" and align_path.exists():
            with open(align_path) as f:
                align_data = json.load(f)
            aligned_acc = float(align_data["aligned_accuracy"])
            rows.append(
                {
                    "dataset": label,
                    "model": "aligned_cbm",
                    "budget": 0,
                    "threshold": "",
                    "accuracy": round(aligned_acc, 4),
                    "gain": round(aligned_acc - gain_ref, 4),
                    "predictions_intervened_on": "",
                    "avg_concepts_per_sample": "",
                    "predictions_changed": "",
                }
            )

            # Aligned CBM with intervention at k=3
            aligned_weights = align_data.get("aligned_weights")
            if aligned_weights is not None:
                from experiments.alignment import align_frontend_weights
                import copy as _copy

                # Load the config's own dataset so concept shapes match
                cfg_data = load(cfg.get_dataset_path())
                aligned_fe = _copy.deepcopy(cbm.label_predictor)
                aligned_fe = align_frontend_weights(
                    aligned_fe,
                    list(cfg_data.test.concepts),
                    aligned_weights,
                )
                c_preds = cbm.concept_detector.predict_proba(cfg_data.test)
                isettings = InterventionSettings(
                    seed=cfg.seed,
                    budgets=[3],
                    intervention_accuracy=cfg.intervention_accuracy,
                    intervention_threshold=0.2,
                )
                _, _, int_results = _test_interventions(
                    prob_test=c_preds,
                    settings=isettings,
                    acc_det=aligned_acc,
                    fe=aligned_fe,
                    test=cfg_data.test,
                )
                for key, res in int_results.items():
                    pio = int(res["predictions_intervened_on"])
                    tcc = int(res["total_concept_confirmations"])
                    avg_cps = round(tcc / pio, 2) if pio > 0 else 0.0
                    rows.append(
                        {
                            "dataset": label,
                            "model": "aligned_cbm",
                            "budget": 3,
                            "threshold": 0.2,
                            "accuracy": round(float(res["accuracy"]), 4),
                            "gain": round(float(res["accuracy"]) - gain_ref, 4),
                            "predictions_intervened_on": pio,
                            "avg_concepts_per_sample": avg_cps,
                            "predictions_changed": int(res["predictions_changed"]),
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
        config.seed,
        variant,
        stages,
        device,
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
                logger.info(
                    "No cached data found — generating dataset and robot images (this may take a minute)"
                )
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

    def _model_fp_path(model_key: str) -> Path:
        return config.get_model_path(model_key).with_suffix(".fingerprint")

    def _current_model_fp(model_key: str) -> str:
        return config.model_fingerprint(model_key)

    def _should_train(model_key: str) -> bool:
        model_path = config.get_model_path(model_key)
        model_fp_path = _model_fp_path(model_key)
        current_model_fp = _current_model_fp(model_key)
        cached_model_fp = (
            model_fp_path.read_text().strip() if model_fp_path.exists() else None
        )
        model_stale = cached_model_fp != current_model_fp
        if config.force_retrain:
            return True
        if not model_path.exists():
            return True
        if model_stale:
            logger.info("Config changed since last training — retraining %s", model_key)
            return True
        return False

    def _write_model_fingerprint(model_key: str) -> None:
        model_fp_path = _model_fp_path(model_key)
        model_fp_path.parent.mkdir(parents=True, exist_ok=True)
        model_fp_path.write_text(_current_model_fp(model_key))

    if "cbm" in stages:
        logger.info("=== [%d/%d] Train CBM ===", _si["cbm"], n_stages)
        selected_key = _selected_cbm_key(config)
        if _should_train(selected_key):
            if selected_key == "cbm":
                train_cbm(
                    config,
                    missing_fraction=missing_fraction,
                    missing_mechanism=missing_mechanism,
                )
            elif selected_key in {"cem", "probcbm", "ecbm"}:
                _train_wrapped_cbm_family(
                    config, family=selected_key, save_key=selected_key
                )
            else:
                raise ValueError(f"Unsupported cbm_family: {selected_key!r}")
            _write_model_fingerprint(selected_key)
        else:
            logger.info(
                "Using existing %s model: %s",
                selected_key,
                config.get_model_path(selected_key),
            )
        # LFCBM and family-on-LFCBM models are trained lazily during
        # the intervene stage by _prepare_model_for_concept_source().

    if "dnn" in stages:
        logger.info("=== [%d/%d] Train DNN ===", _si["dnn"], n_stages)
        if _should_train("dnn"):
            train_dnn(config)
            _write_model_fingerprint("dnn")
        else:
            logger.info("Using existing DNN: %s", config.get_model_path("dnn"))

    if "intervene" in stages:
        logger.info("=== [%d/%d] Intervene ===", _si["intervene"], n_stages)
        run_interventions(
            config,
            missing_fraction=missing_fraction,
            missing_mechanism=missing_mechanism,
        )

    if "align" in stages:
        logger.info("=== [%d/%d] Align ===", _si["align"], n_stages)
        align(config)

    if "collect" in stages:
        logger.info("=== [%d/%d] Collect ===", _si["collect"], n_stages)
        collect_results([config])

    if "plot" in stages:
        logger.info("=== [%d/%d] Plot ===", _si.get("plot", n_stages), n_stages)
        plot_results(config)


def plot_results(config: RobotBenchmarkConfig) -> None:
    """Generate figures from collected results and save to results/figures/."""
    import json
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from concept_benchmark.evaluation.plots import (
        plot_alignment_comparison,
        plot_concept_discovery,
        plot_intervention_curve,
        plot_regime_comparison,
    )

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    variant = "subconcept" if config.concept_preset == "foot_subtypes" else "ideal"
    family = _selected_cbm_key(config)
    results_path = config.get_results_path(family)
    if not results_path.exists():
        logger.info("No results CSV found at %s — skipping plots.", results_path)
        return

    interv_df = pd.read_csv(results_path)

    # 1. Intervention curve (always)
    if "regime" in interv_df.columns:
        baseline = interv_df[
            (interv_df["regime"] == "baseline") & (interv_df["threshold"] == 0.2)
        ]
    else:
        baseline = interv_df[interv_df["threshold"] == 0.2]
    if len(baseline) > 0:
        fig, _ = plot_intervention_curve(baseline)
        fname = f"robot_{variant}_{family}_intervention_curve.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)

    # 2. Regime comparison (if multiple regimes)
    if "regime" in interv_df.columns and interv_df["regime"].nunique() > 1:
        regime_df = interv_df[interv_df["threshold"] == 0.2]
        fig, _ = plot_regime_comparison(regime_df)
        fig.savefig(
            out_dir / "robot_regime_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        logger.info("Saved robot_regime_comparison.png")

    # 3. Concept discovery (if both ideal and subconcept results exist)
    other_suffix = "_subconcept" if variant == "ideal" else "_ideal"
    this_suffix = f"_{variant}"
    other_path = Path(str(results_path).replace(this_suffix, other_suffix))
    if other_path.exists():
        other_df = pd.read_csv(other_path)

        # Get baseline (regime=baseline or no regime column) at threshold=0.2
        def _extract_baseline(df):
            if "regime" in df.columns:
                return df[(df["regime"] == "baseline") & (df["threshold"] == 0.2)]
            return df[df["threshold"] == 0.2]

        this_bl = _extract_baseline(interv_df)
        other_bl = _extract_baseline(other_df)

        if variant == "ideal":
            ideal_bl, sub_bl = this_bl, other_bl
        else:
            ideal_bl, sub_bl = other_bl, this_bl

        if len(ideal_bl) > 0 and len(sub_bl) > 0:
            # Get DNN accuracy from collect CSV if available
            dnn_acc = None
            collect_path = config.get_collect_path()
            if collect_path.exists():
                cdf = pd.read_csv(collect_path)
                dnn_rows = cdf[cdf["model"] == "dnn"]
                if len(dnn_rows) > 0:
                    dnn_acc = float(dnn_rows["accuracy"].values[0])
            fig, _ = plot_concept_discovery(
                ideal_bl, sub_bl, dnn_accuracy=dnn_acc or 0.8746
            )
            fname = f"robot_{family}_concept_discovery.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved %s", fname)

    # 4. Alignment comparison (if alignment JSONs exist for both variants)
    align_path = results_path.with_name(
        results_path.name.replace("_cbm_results.csv", "_alignment.json")
    )
    other_align = Path(str(align_path).replace(this_suffix, other_suffix))
    if align_path.exists() and other_align.exists():
        this_align = json.loads(align_path.read_text())
        that_align = json.loads(other_align.read_text())
        # Compute gains relative to DNN
        dnn_acc = dnn_acc if "dnn_acc" in dir() and dnn_acc else 0.8746
        if variant == "ideal":
            results_dict = {
                "ideal": {
                    "cbm_gain": this_align["original_accuracy"] - dnn_acc,
                    "aligned_gain": this_align["aligned_accuracy"] - dnn_acc,
                },
                "subconcept": {
                    "cbm_gain": that_align["original_accuracy"] - dnn_acc,
                    "aligned_gain": that_align["aligned_accuracy"] - dnn_acc,
                },
            }
        else:
            results_dict = {
                "ideal": {
                    "cbm_gain": that_align["original_accuracy"] - dnn_acc,
                    "aligned_gain": that_align["aligned_accuracy"] - dnn_acc,
                },
                "subconcept": {
                    "cbm_gain": this_align["original_accuracy"] - dnn_acc,
                    "aligned_gain": this_align["aligned_accuracy"] - dnn_acc,
                },
            }
        fig, _ = plot_alignment_comparison(results_dict)
        fig.savefig(out_dir / "robot_alignment.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved robot_alignment.png")


# ── CLI entry point ──────────────────────────────────────────────────

ROBOT_STAGES = ("setup", "cbm", "dnn", "intervene", "align", "collect", "plot")


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the robot classification benchmark pipeline.",
    )
    parser.add_argument("--seed", type=int, default=1014)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(ROBOT_STAGES),
        help=f"Pipeline stages to run (default: all). Valid: {' -> '.join(ROBOT_STAGES)}",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--concept-preset",
        choices=["ground_truth", "foot_subtypes"],
        default="ground_truth",
    )
    parser.add_argument(
        "--cbm-family",
        choices=["cbm", "cem", "probcbm", "ecbm"],
        default=None,
        help="Concept-model family to train/evaluate (default: config or cbm).",
    )
    parser.add_argument(
        "--missing-fraction",
        type=float,
        default=None,
        help="Fraction of concept labels to mask (e.g. 0.2).",
    )
    parser.add_argument(
        "--missing-mechanism", type=str, default=None, choices=["mcar", "mnar"]
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help="Intervention budgets (e.g. 1 3 5 max).",
    )
    parser.add_argument(
        "--concept-sources",
        nargs="+",
        default=None,
        choices=CONCEPT_SOURCES,
        help="Concept sources (e.g. human_concepts machine_annotation llm_concepts clip_concepts).",
    )
    parser.add_argument(
        "--intervention-sources",
        nargs="+",
        default=None,
        choices=INTERVENTION_SOURCES,
        help="Intervention sources (e.g. perfect expert llm).",
    )
    parser.add_argument(
        "--strategy", type=str, default=None, choices=["up_to_k", "exactly_k"]
    )
    parser.add_argument("--llm-provider", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-reasoning-effort", type=str, default=None)
    parser.add_argument("--llm-cache-only", action="store_true")
    parser.add_argument("--llm-cache-all-concepts", action="store_true")
    parser.add_argument("--llm-workers", type=int, default=None)
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--llm-batch-sleep", type=float, default=None)
    parser.add_argument(
        "--llm-api-key-env",
        type=str,
        default=None,
        help="Environment variable name that stores the LLM API key.",
    )
    parser.add_argument(
        "--llm-concepts-file",
        type=str,
        default=None,
        help="Concept descriptions JSONL for the llm automated regime.",
    )
    parser.add_argument(
        "--clip-concepts-file",
        type=str,
        default=None,
        help="Concept descriptions JSONL for the clip automated regime.",
    )
    parser.add_argument(
        "--placeholder3-concepts-file",
        type=str,
        default=None,
        help="Concept descriptions JSONL for the placeholder3 automated regime.",
    )
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--force-retrain", action="store_true", dest="force_retrain")
    parser.add_argument("--force-setup", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    unknown = set(args.stages) - set(ROBOT_STAGES)
    if unknown:
        raise ValueError(
            f"unknown stages: {sorted(unknown)}. Valid: {list(ROBOT_STAGES)}"
        )

    if args.config:
        config = RobotBenchmarkConfig.from_yaml(args.config)
    elif args.concept_preset == "foot_subtypes":
        config = RobotBenchmarkConfig.default_subconcept()
        config.seed = args.seed
    else:
        config = RobotBenchmarkConfig(seed=args.seed)

    # Paper data was generated with rng_seed=12345 (the old hardcoded default
    # in create_robot_image_dataset).  Pin it here so regeneration reproduces
    # the exact same stochastic labels.
    config.rng_seed = 12345

    if args.budgets:
        config.intervention_budgets = parse_budgets(args.budgets)
    if args.cbm_family:
        config.cbm_family = args.cbm_family
    if args.concept_sources:
        config.concept_sources = args.concept_sources
    if args.intervention_sources:
        config.intervention_sources = args.intervention_sources
    if args.strategy:
        config.intervention_strategy = args.strategy
    if args.llm_provider:
        config.llm_provider = args.llm_provider
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.llm_reasoning_effort:
        config.llm_reasoning_effort = args.llm_reasoning_effort
    if args.llm_cache_only:
        config.llm_cache_only = True
    if args.llm_cache_all_concepts:
        config.llm_cache_all_concepts = True
    if args.llm_workers is not None:
        config.llm_workers = args.llm_workers
    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    if args.llm_batch_sleep is not None:
        config.llm_batch_sleep = args.llm_batch_sleep
    if args.llm_api_key_env:
        config.llm_api_key_env = args.llm_api_key_env
    if args.llm_concepts_file:
        config.llm_concepts_file = args.llm_concepts_file
    if args.clip_concepts_file:
        config.clip_concepts_file = args.clip_concepts_file
    if args.placeholder3_concepts_file:
        config.placeholder3_concepts_file = args.placeholder3_concepts_file
    if args.llm_api_key:
        config.llm_api_key = args.llm_api_key
    if args.force_retrain:
        config.force_retrain = True
    missing_fraction = args.missing_fraction or 0.0
    missing_mechanism = args.missing_mechanism or "mcar"

    run(
        config,
        stages=args.stages,
        force_setup=args.force_setup,
        missing_fraction=missing_fraction,
        missing_mechanism=missing_mechanism,
    )


if __name__ == "__main__":
    main()
