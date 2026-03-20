"""Robot pipeline — model training functions."""
from __future__ import annotations

import copy
import logging
import platform

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from concept_benchmark.utils import (
    compute_accuracy,
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
    set_deterministic_seed,
)
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.generators import DatasetGenerator
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.paths import data_dir
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
    FrontEndModel,
    RobotClassifierCNN,
    RobotConceptClassifier,
)

logger = logging.getLogger(__name__)


class FEOnProbs(FrontEndModel):
    """Wrap an LFCBM sklearn classifier so it works as a FrontEndModel.

    Applies logit transform before predict_proba, matching how the
    LFCBM classifier was trained (on z-scored, then logit-transformed features).
    """
    _kflip_fast_path = False  # class-level: fast path skips our logit transform

    def __init__(self, clf):
        super().__init__()
        self.model = clf

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        P = np.clip(P, 1e-6, 1 - 1e-6)
        Z = np.log(P / (1.0 - P))
        return self.model.predict_proba(Z)


# ── Concept noise helpers ─────────────────────────────────────────────

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
        idxs = [i for i, c in enumerate(concept_names)
                if c == base or c.startswith(f"{base}_")]
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


def _clone_sample_with_C(sample, C_new):
    """Clone a dataset sample with replaced concept matrix."""
    return sample.__class__(
        parent=sample.parent,
        X=sample.X,
        C=C_new.astype(np.float32),
        y=sample.y,
        meta=sample.meta,
        transform=sample.transform,
        concept_transform=sample.concept_transform,
        target_transform=sample.target_transform,
        base_dir=getattr(sample, "base_dir", None),
    )


# ── Training functions ────────────────────────────────────────────────

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
    data.drop_concepts(PRESET_EXCLUDED_CONCEPTS[config.concept_preset])

    # Only apply sampling constraints when the referenced concepts exist
    # (constraints reference foot_shape subconcepts, only present in foot_subtypes preset)
    constraints = [
        c for c in ROBOT_SAMPLING_CONSTRAINTS
        if all(name in data.concepts for name in c["concepts"])
    ]

    # Absolute test size: paper requires fixed 10k test set for comparable results
    test_size = 10000
    train_size = 3800
    remaining = data.n - test_size
    n_val = int((remaining - train_size) * 0.2)
    data.sample(
        test_size=test_size,
        val_size=n_val,
        train_size=train_size,
        sampling_constraints=constraints or None,
        seed=config.seed,
    )

    save(data, config.get_dataset_path(), overwrite=True)
    return data


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
        data.training.has_concept_missing = True

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
    rng = np.random.default_rng(config.seed + 555)
    concept_names = list(noisy_data.training.concepts)
    concept_spec = config.concepts

    # Apply group-level noise to training and validation splits
    noisy_data.training = _clone_sample_with_C(
        noisy_data.training,
        _apply_concept_noise_grouped(
            noisy_data.training.C, concept_names, concept_spec,
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

    cbm = train_cbm(config, data=noisy_data, save_key=None)
    save(cbm, config.get_model_path("cbm_subjective"), overwrite=True)
    return cbm


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
    train_paths = [str(image_dir / p) for p in data.training.X]
    valid_paths = [str(image_dir / p) for p in data.validation.X]

    stats = lfcbm.fit(
        train_X=train_paths,
        train_y=data.training.y.astype(int),
        valid_X=valid_paths,
        valid_y=data.validation.y.astype(int),
        concept_set=concept_set,
    )
    logger.info("LFCBM stats: %s/%s concepts kept", stats.get("kept_concepts"), stats.get("total_concepts"))

    out_dir = str(config.get_model_path("lfcbm")) + "_bundle"
    lfcbm.save(out_dir)
    bundle = {"lfcbm": lfcbm, "frontend": FEOnProbs(lfcbm.classifier)}
    save(bundle, config.get_model_path("lfcbm"), overwrite=True)
    return stats


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
                logger.info(
                    "Early stopping at epoch %d with best val loss %.6f",
                    epoch + 1, best_val_loss,
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
