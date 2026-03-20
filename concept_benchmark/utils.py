"""Shared utilities used by all benchmark pipelines."""

from __future__ import annotations

import logging
import os
import platform
import random

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int):
    """Full reproducibility: numpy, torch, random, PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def determine_device() -> torch.device:
    """Determine the best available compute device.

    Respects ``PYTORCH_DEVICE`` env var to override auto-detection.
    """
    override = os.environ.get("PYTORCH_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_accuracy(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Compute binary classification accuracy (threshold 0.5)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, _, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs.squeeze() > 0.5).long()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0


def get_loader_config() -> dict:
    """Return DataLoader kwargs safe for the current platform."""
    _macos = platform.system() == "Darwin"
    return {
        "batch_size": 32,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }


def patch_macos_dataloader() -> None:
    """Force num_workers=0 on macOS to avoid MPS/fork hangs.

    Uses a real subclass so that third-party libraries (e.g. timm) can
    subclass ``torch.utils.data.DataLoader`` after the patch.
    """
    if platform.system() != "Darwin":
        return

    import torch.utils.data as _tud
    import concept_benchmark.data as _cb_data

    _OrigDataLoader = _tud.DataLoader

    # Already patched
    if getattr(_OrigDataLoader, "_macos_patched", False):
        return

    class _SafeDataLoader(_OrigDataLoader):
        _macos_patched = True

        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            super().__init__(*args, **kwargs)

    _tud.DataLoader = _SafeDataLoader
    _cb_data.DataLoader = _SafeDataLoader


# ── Dataset skewing ──────────────────────────────────────────────────


def _create_sample(size, indices, dataset):
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return dataset._full.filter(mask)


def parse_budgets(raw: list[str]) -> list[int]:
    """Parse CLI budget strings (e.g. ``['1', '3', 'max']``) to ints (-1 for max)."""
    budgets = []
    for v in raw:
        budgets.append(-1 if v.lower() == "max" else int(v))
    return budgets


def create_skewed_splits_full(
    dataset,
    skew_specs,
    test_size=10000,
    train_skew_size=None,
    val_fraction=0.2,
    rng=None,
    drop_concepts=None,
    **kwargs,
):
    """Skew training set by ensuring minimum representation of concept patterns.

    Returns the full dataset with training/validation/test splits set.
    """
    if drop_concepts is None:
        drop_concepts = []
    if rng is None:
        rng = np.random.default_rng()

    total_size = len(dataset.C)
    all_indices = np.arange(total_size)
    rng.shuffle(all_indices)
    test_indices = all_indices[:test_size]
    remaining_indices = all_indices[test_size:]

    if train_skew_size is None:
        train_skew_size = int(len(remaining_indices) * (1 - val_fraction))
        val_size = len(remaining_indices) - train_skew_size
    else:
        val_size = int((len(remaining_indices) - train_skew_size) * val_fraction)

    train_indices = _create_skewed_training_set(
        dataset, skew_specs, remaining_indices, train_skew_size, rng
    )

    used_for_training = set(train_indices)
    val_candidates = [i for i in remaining_indices if i not in used_for_training]
    rng.shuffle(val_candidates)
    val_indices = np.array(val_candidates)[:val_size]

    logger.info(
        "Final splits - Train: %d, Val: %d, Test: %d",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    dataset.drop_concepts(drop_concepts)
    dataset.training = _create_sample(total_size, train_indices, dataset)
    dataset.validation = _create_sample(total_size, val_indices, dataset)
    dataset.test = _create_sample(total_size, test_indices, dataset)
    dataset._apply_noise_settings()

    return dataset


def _create_skewed_training_set(
    dataset, skew_specs, available_indices, target_size, rng
):
    """Create training set that satisfies skewing requirements."""
    available_set = set(available_indices)
    train_indices = []
    used = set()

    for spec in skew_specs:
        mask = np.ones(len(dataset.C), dtype=bool)
        for concept_name, target_value in spec["concepts"].items():
            concept_idx = dataset.concepts.index(concept_name)
            mask &= dataset.C[:, concept_idx] == target_value

        spec_indices = [
            i for i in np.where(mask)[0] if i in available_set and i not in used
        ]
        needed = int(target_size * spec["min_fraction"])

        rng.shuffle(spec_indices)
        take = spec_indices[: min(needed, len(spec_indices))]
        train_indices.extend(take)
        used.update(take)

        logger.info(
            "Skew spec %s: needed %d, got %d (max available %d)",
            spec["concepts"],
            needed,
            len(take),
            len(spec_indices),
        )

    remaining_slots = target_size - len(train_indices)
    if remaining_slots > 0:
        unused = [i for i in available_indices if i not in used]
        rng.shuffle(unused)
        train_indices.extend(unused[:remaining_slots])

    return np.array(train_indices)
