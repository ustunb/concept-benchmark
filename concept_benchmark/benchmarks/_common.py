"""Shared utilities used by all benchmark pipelines.

Consolidates code that was duplicated across scripts/robot_demo/utils.py,
scripts/sudoku_demo/utils.py, and scripts/utils/dataset_skewing.py.
"""
from __future__ import annotations

import os
import platform

import numpy as np
import torch
import torch.nn as nn


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


def get_loader_config(device: torch.device | None = None) -> dict:
    """Return DataLoader kwargs safe for the current platform."""
    _macos = platform.system() == "Darwin"
    return {
        "batch_size": 32,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }


def patch_macos_dataloader() -> None:
    """Force num_workers=0 on macOS to avoid MPS/fork hangs."""
    if platform.system() != "Darwin":
        return

    import torch.utils.data as _tud
    import concept_benchmark.data as _cb_data

    _OrigDataLoader = _tud.DataLoader

    def _safe_dataloader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return _OrigDataLoader(*args, **kwargs)

    _tud.DataLoader = _safe_dataloader
    _cb_data.DataLoader = _safe_dataloader


# ── Dataset skewing ──────────────────────────────────────────────────

def _create_sample(size, indices, dataset):
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return dataset._full.filter(mask)


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

    print(
        f"Final splits - Train: {len(train_indices)}, "
        f"Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    dataset.drop_concepts(drop_concepts)
    dataset.training = _create_sample(total_size, train_indices, dataset)
    dataset.validation = _create_sample(total_size, val_indices, dataset)
    dataset.test = _create_sample(total_size, test_indices, dataset)

    return dataset


def _create_skewed_training_set(dataset, skew_specs, available_indices, target_size, rng):
    """Create training set that satisfies skewing requirements."""
    available_set = set(available_indices)
    train_indices = []
    used = set()

    for spec in skew_specs:
        mask = np.ones(len(dataset.C), dtype=bool)
        for concept_name, target_value in spec["concepts"].items():
            concept_idx = dataset.concepts.index(concept_name)
            mask &= dataset.C[:, concept_idx] == target_value

        spec_indices = [i for i in np.where(mask)[0] if i in available_set and i not in used]
        needed = int(target_size * spec["min_fraction"])

        rng.shuffle(spec_indices)
        take = spec_indices[: min(needed, len(spec_indices))]
        train_indices.extend(take)
        used.update(take)

        print(
            f"Skew spec {spec['concepts']}: needed {needed}, "
            f"got {len(take)} (max available {len(spec_indices)})"
        )

    remaining_slots = target_size - len(train_indices)
    if remaining_slots > 0:
        unused = [i for i in available_indices if i not in used]
        rng.shuffle(unused)
        train_indices.extend(unused[:remaining_slots])

    return np.array(train_indices)
