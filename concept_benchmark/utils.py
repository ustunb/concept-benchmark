"""Shared utilities for dataset generation (pip-installable package)."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int):
    """Full reproducibility: numpy, torch, random, PYTHONHASHSEED."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
