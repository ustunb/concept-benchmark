"""Utility helpers for concept dataset masking and reproducibility."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Return a numpy Generator from either an int seed, Generator, or RandomState."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        seed = rng.randint(low=0, high=2**32 - 1)
        return np.random.default_rng(seed)
    raise TypeError("rng must be None, an int seed, np.random.Generator, or RandomState")


def broadcast_prob(value, n_concepts: int, *, default: float) -> np.ndarray:
    """Coerce scalar/per-concept probabilities to a 1D array of length n_concepts."""
    if value is None:
        value = default
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full((n_concepts,), float(arr), dtype=float)
    elif arr.shape != (n_concepts,):
        raise ValueError("Probability specification must be a scalar or length-n_concepts array")
    return np.clip(arr, 0.0, 1.0)


def sample_mcar_mask(rng: np.random.Generator, shape: tuple[int, ...], p: float) -> np.ndarray:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1 inclusive")
    return rng.random(shape) < p


def sample_mnar_mask(
    rng: np.random.Generator,
    concepts: np.ndarray,
    *,
    base_p: float,
    config: Mapping[str, object] | None,
) -> np.ndarray:
    if not 0.0 <= base_p <= 1.0:
        raise ValueError("base_p must be between 0 and 1 inclusive")

    config = dict(config or {})
    n_concepts = concepts.shape[1]

    prob_matrix = config.get("prob_matrix")
    if prob_matrix is not None:
        prob_matrix = np.asarray(prob_matrix, dtype=float)
        if prob_matrix.shape != concepts.shape:
            raise ValueError("Provided prob_matrix must match concepts shape")
    else:
        present_default = min(base_p * 1.5, 1.0)
        absent_default = max(base_p * 0.5, 0.0)
        present_prob = broadcast_prob(
            config.get("present_prob"), n_concepts, default=present_default
        )
        absent_prob = broadcast_prob(
            config.get("absent_prob"), n_concepts, default=absent_default
        )

        concepts_bool = concepts.astype(bool)
        prob_matrix = np.where(concepts_bool, present_prob, absent_prob)

    prob_matrix = np.clip(prob_matrix, 0.0, 1.0)

    return rng.random(concepts.shape) < prob_matrix


def sample_concept_noise_mask(
    rng: np.random.Generator,
    concepts: np.ndarray,
    *,
    base_p: float,
    config: Mapping[str, object] | None,
) -> np.ndarray:
    config = dict(config or {})
    n_concepts = concepts.shape[1]

    prob_matrix = config.get("prob_matrix")
    if prob_matrix is not None:
        prob_matrix = np.asarray(prob_matrix, dtype=float)
        if prob_matrix.shape != concepts.shape:
            raise ValueError("Provided prob_matrix must match concepts shape")
    else:
        flip_prob = config.get("flip_prob")
        prob_01 = config.get("p01")
        prob_10 = config.get("p10")

        if prob_01 is None and prob_10 is None:
            flip_prob = broadcast_prob(flip_prob, n_concepts, default=base_p)
            prob_matrix = np.broadcast_to(flip_prob, concepts.shape)
        else:
            p01 = broadcast_prob(prob_01, n_concepts, default=base_p)
            p10 = broadcast_prob(prob_10, n_concepts, default=base_p)
            concepts_bool = concepts.astype(bool)
            prob_matrix = np.where(concepts_bool, p10, p01)

    prob_matrix = np.clip(prob_matrix, 0.0, 1.0)

    return rng.random(concepts.shape) < prob_matrix


def sample_label_noise(
    rng: np.random.Generator,
    labels: np.ndarray,
    *,
    num_classes: int,
    base_p: float,
    config: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Return a noisy copy of ``labels`` according to class-dependent probabilities."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be integers")
    if (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError("labels must be within [0, num_classes)")

    config = dict(config or {})

    flip_matrix = config.get("flip_matrix")
    if flip_matrix is not None:
        matrix = np.asarray(flip_matrix, dtype=float)
        if matrix.shape != (num_classes, num_classes):
            raise ValueError("flip_matrix must have shape (num_classes, num_classes)")
        row_sums = matrix.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("flip_matrix rows must sum to a positive value")
        matrix = matrix / row_sums[:, None]
        cdf = np.cumsum(matrix, axis=1)
        rand = rng.random(labels.size)
        rand = np.minimum(rand, 1.0 - np.finfo(float).eps)
        rows = cdf[labels]
        new_labels = np.searchsorted(rows, rand, side="right")
        return new_labels.astype(labels.dtype, copy=False)

    if not 0.0 <= base_p <= 1.0:
        raise ValueError("base_p must be between 0 and 1 inclusive")

    new_labels = labels.copy()
    if num_classes <= 1 or base_p == 0.0:
        return new_labels

    flip_mask = rng.random(labels.size) < base_p
    if flip_mask.any():
        base_vals = labels[flip_mask]
        choices = rng.integers(0, num_classes - 1, size=flip_mask.sum())
        choices = choices + (choices >= base_vals)
        new_labels[flip_mask] = choices.astype(labels.dtype, copy=False)
    return new_labels
