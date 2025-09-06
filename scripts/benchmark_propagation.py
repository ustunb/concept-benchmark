#!/usr/bin/env python3
"""
Quick benchmark comparing naive enumeration, vectorized exact propagation,
and Monte Carlo propagation on a synthetic tabular dataset.

Usage:
  uv run python scripts/benchmark_propagation.py  # or `python ...` in your env
"""

import time
import numpy as np

from concept_benchmark.data import ConceptDataset
from concept_benchmark.models import ConceptBasedModel, ConceptDetector


def make_tabular_dataset(n: int, d: int, k: int, n_classes: int, *, density: float = 0.3):
    X = np.random.normal(size=(n, d)).astype(np.float32)
    C = (np.random.rand(n, k) < density).astype(np.int8)
    # balanced-ish labels
    y = np.arange(n_classes, dtype=np.int32).repeat(int(np.ceil(n / n_classes)))[:n]
    rng = np.random.default_rng(202)
    rng.shuffle(y)
    meta = {
        "classes": [f"class_{i}" for i in range(n_classes)],
        "concepts": [f"z{i}" for i in range(k)],
        "data_type": "tabular",
    }
    # Simple 2-fold CV to obtain training/validation samples through ConceptDataset API
    folds = (np.arange(n) % 2) + 1
    cv = {"K02N01": folds.astype(np.int32)}
    ds = ConceptDataset(X=X, C=C, y=y, meta=meta, cvindices=cv)
    ds.split("K02N01", fold_num_validation=2, fold_num_test=None)
    return ds.training, ds.validation


def aggregate_naive(model: ConceptBasedModel, dataset) -> np.ndarray:
    """Explicit enumeration aggregation using precomputed combos and label probs.

    Computes concept probabilities internally for fairness, then aggregates
    over all concept combinations using cached exact tables.

    Args:
        model: Trained ConceptBasedModel.
        dataset: ConceptDatasetSample to predict on.
    Returns: (N, K) aggregated label probabilities.
    """
    # Ensure caches exist (exact path preparation)
    if model._concept_poss is None or model._y_proba_all_concepts is None:
        model._prep_propagation()

    combs = model._concept_poss.astype(np.float64)  # (M, C)
    # Stack dict values in comb order into (M, K)
    y_mat = np.vstack([
        np.asarray(model._y_proba_all_concepts[tuple(c)]).reshape(1, -1)
        for c in combs
    ])

    # Compute concept probabilities inside for fair timing
    P = np.asarray(model.concept_detector.predict(dataset, calibrate=False), dtype=np.float64)
    P = np.clip(P, 1e-12, 1.0 - 1e-12)
    N = P.shape[0]
    out = np.zeros((N, y_mat.shape[1]), dtype=np.float64)

    # Per-sample aggregation; weights vectorized over all combos
    for i in range(N):
        p = P[i]
        w = np.prod((p ** combs) * ((1.0 - p) ** (1.0 - combs)), axis=1)  # (M,)
        out[i] = w @ y_mat

    return out


def main():
    np.random.seed(1337)

    # Problem sizes
    n_train, n_valid = 256, 256
    d, k, n_classes = 8, 10, 3  # 2^k = 1024 (feasible for exact)

    train, valid = make_tabular_dataset(n_train + n_valid, d, k, n_classes)

    model = ConceptBasedModel(
        concept_detector=ConceptDetector(embedding_model=None),
        propagate=True,
        mc_mode="exact",
        mc_exact_threshold=1 << 14,  # 16384
    )

    # Train quickly
    t0 = time.perf_counter()
    model.fit(
        train_dataset=train,
        valid_dataset=valid,
        freeze=False,
        concept_fit_params={"epochs": 1, "device": "cpu", "batch_size": 32},
        front_fit_params={"max_iter": 200},
        calibrate=False,
    )
    t1 = time.perf_counter()

    # Naive
    t2 = time.perf_counter()
    proba_naive = aggregate_naive(model, valid)
    t3 = time.perf_counter()

    # Vectorized exact
    model._mc_mode = "exact"
    t4 = time.perf_counter()
    proba_exact = model.predict_proba(valid, propagate=True)
    t5 = time.perf_counter()

    # Monte Carlo (deterministic for fair compare)
    model._mc_mode = "mc"
    model._random_state = 123
    model._mc_samples = 4096
    model._mc_max_samples = 4096
    model._mc_chunk_size = 1024
    model._mc_tol = 0.0
    t6 = time.perf_counter()
    proba_mc = model.predict_proba(valid, propagate=True)
    t7 = time.perf_counter()

    # Timings
    fit_s = t1 - t0
    naive_s = t3 - t2
    vector_s = t5 - t4
    mc_s = t7 - t6

    # Accuracy deltas
    max_abs_ne = float(np.max(np.abs(proba_naive - proba_exact)))
    max_abs_mc = float(np.max(np.abs(proba_mc - proba_exact)))

    print("=== Propagation Benchmark ===")
    print(f"n_train={n_train} n_valid={n_valid} d={d} k={k} classes={n_classes}")
    print(f"fit time:              {fit_s:8.4f}s")
    print(f"naive enumeration:     {naive_s:8.4f}s  (baseline)")
    print(f"vectorized exact:      {vector_s:8.4f}s  (speedup x{naive_s/max(1e-9, vector_s):.1f})")
    print(f"monte carlo (4096):    {mc_s:8.4f}s")
    print(f"max |naive-exact|:     {max_abs_ne:.3e}")
    print(f"max |mc-exact|:        {max_abs_mc:.3e}")


if __name__ == "__main__":
    main()
