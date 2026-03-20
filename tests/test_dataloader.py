"""Tests for DataLoader correctness."""
from __future__ import annotations

import platform

import numpy as np
import pytest

from concept_benchmark.data import ConceptDataset


def _make_small_dataset(n=30, k=4, seed=42):
    """Create a small tabular dataset for loader tests."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 8)).astype(np.float32)
    C = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y = np.tile([0, 1], n // 2 + 1)[:n].astype(np.int32)
    meta = {
        "classes": ["c0", "c1"],
        "concepts": [f"z{i}" for i in range(k)],
        "data_type": "tabular",
    }
    return ConceptDataset(X=X, C=C, y=y, meta=meta)


def _collect(loader):
    xs, cs, ys = [], [], []
    for X, C, y in loader:
        xs.append(X.numpy())
        cs.append(C.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(cs), np.concatenate(ys)


def test_loader_deterministic():
    """Two sequential loaders with shuffle=False produce identical batches."""
    ds = _make_small_dataset()
    ds.sample(test_size=0.2, val_size=0.2, seed=42)

    loader1 = ds.training.loader(
        shuffle=False, batch_size=8, num_workers=0, pin_memory=False
    )
    loader2 = ds.training.loader(
        shuffle=False, batch_size=8, num_workers=0, pin_memory=False
    )

    X1, C1, y1 = _collect(loader1)
    X2, C2, y2 = _collect(loader2)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(C1, C2)
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="macOS forces num_workers=0; parallel test not applicable",
)
def test_parallel_loader_matches_sequential():
    """Verify num_workers=2 produces the same data as num_workers=0."""
    ds = _make_small_dataset()
    ds.sample(test_size=0.2, val_size=0.2, seed=42)

    loader_seq = ds.training.loader(
        shuffle=False, batch_size=8, num_workers=0, pin_memory=False
    )
    loader_par = ds.training.loader(
        shuffle=False, batch_size=8, num_workers=2, pin_memory=False
    )

    X0, C0, y0 = _collect(loader_seq)
    X2, C2, y2 = _collect(loader_par)

    np.testing.assert_array_equal(X0, X2)
    np.testing.assert_array_equal(C0, C2)
    np.testing.assert_array_equal(y0, y2)
