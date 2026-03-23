"""Tests for dataset transform generators.

Each test verifies that the transform returns a new dataset and
does not modify the original.
"""

import numpy as np

from concept_benchmark.data import ConceptDataset
from concept_benchmark.transforms import (
    ConceptDropGenerator,
    ConceptMissingnessGenerator,
    ConceptNoiseGenerator,
    LabelNoiseGenerator,
)


def _make_dataset(n=60, k=4, seed=42):
    """Create a small tabular dataset with splits."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 8)).astype(np.float32)
    C = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    y = np.tile([0, 1], n // 2 + 1)[:n].astype(np.int32)
    rng.shuffle(y)
    concepts = [f"c{i}" for i in range(k)]
    meta = {
        "classes": ["a", "b"],
        "concepts": concepts,
        "data_type": "tabular",
    }
    ds = ConceptDataset(X, C, y, meta)
    ds.sample(test_size=0.2, val_size=0.2, seed=seed)
    return ds


# ── ConceptDropGenerator ────────────────────────────────────────────


class TestConceptDropGenerator:
    def test_returns_new_dataset(self):
        ds = _make_dataset()
        result = ConceptDropGenerator(ds, ["c0"]).generate()
        assert result is not ds

    def test_original_unchanged(self):
        ds = _make_dataset()
        original_n_concepts = ds.n_concepts
        ConceptDropGenerator(ds, ["c0"]).generate()
        assert ds.n_concepts == original_n_concepts

    def test_concepts_removed(self):
        ds = _make_dataset()
        result = ConceptDropGenerator(ds, ["c0", "c1"]).generate()
        assert result.n_concepts == ds.n_concepts - 2
        assert "c0" not in result.concepts
        assert "c1" not in result.concepts

    def test_splits_preserved(self):
        ds = _make_dataset()
        result = ConceptDropGenerator(ds, ["c0"]).generate()
        assert result.train.n == ds.train.n
        assert result.test.n == ds.test.n


# ── ConceptNoiseGenerator ───────────────────────────────────────────


class TestConceptNoiseGenerator:
    def test_returns_new_dataset(self):
        ds = _make_dataset()
        result = ConceptNoiseGenerator(ds, p=0.5, seed=0).generate()
        assert result is not ds

    def test_original_unchanged(self):
        ds = _make_dataset()
        original_C = ds.train.C.copy()
        ConceptNoiseGenerator(ds, p=1.0, seed=0).generate()
        np.testing.assert_array_equal(ds.train.C, original_C)

    def test_noise_applied(self):
        ds = _make_dataset()
        result = ConceptNoiseGenerator(ds, p=1.0, seed=0).generate()
        # With p=1.0, every bit should flip
        assert not np.array_equal(result.train.C, ds.train.C)

    def test_reproducible(self):
        ds = _make_dataset()
        r1 = ConceptNoiseGenerator(ds, p=0.5, seed=42).generate()
        r2 = ConceptNoiseGenerator(ds, p=0.5, seed=42).generate()
        np.testing.assert_array_equal(r1.train.C, r2.train.C)

    def test_zero_noise_unchanged(self):
        ds = _make_dataset()
        result = ConceptNoiseGenerator(ds, p=0.0, seed=0).generate()
        np.testing.assert_array_equal(result.train.C, ds.train.C)


# ── ConceptMissingnessGenerator ─────────────────────────────────────


class TestConceptMissingnessGenerator:
    def test_returns_new_dataset(self):
        ds = _make_dataset()
        result = ConceptMissingnessGenerator(ds, p=0.3, seed=0).generate()
        assert result is not ds

    def test_original_unchanged(self):
        ds = _make_dataset()
        original_C = ds.train.C.copy()
        ConceptMissingnessGenerator(ds, p=0.5, seed=0).generate()
        np.testing.assert_array_equal(ds.train.C, original_C)

    def test_missingness_applied(self):
        ds = _make_dataset()
        result = ConceptMissingnessGenerator(
            ds, p=0.5, seed=0, fill_value=-1.0
        ).generate()
        assert np.any(result.train.C == -1.0)

    def test_reproducible(self):
        ds = _make_dataset()
        r1 = ConceptMissingnessGenerator(ds, p=0.3, seed=42).generate()
        r2 = ConceptMissingnessGenerator(ds, p=0.3, seed=42).generate()
        np.testing.assert_array_equal(r1.train.C, r2.train.C)


# ── LabelNoiseGenerator ────────────────────────────────────────────


class TestLabelNoiseGenerator:
    def test_returns_new_dataset(self):
        ds = _make_dataset()
        result = LabelNoiseGenerator(ds, p=0.5, seed=0).generate()
        assert result is not ds

    def test_original_unchanged(self):
        ds = _make_dataset()
        original_y = ds.train.y.copy()
        LabelNoiseGenerator(ds, p=1.0, seed=0).generate()
        np.testing.assert_array_equal(ds.train.y, original_y)

    def test_noise_applied(self):
        ds = _make_dataset()
        result = LabelNoiseGenerator(
            ds, p=1.0, seed=0, config={"flip_matrix": [[0, 1], [1, 0]]}
        ).generate()
        assert not np.array_equal(result.train.y, ds.train.y)

    def test_reproducible(self):
        ds = _make_dataset()
        r1 = LabelNoiseGenerator(ds, p=0.5, seed=42).generate()
        r2 = LabelNoiseGenerator(ds, p=0.5, seed=42).generate()
        np.testing.assert_array_equal(r1.train.y, r2.train.y)


# ── Composition ─────────────────────────────────────────────────────


def test_chaining_transforms():
    """Chaining multiple transforms produces correct result."""
    ds = _make_dataset(k=4)
    dropped = ConceptDropGenerator(ds, ["c0"]).generate()
    noisy = ConceptNoiseGenerator(dropped, p=0.5, seed=1).generate()

    assert noisy.n_concepts == 3
    assert ds.n_concepts == 4  # original untouched
    assert not np.array_equal(noisy.train.C, dropped.train.C)


def test_noise_before_sampling():
    """Applying noise before sampling works correctly."""
    rng = np.random.default_rng(99)
    X = rng.random((40, 4)).astype(np.float32)
    C = rng.integers(0, 2, size=(40, 3)).astype(np.int8)
    y = np.tile([0, 1], 20).astype(np.int32)
    ds = ConceptDataset(
        X,
        C,
        y,
        {"classes": ["a", "b"], "concepts": ["c0", "c1", "c2"], "data_type": "tabular"},
    )

    noisy = ConceptNoiseGenerator(ds, p=0.5, seed=7).generate()
    noisy.sample(test_size=0.2, val_size=0.2, seed=1)

    assert noisy.train is not None
    assert noisy.train.n > 0
