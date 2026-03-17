"""Tests for concept_benchmark.alignment module."""

from __future__ import annotations

import numpy as np
import pytest

from concept_benchmark.alignment import (
    ConstrainedFrontEndModel,
    align_frontend_weights,
    retrain_aligned,
    test_alignment as alignment_test_fn,
)
from concept_benchmark.models import FrontEndModel


def _fit_frontend(k=4, n=50, seed=42):
    """Build and fit a simple FrontEndModel."""
    rng = np.random.default_rng(seed)
    C = rng.random((n, k)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    fe = FrontEndModel()
    fe.fit(C, y)
    return fe, C, y


# ── ConstrainedFrontEndModel ─────────────────────────────────────────


class TestConstrainedFE:
    def test_unconstrained_uses_sklearn(self):
        """No constraints → falls through to LogisticRegression."""
        names = ["c0", "c1", "c2"]
        fe = ConstrainedFrontEndModel(concept_names=names)
        rng = np.random.default_rng(0)
        C = rng.random((40, 3)).astype(np.float32)
        y = rng.integers(0, 2, size=40).astype(np.int32)
        fe.fit(C, y)
        assert hasattr(fe.model, "coef_")
        assert fe.model.coef_.shape[1] == 3

    def test_constrained_respects_sign(self):
        cvxpy = pytest.importorskip("cvxpy")
        names = ["c0", "c1", "c2"]
        fe = ConstrainedFrontEndModel(
            concept_names=names,
            monotonicity_constraints={"c0": 1},  # c0 weight >= 0
        )
        rng = np.random.default_rng(0)
        C = rng.random((60, 3)).astype(np.float32)
        y = rng.integers(0, 2, size=60).astype(np.int32)
        fe.fit(C, y)
        assert fe.model.coef_[0, 0] >= -1e-6  # should be >= 0

    def test_predict_binary(self):
        names = ["c0", "c1"]
        fe = ConstrainedFrontEndModel(concept_names=names)
        rng = np.random.default_rng(1)
        C = rng.random((30, 2)).astype(np.float32)
        y = rng.integers(0, 2, size=30).astype(np.int32)
        fe.fit(C, y)
        preds = fe.predict(C)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_valid(self):
        names = ["c0", "c1"]
        fe = ConstrainedFrontEndModel(concept_names=names)
        rng = np.random.default_rng(2)
        C = rng.random((30, 2)).astype(np.float32)
        y = rng.integers(0, 2, size=30).astype(np.int32)
        fe.fit(C, y)
        proba = fe.predict_proba(C)
        assert proba.shape == (30, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0) and np.all(proba <= 1)


# ── retrain_aligned ──────────────────────────────────────────────────


class TestRetrainAligned:
    def test_returns_expected_keys(self):
        fe, C, y = _fit_frontend(k=3, n=60)
        result = retrain_aligned(
            h_train=C[:40],
            y_train=y[:40],
            h_test=C[40:],
            y_test=y[40:],
            concept_names=["c0", "c1", "c2"],
            original_frontend=fe,
            monotonicity_constraints={"c0": 1},
        )
        for key in (
            "original_accuracy",
            "aligned_accuracy",
            "accuracy_change",
            "predictions_changed",
            "aligned_weights",
        ):
            assert key in result, f"Missing key: {key}"

    def test_accuracy_change_consistent(self):
        fe, C, y = _fit_frontend(k=3, n=60)
        result = retrain_aligned(
            h_train=C[:40],
            y_train=y[:40],
            h_test=C[40:],
            y_test=y[40:],
            concept_names=["c0", "c1", "c2"],
            original_frontend=fe,
            monotonicity_constraints={"c0": 1},
        )
        expected_delta = result["aligned_accuracy"] - result["original_accuracy"]
        assert abs(result["accuracy_change"] - expected_delta) < 1e-8


# ── align_frontend_weights ───────────────────────────────────────────


class TestAlignWeights:
    def test_weights_set_correctly(self):
        fe, _, _ = _fit_frontend(k=3)
        names = ["c0", "c1", "c2"]
        wdict = {"c0": 1.5, "c1": -2.0, "c2": 0.5, "bias": 0.1}
        align_frontend_weights(fe, names, wdict)
        np.testing.assert_allclose(fe.model.coef_[0], [1.5, -2.0, 0.5])

    def test_bias_set(self):
        fe, _, _ = _fit_frontend(k=3)
        names = ["c0", "c1", "c2"]
        wdict = {"c0": 1.0, "c1": 1.0, "c2": 1.0, "bias": -3.0}
        align_frontend_weights(fe, names, wdict)
        assert fe.model.intercept_[0] == pytest.approx(-3.0)


# ── test_alignment ───────────────────────────────────────────────────


class TestTestAlignment:
    def _make_test_dataset(self, k=3, n=20, seed=0):
        """Create a mock dataset with .y and .concepts."""

        class _DS:
            pass

        rng = np.random.default_rng(seed)
        ds = _DS()
        ds.y = rng.integers(0, 2, size=n).astype(np.int32)
        ds.concepts = [f"c{i}" for i in range(k)]
        return ds

    def test_returns_expected_keys(self):
        fe, C, y = _fit_frontend(k=3, n=40)
        ds = self._make_test_dataset(k=3, n=20)
        h_test = np.random.default_rng(1).random((20, 3)).astype(np.float32)
        params = {"c0": 1.0, "c1": -1.0, "c2": 0.5, "bias": 0.0}
        result = alignment_test_fn(h_test, params, fe, ds)
        for key in (
            "original_accuracy",
            "aligned_accuracy",
            "accuracy_change",
            "predictions_changed",
        ):
            assert key in result

    def test_original_not_modified(self):
        fe, C, y = _fit_frontend(k=3, n=40)
        original_coef = fe.model.coef_.copy()
        ds = self._make_test_dataset(k=3, n=20)
        h_test = np.random.default_rng(1).random((20, 3)).astype(np.float32)
        params = {"c0": 99.0, "c1": -99.0, "c2": 0.0, "bias": 5.0}
        alignment_test_fn(h_test, params, fe, ds)
        # Original should NOT be modified (deep copy inside)
        np.testing.assert_array_equal(fe.model.coef_, original_coef)
