"""Tests for concept_benchmark.kflip module."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.intervention import (
    InterventionBatch,
    InterventionConfig,
    InterventionError,
)
from experiments.kflip import KFlipInterventionStrategy
from experiments.models import ConceptBasedModel, FrontEndModel


def _make_model(k=4, seed=42):
    """Build a tiny CBM with logistic regression frontend (enables fast path)."""
    rng = np.random.default_rng(seed)
    C = rng.random((50, k)).astype(np.float32)
    y = rng.integers(0, 2, size=50).astype(np.int32)
    fe = FrontEndModel()
    fe.fit(C, y)
    return ConceptBasedModel(label_predictor=fe)


def _make_batch(n=10, k=4, seed=0):
    rng = np.random.default_rng(seed)
    C_pred = rng.random((n, k)).astype(np.float32)
    C_true = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y_true = rng.integers(0, 2, size=n).astype(np.int32)
    return InterventionBatch(C_pred=C_pred, C_true=C_true, y_true=y_true)


class TestKFlip:
    def test_requires_positive_k(self):
        model = _make_model()
        batch = _make_batch()
        config = InterventionConfig(max_concepts_per_instance=0)
        strat = KFlipInterventionStrategy()
        with pytest.raises(InterventionError, match="positive integer"):
            strat.propose(model, batch, config)

    def test_requires_k_not_none(self):
        model = _make_model()
        batch = _make_batch()
        config = InterventionConfig()  # max_concepts_per_instance=None
        strat = KFlipInterventionStrategy()
        with pytest.raises(InterventionError, match="positive integer"):
            strat.propose(model, batch, config)

    def test_propose_valid_mask(self):
        k = 4
        model = _make_model(k=k)
        batch = _make_batch(n=8, k=k)
        config = InterventionConfig(
            max_concepts_per_instance=2,
            score_threshold=0.1,
            random_state=0,
        )
        strat = KFlipInterventionStrategy()
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (8, k)
        assert proposal.mask.dtype == bool

    def test_at_most_k_per_instance(self):
        k = 4
        model = _make_model(k=k)
        batch = _make_batch(n=10, k=k)
        config = InterventionConfig(
            max_concepts_per_instance=2,
            score_threshold=0.0,  # select everything
            random_state=0,
        )
        strat = KFlipInterventionStrategy()
        proposal = strat.propose(model, batch, config)
        per_row = proposal.mask.sum(axis=1)
        assert np.all(per_row <= 2)

    def test_high_threshold_fewer_selected(self):
        k = 4
        model = _make_model(k=k)
        batch = _make_batch(n=10, k=k)
        low = InterventionConfig(
            max_concepts_per_instance=2, score_threshold=0.01, random_state=0
        )
        high = InterventionConfig(
            max_concepts_per_instance=2, score_threshold=0.99, random_state=0
        )
        strat_low = KFlipInterventionStrategy()
        strat_high = KFlipInterventionStrategy()
        m_low = strat_low.propose(model, batch, low).mask.sum()
        m_high = strat_high.propose(model, batch, high).mask.sum()
        assert m_high <= m_low

    def test_details_keys(self):
        k = 3
        model = _make_model(k=k)
        batch = _make_batch(n=6, k=k)
        config = InterventionConfig(
            max_concepts_per_instance=1,
            score_threshold=0.1,
            random_state=0,
        )
        strat = KFlipInterventionStrategy()
        proposal = strat.propose(model, batch, config)
        for key in ("flip_prob", "best_subset", "k", "threshold"):
            assert key in proposal.details, f"Missing detail key: {key}"

    def test_fast_path_matches_general(self):
        """Disable fast path and compare results to fast path."""
        k = 3
        model = _make_model(k=k)
        batch = _make_batch(n=8, k=k, seed=7)
        config = InterventionConfig(
            max_concepts_per_instance=1,
            score_threshold=0.1,
            random_state=0,
        )
        # Fast path (default for logistic regression)
        strat_fast = KFlipInterventionStrategy()
        p_fast = strat_fast.propose(model, batch, config)

        # General path (disable fast path)
        model.label_predictor._kflip_fast_path = False
        config2 = InterventionConfig(
            max_concepts_per_instance=1,
            score_threshold=0.1,
            random_state=0,
        )
        strat_gen = KFlipInterventionStrategy()
        p_gen = strat_gen.propose(model, batch, config2)

        np.testing.assert_array_equal(p_fast.mask, p_gen.mask)
        np.testing.assert_allclose(
            p_fast.details["flip_prob"],
            p_gen.details["flip_prob"],
            atol=1e-6,
        )

    def test_exact_k_true(self):
        k = 3
        model = _make_model(k=k)
        batch = _make_batch(n=8, k=k)
        config = InterventionConfig(
            max_concepts_per_instance=2,
            score_threshold=0.0,
            random_state=0,
        )
        strat = KFlipInterventionStrategy(use_exact_k=True)
        proposal = strat.propose(model, batch, config)
        # Every intervened row should have exactly 2 concepts (or 0 if not selected)
        per_row = proposal.mask.sum(axis=1)
        assert np.all((per_row == 2) | (per_row == 0))

    def test_exact_k_false_includes_smaller(self):
        k = 4
        model = _make_model(k=k)
        batch = _make_batch(n=10, k=k, seed=3)
        config = InterventionConfig(
            max_concepts_per_instance=3,
            score_threshold=0.0,
            random_state=0,
        )
        strat = KFlipInterventionStrategy(use_exact_k=False)
        proposal = strat.propose(model, batch, config)
        per_row = proposal.mask.sum(axis=1)
        # With use_exact_k=False, subsets of size 1..k are allowed
        assert np.all(per_row <= 3)
        # Verify that at least one row selected fewer than max (smaller subset)
        assert np.any(per_row < 3), (
            "Expected at least one row with fewer than 3 concepts selected"
        )

    def test_limit_subsets(self):
        k = 4
        model = _make_model(k=k)
        batch = _make_batch(n=6, k=k)
        config = InterventionConfig(
            max_concepts_per_instance=2,
            score_threshold=0.1,
            random_state=0,
        )
        strat = KFlipInterventionStrategy(limit_subsets=3)
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (6, k)

    def test_large_k_with_limit_subsets(self):
        """k=20 concepts with limit_subsets cap — should complete quickly."""
        k = 20
        model = _make_model(k=k)
        batch = _make_batch(n=5, k=k, seed=7)
        config = InterventionConfig(
            max_concepts_per_instance=20,
            score_threshold=0.0,
            random_state=0,
        )
        # Without limit_subsets, C(20,1)+...+C(20,20) = 2^20-1 ~1M subsets per sample.
        # With limit_subsets=500, it stays tractable.
        strat = KFlipInterventionStrategy(limit_subsets=500)
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (5, k)
        assert proposal.mask.dtype == bool
