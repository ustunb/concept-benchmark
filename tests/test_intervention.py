"""Tests for concept_benchmark.intervention module."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionBatch,
    InterventionConfig,
    InterventionError,
    OrderedCBMStrategy,
    RandomInterventionStrategy,
    ScoreIntervention,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_batch(n=10, k=4, *, seed=0):
    """Create a tiny InterventionBatch with random data."""
    rng = np.random.default_rng(seed)
    C_pred = rng.random((n, k)).astype(np.float32)
    C_true = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y_true = rng.integers(0, 2, size=n).astype(np.int32)
    return InterventionBatch(C_pred=C_pred, C_true=C_true, y_true=y_true)


def _make_cbm(k=4, seed=42):
    """Build a tiny trained ConceptBasedModel on random tabular data."""
    from experiments.models import ConceptBasedModel, FrontEndModel

    fe = FrontEndModel()
    rng = np.random.default_rng(seed)
    C = rng.random((40, k)).astype(np.float32)
    y = rng.integers(0, 2, size=40).astype(np.int32)
    fe.fit(C, y)
    return ConceptBasedModel(front_end_model=fe)


# ── InterventionConfig ───────────────────────────────────────────────


class TestInterventionConfig:
    def test_default_values(self):
        cfg = InterventionConfig()
        assert cfg.tau is None
        assert cfg.concept_budget is None
        assert cfg.max_concepts_per_instance is None
        assert cfg.random_state is None
        assert cfg.score_threshold == 0.2

    def test_rng_seeded(self):
        cfg1 = InterventionConfig(random_state=42)
        cfg2 = InterventionConfig(random_state=42)
        v1 = cfg1.rng.random(5)
        v2 = cfg2.rng.random(5)
        np.testing.assert_array_equal(v1, v2)

    def test_per_instance_limit(self):
        cfg = InterventionConfig(max_concepts_per_instance=3)
        assert cfg.per_instance_limit(10) == 3
        assert cfg.per_instance_limit(2) == 2

    def test_per_instance_limit_none(self):
        cfg = InterventionConfig()
        assert cfg.per_instance_limit(7) == 7

    def test_tau_validation(self):
        with pytest.raises(ValueError, match="tau must lie within"):
            InterventionConfig(tau=0.8)

    def test_resolve_budget_fraction(self):
        cfg = InterventionConfig(concept_budget=0.5)
        assert cfg.resolve_concept_budget(100) == 50

    def test_resolve_budget_count(self):
        cfg = InterventionConfig(concept_budget=10)
        assert cfg.resolve_concept_budget(100) == 10

    def test_resolve_budget_none(self):
        cfg = InterventionConfig()
        assert cfg.resolve_concept_budget(50) == 50


# ── InterventionBatch ────────────────────────────────────────────────


class TestInterventionBatch:
    def test_shape_validation(self):
        with pytest.raises(ValueError, match="identical shape"):
            InterventionBatch(
                C_pred=np.zeros((5, 3)),
                C_true=np.zeros((5, 4)),
            )

    def test_auto_instance_ids(self):
        batch = _make_batch(n=8, k=3)
        np.testing.assert_array_equal(batch.instance_ids, np.arange(8))

    def test_properties(self):
        batch = _make_batch(n=12, k=5)
        assert batch.n_samples == 12
        assert batch.n_concepts == 5

    def test_explicit_instance_ids(self):
        ids = np.array([10, 20, 30])
        batch = InterventionBatch(
            C_pred=np.zeros((3, 2)),
            C_true=np.ones((3, 2)),
            instance_ids=ids,
        )
        np.testing.assert_array_equal(batch.instance_ids, ids)


# ── OrderedCBMStrategy ───────────────────────────────────────────────


class TestOrderedCBMStrategy:
    def test_propose_shape(self):
        k = 4
        model = _make_cbm(k=k)
        batch = _make_batch(n=10, k=k)
        config = InterventionConfig(max_concepts_per_instance=2, random_state=0)
        strat = OrderedCBMStrategy()
        strat.prepare(model, batch, config)
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (10, k)
        assert proposal.mask.dtype == bool

    def test_respects_k(self):
        k = 4
        model = _make_cbm(k=k)
        batch = _make_batch(n=10, k=k)
        config = InterventionConfig(max_concepts_per_instance=1, random_state=0)
        strat = OrderedCBMStrategy()
        strat.prepare(model, batch, config)
        proposal = strat.propose(model, batch, config)
        per_row = proposal.mask.sum(axis=1)
        assert np.all(per_row <= 1)


# ── RandomInterventionStrategy ───────────────────────────────────────


class TestRandomInterventionStrategy:
    def test_valid_mask(self):
        k = 4
        model = _make_cbm(k=k)
        batch = _make_batch(n=10, k=k)
        config = InterventionConfig(max_concepts_per_instance=2, random_state=0)
        strat = RandomInterventionStrategy()
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (10, k)
        assert proposal.mask.dtype == bool

    def test_reproducible(self):
        k = 4
        model = _make_cbm(k=k)
        batch = _make_batch(n=10, k=k)
        config1 = InterventionConfig(max_concepts_per_instance=2, random_state=99)
        config2 = InterventionConfig(max_concepts_per_instance=2, random_state=99)
        strat1 = RandomInterventionStrategy()
        strat2 = RandomInterventionStrategy()
        m1 = strat1.propose(model, batch, config1).mask
        m2 = strat2.propose(model, batch, config2).mask
        np.testing.assert_array_equal(m1, m2)


# ── ConceptualSafeguardsStrategy ─────────────────────────────────────


class TestConceptualSafeguardsStrategy:
    def test_requires_tau(self):
        k = 4
        model = _make_cbm(k=k)
        batch = _make_batch(n=10, k=k)
        config = InterventionConfig()  # tau=None
        strat = ConceptualSafeguardsStrategy()
        with pytest.raises(InterventionError, match="tau"):
            strat.propose(model, batch, config)

    def test_targets_uncertain(self):
        k = 4
        model = _make_cbm(k=k)
        # Make predictions very confident (all 0 or 1)
        batch = InterventionBatch(
            C_pred=np.round(np.random.default_rng(0).random((10, k))).astype(
                np.float32
            ),
            C_true=np.ones((10, k), dtype=np.float32),
            y_true=np.zeros(10, dtype=np.int32),
        )
        config = InterventionConfig(
            tau=0.3, max_concepts_per_instance=2, random_state=0
        )
        strat = ConceptualSafeguardsStrategy()
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (10, k)


# ── ScoreIntervention ────────────────────────────────────────────────


class TestScoreIntervention:
    def test_propose_shape(self):
        k = 3
        model = _make_cbm(k=k)
        batch = _make_batch(n=8, k=k)
        config = InterventionConfig(
            max_concepts_per_instance=2,
            score_threshold=0.1,
            random_state=0,
        )
        strat = ScoreIntervention()
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (8, k)
        assert proposal.mask.dtype == bool
        assert "scores" in proposal.details
        assert "threshold" in proposal.details


# ── ConceptInterventionRunner ────────────────────────────────────────


class TestRunner:
    def _make_runner_data(self, n=20, k=4, seed=42):
        """Build model + ConceptDatasetSample for runner tests."""
        from concept_benchmark.data import ConceptDatasetSample
        from experiments.models import ConceptBasedModel, FrontEndModel

        rng = np.random.default_rng(seed)
        X = rng.random((n, 8)).astype(np.float32)
        C = rng.random((n, k)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.int32)
        meta = {
            "classes": ["c0", "c1"],
            "concepts": [f"z{i}" for i in range(k)],
            "data_type": "tabular",
        }
        sample = ConceptDatasetSample(X=X, C=C, y=y, meta=meta)

        fe = FrontEndModel()
        fe.fit((C > 0.5).astype(float), y)
        model = ConceptBasedModel(front_end_model=fe)
        return model, sample

    def test_run_shapes(self):
        model, sample = self._make_runner_data()
        runner = ConceptInterventionRunner(model)
        config = InterventionConfig(max_concepts_per_instance=2, random_state=0)
        strat = RandomInterventionStrategy()
        # Pass concept_proba explicitly since model has no concept_detector
        C_pred = sample.C.copy()
        result = runner.run(strat, config, sample, concept_proba=C_pred)
        n = sample.n
        assert result.C_pred.shape[0] == n
        assert result.C_intervened.shape[0] == n
        assert result.mask.shape[0] == n
        assert result.y_prob_before.shape[0] == n
        assert result.y_prob_after.shape[0] == n
        assert result.y_pred_after.shape[0] == n

    def test_applies_ground_truth(self):
        model, sample = self._make_runner_data(n=10, k=3)
        runner = ConceptInterventionRunner(model)
        config = InterventionConfig(max_concepts_per_instance=3, random_state=0)
        strat = RandomInterventionStrategy()
        C_pred = sample.C.copy()
        result = runner.run(strat, config, sample, concept_proba=C_pred)
        # Where mask is True, C_intervened should equal C_true (base_concepts)
        where_mask = result.mask
        if where_mask.any():
            np.testing.assert_array_equal(
                result.C_intervened[where_mask],
                sample.base_concepts[where_mask],
            )
