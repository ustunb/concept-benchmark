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
    InterventionStrategy,
    OrderedCBMStrategy,
    predict_label_proba_from_concepts,
    RandomInterventionStrategy,
    ScoreIntervention,
    StrategyProposal,
)
from experiments.models import FrontEndModel


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
    return ConceptBasedModel(label_predictor=fe)


class _RecordingAlignedFrontEnd(FrontEndModel):
    supports_aligned_concept_replay = True
    _kflip_fast_path = False

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, np.ndarray | None]] = []

    def fit(self, C: np.ndarray, y: np.ndarray, fit_params: dict | None = None) -> None:
        super().fit(C, y, fit_params=fit_params)

    @staticmethod
    def _label_probs(concepts: np.ndarray) -> np.ndarray:
        score = concepts.sum(axis=1)
        prob1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - prob1, prob1]).astype(np.float32)

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        raise AssertionError("Aligned replay should use predict_proba_from_concepts.")

    def predict_proba_from_concepts(
        self,
        concepts: np.ndarray,
        *,
        row_indices: np.ndarray | None = None,
        baseline_concepts: np.ndarray | None = None,
        intervention_mask: np.ndarray | None = None,
        **_: dict,
    ) -> np.ndarray:
        concepts = np.asarray(concepts, dtype=np.float32)
        if baseline_concepts is None:
            baseline_concepts = concepts
        else:
            baseline_concepts = np.asarray(baseline_concepts, dtype=np.float32)
        if intervention_mask is None:
            effective = concepts
        else:
            effective = np.where(
                np.asarray(intervention_mask, dtype=bool), concepts, baseline_concepts
            )
        self.calls.append(
            {
                "concepts": concepts.copy(),
                "row_indices": None
                if row_indices is None
                else np.asarray(row_indices, dtype=int).copy(),
                "baseline_concepts": baseline_concepts.copy(),
                "intervention_mask": None
                if intervention_mask is None
                else np.asarray(intervention_mask, dtype=bool).copy(),
                "effective": effective.copy(),
            }
        )
        return self._label_probs(effective)


class _FixedMaskStrategy(InterventionStrategy):
    def __init__(self, mask: np.ndarray) -> None:
        super().__init__(name="fixed_mask")
        self._mask = np.asarray(mask, dtype=bool)

    def propose(self, model, batch, config) -> StrategyProposal:
        return StrategyProposal(mask=self._mask.copy())


class _PrimingSensitiveAlignedFrontEnd(FrontEndModel):
    supports_aligned_concept_replay = True
    _kflip_fast_path = False

    def __init__(self) -> None:
        super().__init__()
        self.primed_dataset_id: int | None = None
        self.calls: list[dict[str, np.ndarray | int | None]] = []

    @staticmethod
    def _label_probs(concepts: np.ndarray) -> np.ndarray:
        score = 0.6 * concepts[:, 0] - 0.3 * concepts[:, 1]
        if concepts.shape[1] > 2:
            score = score + 0.2 * concepts[:, 2]
        prob1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - prob1, prob1]).astype(np.float32)

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        raise AssertionError("Aligned replay test should not use predict_proba.")

    def predict_proba_from_concepts(
        self,
        concepts: np.ndarray,
        *,
        dataset=None,
        row_indices: np.ndarray | None = None,
        baseline_concepts: np.ndarray | None = None,
        intervention_mask: np.ndarray | None = None,
        **_: dict,
    ) -> np.ndarray:
        concepts = np.asarray(concepts, dtype=np.float32)
        if dataset is not None:
            self.primed_dataset_id = id(dataset)
        elif self.primed_dataset_id is None:
            raise RuntimeError("cache not primed")

        if baseline_concepts is None:
            baseline_concepts = concepts
        else:
            baseline_concepts = np.asarray(baseline_concepts, dtype=np.float32)
        if intervention_mask is None:
            effective = concepts
        else:
            effective = np.where(
                np.asarray(intervention_mask, dtype=bool), concepts, baseline_concepts
            )
        self.calls.append(
            {
                "dataset_id": None if dataset is None else id(dataset),
                "row_indices": None
                if row_indices is None
                else np.asarray(row_indices, dtype=int).copy(),
                "effective": effective.copy(),
            }
        )
        return self._label_probs(effective)


class _ReplayDuringPrepareStrategy(InterventionStrategy):
    def __init__(self) -> None:
        super().__init__(name="replay_prepare")

    def prepare(self, model, batch, config) -> None:
        super().prepare(model, batch, config)
        predict_label_proba_from_concepts(
            model,
            batch.C_pred,
            row_indices=np.arange(batch.n_samples, dtype=int),
            baseline_concepts=batch.C_pred,
        )

    def propose(self, model, batch, config) -> StrategyProposal:
        return StrategyProposal(mask=np.zeros_like(batch.C_pred, dtype=bool))


class _ReplayDuringProposeStrategy(InterventionStrategy):
    def __init__(self) -> None:
        super().__init__(name="replay_propose")

    def propose(self, model, batch, config) -> StrategyProposal:
        predict_label_proba_from_concepts(
            model,
            batch.C_pred,
            row_indices=np.arange(batch.n_samples, dtype=int),
            baseline_concepts=batch.C_pred,
        )
        return StrategyProposal(mask=np.zeros_like(batch.C_pred, dtype=bool))


# ── InterventionConfig ───────────────────────────────────────────────


class TestInterventionConfig:
    def test_default_values(self):
        cfg = InterventionConfig()
        assert cfg.abstention_threshold is None
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
        with pytest.raises(ValueError, match="abstention_threshold must lie within"):
            InterventionConfig(abstention_threshold=0.8)

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
        config = InterventionConfig()  # abstention_threshold=None
        strat = ConceptualSafeguardsStrategy()
        with pytest.raises(InterventionError, match="abstention_threshold"):
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
            abstention_threshold=0.3, max_concepts_per_instance=2, random_state=0
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
        sample = ConceptDatasetSample(X=X, C=C, y=y, meta=meta, input_type="tabular", classes=(0, 1))

        fe = FrontEndModel()
        fe.fit((C > 0.5).astype(float), y)
        model = ConceptBasedModel(label_predictor=fe)
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

    def test_run_preserves_continuous_non_intervened_concepts_for_aligned_replay(self):
        from concept_benchmark.data import ConceptDatasetSample
        from experiments.models import ConceptBasedModel

        instance_ids = np.array([7, 3], dtype=int)
        C_pred = np.array(
            [
                [0.20, 0.80, 0.40],
                [0.75, 0.25, 0.60],
            ],
            dtype=np.float32,
        )
        C_true = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        sample = ConceptDatasetSample(
            X=np.zeros((2, 2), dtype=np.float32),
            C=C_true.astype(np.int8),
            y=np.array([0, 1], dtype=np.int32),
            meta={
                "classes": ["c0", "c1"],
                "concepts": ["z0", "z1", "z2"],
                "data_type": "tabular",
            },
            input_type="tabular",
            classes=(0, 1),
        )

        fe = _RecordingAlignedFrontEnd()
        model = ConceptBasedModel(concept_detector=None, label_predictor=fe)
        mask = np.array(
            [
                [True, False, False],
                [False, True, False],
            ],
            dtype=bool,
        )
        result = ConceptInterventionRunner(model).run(
            _FixedMaskStrategy(mask),
            InterventionConfig(max_concepts_per_instance=1, random_state=0),
            sample,
            concept_proba=C_pred,
            instance_ids=instance_ids,
        )

        assert len(fe.calls) == 3
        np.testing.assert_array_equal(fe.calls[0]["row_indices"], instance_ids)
        np.testing.assert_allclose(fe.calls[0]["effective"], C_pred)
        np.testing.assert_array_equal(fe.calls[1]["row_indices"], instance_ids)
        np.testing.assert_allclose(fe.calls[1]["effective"], C_pred)

        expected_after = np.where(mask, C_true, C_pred)
        np.testing.assert_array_equal(fe.calls[2]["row_indices"], instance_ids)
        np.testing.assert_allclose(fe.calls[2]["baseline_concepts"], C_pred)
        np.testing.assert_allclose(fe.calls[2]["effective"], expected_after)
        np.testing.assert_allclose(result.C_intervened, expected_after)
        np.testing.assert_allclose(
            result.y_prob_after,
            _RecordingAlignedFrontEnd._label_probs(expected_after),
        )
        assert result.mask.shape[0] == sample.n
        assert result.y_prob_before.shape[0] == sample.n
        assert result.y_prob_after.shape[0] == sample.n
        assert result.y_pred_after.shape[0] == sample.n

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

    def test_prepare_primes_aligned_replay_context_before_strategy_prepare(self):
        from concept_benchmark.data import ConceptDatasetSample
        from experiments.models import ConceptBasedModel

        sample = ConceptDatasetSample(
            X=np.zeros((3, 2), dtype=np.float32),
            C=np.array(
                [[1, 0, 0], [0, 1, 1], [1, 1, 0]],
                dtype=np.int8,
            ),
            y=np.array([0, 1, 0], dtype=np.int32),
            meta={
                "classes": ["c0", "c1"],
                "concepts": ["z0", "z1", "z2"],
                "data_type": "tabular",
            },
            input_type="tabular",
            classes=(0, 1),
        )
        concept_proba = np.array(
            [
                [0.2, 0.8, 0.4],
                [0.7, 0.3, 0.6],
                [0.6, 0.5, 0.1],
            ],
            dtype=np.float32,
        )

        fe = _PrimingSensitiveAlignedFrontEnd()
        model = ConceptBasedModel(concept_detector=None, label_predictor=fe)
        ConceptInterventionRunner(model).prepare(
            _ReplayDuringPrepareStrategy(),
            InterventionConfig(max_concepts_per_instance=1, random_state=0),
            sample,
            concept_proba=concept_proba,
        )

        assert fe.primed_dataset_id == id(sample)
        assert len(fe.calls) >= 2
        assert fe.calls[0]["dataset_id"] == id(sample)
        assert fe.calls[1]["dataset_id"] is None

    def test_run_primes_aligned_replay_context_before_strategy_propose(self):
        from concept_benchmark.data import ConceptDatasetSample
        from experiments.models import ConceptBasedModel

        sample = ConceptDatasetSample(
            X=np.zeros((3, 2), dtype=np.float32),
            C=np.array(
                [[1, 0, 0], [0, 1, 1], [1, 1, 0]],
                dtype=np.int8,
            ),
            y=np.array([0, 1, 0], dtype=np.int32),
            meta={
                "classes": ["c0", "c1"],
                "concepts": ["z0", "z1", "z2"],
                "data_type": "tabular",
            },
            input_type="tabular",
            classes=(0, 1),
        )
        concept_proba = np.array(
            [
                [0.2, 0.8, 0.4],
                [0.7, 0.3, 0.6],
                [0.6, 0.5, 0.1],
            ],
            dtype=np.float32,
        )

        fe = _PrimingSensitiveAlignedFrontEnd()
        model = ConceptBasedModel(concept_detector=None, label_predictor=fe)
        result = ConceptInterventionRunner(model).run(
            _ReplayDuringProposeStrategy(),
            InterventionConfig(max_concepts_per_instance=1, random_state=0),
            sample,
            concept_proba=concept_proba,
        )

        assert fe.primed_dataset_id == id(sample)
        assert len(fe.calls) >= 4
        assert fe.calls[0]["dataset_id"] == id(sample)
        assert fe.calls[1]["dataset_id"] is None
        assert result.mask.shape == concept_proba.shape
