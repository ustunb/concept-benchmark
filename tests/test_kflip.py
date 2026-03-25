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


class _RecordingAlignedFrontEnd(FrontEndModel):
    supports_aligned_concept_replay = True
    _kflip_fast_path = False

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, np.ndarray | None]] = []

    @staticmethod
    def _label_probs(concepts: np.ndarray) -> np.ndarray:
        score = 1.2 * concepts[:, 0] - 0.8 * concepts[:, 1]
        if concepts.shape[1] > 2:
            score = score + 0.3 * concepts[:, 2]
        prob1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - prob1, prob1]).astype(np.float32)

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        raise AssertionError("KFlip should use aligned replay for this frontend.")

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

    def test_aligned_replay_metadata_supports_expanded_candidate_rows(self):
        batch = InterventionBatch(
            C_pred=np.array(
                [
                    [0.20, 0.80],
                    [0.75, 0.35],
                ],
                dtype=np.float32,
            ),
            C_true=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            y_true=np.array([0, 1], dtype=np.int32),
        )
        fe = _RecordingAlignedFrontEnd()
        model = ConceptBasedModel(label_predictor=fe)
        proposal = KFlipInterventionStrategy(batch_size=4).propose(
            model,
            batch,
            InterventionConfig(
                max_concepts_per_instance=1,
                score_threshold=0.0,
                random_state=0,
            ),
        )

        assert proposal.mask.shape == batch.C_pred.shape
        expanded_calls = [
            call
            for call in fe.calls
            if call["row_indices"] is not None
            and call["row_indices"].shape[0] > batch.n_samples
        ]
        assert expanded_calls, "Expected expanded candidate replay calls."

        candidate_call = expanded_calls[0]
        row_indices = candidate_call["row_indices"]
        assert len(np.unique(row_indices)) < len(row_indices)
        np.testing.assert_allclose(
            candidate_call["baseline_concepts"],
            batch.C_pred[row_indices],
        )
        np.testing.assert_allclose(
            candidate_call["effective"],
            np.where(
                candidate_call["intervention_mask"],
                candidate_call["concepts"],
                candidate_call["baseline_concepts"],
            ),
        )
