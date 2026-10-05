"""Tests for concept_benchmark.kflip module."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

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
        config = InterventionConfig(per_instance_budget=0)
        strat = KFlipInterventionStrategy()
        with pytest.raises(InterventionError, match="positive integer"):
            strat.propose(model, batch, config)

    def test_requires_k_not_none(self):
        model = _make_model()
        batch = _make_batch()
        config = InterventionConfig()  # per_instance_budget=None
        strat = KFlipInterventionStrategy()
        with pytest.raises(InterventionError, match="positive integer"):
            strat.propose(model, batch, config)

    def test_propose_valid_mask(self):
        k = 4
        model = _make_model(k=k)
        batch = _make_batch(n=8, k=k)
        config = InterventionConfig(
            per_instance_budget=2,
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
            per_instance_budget=2,
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
            per_instance_budget=2, score_threshold=0.01, random_state=0
        )
        high = InterventionConfig(
            per_instance_budget=2, score_threshold=0.99, random_state=0
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
            per_instance_budget=1,
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
            per_instance_budget=1,
            score_threshold=0.1,
            random_state=0,
        )
        # Fast path (default for logistic regression)
        strat_fast = KFlipInterventionStrategy()
        p_fast = strat_fast.propose(model, batch, config)

        # General path (disable fast path)
        model.label_predictor._kflip_fast_path = False
        config2 = InterventionConfig(
            per_instance_budget=1,
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
            per_instance_budget=2,
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
            per_instance_budget=3,
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
            per_instance_budget=2,
            score_threshold=0.1,
            random_state=0,
        )
        strat = KFlipInterventionStrategy(limit_subsets=3)
        proposal = strat.propose(model, batch, config)
        assert proposal.mask.shape == (6, k)

    def test_aligned_replay_metadata_supports_expanded_candidate_rows(self):
        instance_ids = np.array([5, 2], dtype=int)
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
            instance_ids=instance_ids,
        )
        fe = _RecordingAlignedFrontEnd()
        model = ConceptBasedModel(label_predictor=fe)
        proposal = KFlipInterventionStrategy(batch_size=4).propose(
            model,
            batch,
            InterventionConfig(
                per_instance_budget=1,
                score_threshold=0.0,
                random_state=0,
            ),
        )

        assert proposal.mask.shape == batch.C_pred.shape
        np.testing.assert_array_equal(fe.calls[0]["row_indices"], instance_ids)
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
        assert set(np.unique(row_indices)) == {2, 5}
        assert not set(np.unique(row_indices)) == {0, 1}
        row_lookup = {instance_id: pos for pos, instance_id in enumerate(instance_ids)}
        expected_rows = np.array([row_lookup[idx] for idx in row_indices], dtype=int)
        np.testing.assert_allclose(
            candidate_call["baseline_concepts"],
            batch.C_pred[expected_rows],
        )
        np.testing.assert_allclose(
            candidate_call["effective"],
            np.where(
                candidate_call["intervention_mask"],
                candidate_call["concepts"],
                candidate_call["baseline_concepts"],
            ),
        )


def _flip_probability(model, p, subset):
    """Brute-force flip probability of `subset` for one robot with concept probabilities `p` (hard mode)."""
    base = (p >= 0.5).astype(np.float32)
    base_label = model.label_predictor.predict_proba(base[None])[0].argmax()
    total = 0.0
    for values in itertools.product([0.0, 1.0], repeat=len(subset)):
        z = base.copy()
        z[list(subset)] = values
        weight = np.prod(
            [p[j] if v == 1.0 else 1.0 - p[j] for j, v in zip(subset, values)]
        )
        total += weight * (
            model.label_predictor.predict_proba(z[None])[0].argmax() != base_label
        )
    return total


def _mixed_batch(n, k, seed):
    """Concept probabilities with many near-certain values, as concept detectors produce."""
    rng = np.random.default_rng(seed)
    p = rng.random((n, k))
    sure = rng.random((n, k)) < 0.5
    p = np.where(
        sure,
        np.where(
            rng.random((n, k)) < 0.5,
            rng.uniform(0, 0.009, (n, k)),
            rng.uniform(0.991, 1, (n, k)),
        ),
        p,
    )
    return InterventionBatch(
        C_pred=p.astype(np.float32),
        C_true=rng.integers(0, 2, size=(n, k)).astype(np.float32),
        y_true=rng.integers(0, 2, size=n).astype(np.int32),
    )


def _load_reference(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KFlipInterventionStrategy


class TestKFlipDefaultIsThePaperPolicy:
    """The default strategy must choose exactly what the code behind the paper's results chose."""

    @pytest.mark.parametrize("n_concepts", [1, 2, 3, 5, 7, 9, 12])
    def test_same_choices_as_original(self, n_concepts):
        original = _load_reference("kflip_original_reference")
        for seed in range(10):
            rng = np.random.default_rng(77 * n_concepts + seed)
            batch = (
                _mixed_batch(n=30, k=n_concepts, seed=seed)
                if seed % 2
                else _make_batch(n=30, k=n_concepts, seed=seed)
            )
            weights, bias = rng.normal(size=n_concepts) * 2, float(rng.normal())
            for make_model in (
                lambda: _make_model(k=n_concepts, seed=seed),
                lambda: ConceptBasedModel(
                    label_predictor=_EmbeddingStyleFrontEnd(weights, bias)
                ),
            ):
                for budget in sorted({1, min(3, n_concepts)}):
                    for exact in (False, True):
                        config = InterventionConfig(
                            per_instance_budget=budget,
                            score_threshold=0.2,
                            random_state=seed,
                        )
                        new = KFlipInterventionStrategy(use_exact_k=exact).propose(
                            make_model(), batch, config
                        )
                        old = original(use_exact_k=exact).propose(
                            make_model(), batch, config
                        )
                        np.testing.assert_array_equal(new.mask, old.mask)
                        assert new.details["best_subset"] == old.details["best_subset"]
                        np.testing.assert_allclose(
                            new.details["flip_prob"],
                            old.details["flip_prob"],
                            atol=1e-12,
                        )
                # k = max in the pipeline: exactly every concept of the selected instances
                config = InterventionConfig(
                    per_instance_budget=n_concepts,
                    score_threshold=0.2,
                    random_state=seed,
                )
                new = KFlipInterventionStrategy(use_exact_k=True).propose(
                    make_model(), batch, config
                )
                old = original(use_exact_k=True).propose(make_model(), batch, config)
                np.testing.assert_array_equal(new.mask, old.mask)


class _EmbeddingStyleFrontEnd(FrontEndModel):
    """Label predictor that reads continuous concepts and is replayed row by row (no fast path)."""

    supports_aligned_concept_replay = True
    _kflip_fast_path = False

    def __init__(self, weights: np.ndarray, bias: float) -> None:
        super().__init__()
        self.weights, self.bias = weights, bias

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        raise AssertionError("KFlip should use aligned replay for this frontend.")

    def predict_proba_from_concepts(
        self,
        concepts,
        *,
        row_indices=None,
        baseline_concepts=None,
        intervention_mask=None,
    ):
        prob1 = 1.0 / (
            1.0
            + np.exp(
                -(np.asarray(concepts, dtype=np.float64) @ self.weights + self.bias)
            )
        )
        return np.column_stack([1.0 - prob1, prob1]).astype(np.float32)


class TestKFlipManyConcepts:
    """More than 12 concepts: greedy search from the best single concept, sampled flip probability for big subsets."""

    def test_any_candidate_and_any_size_can_be_chosen(self):
        # the label flips as soon as one concept turns out absent, so every concept adds (1 - p) x what is left:
        # the subset should take the least certain concepts, wherever they are, and grow well past twelve
        class _AnyAbsent(_EmbeddingStyleFrontEnd):
            def predict_proba_from_concepts(self, concepts, **kwargs):
                prob1 = (np.asarray(concepts) >= 0.5).all(axis=1).astype(np.float32)
                return np.column_stack([1.0 - prob1, prob1])

        n = 40
        rng = np.random.default_rng(7)
        p = rng.uniform(0.85, 0.95, size=(6, n)).astype(np.float32)
        model = ConceptBasedModel(label_predictor=_AnyAbsent(np.zeros(n), 0.0))
        batch = InterventionBatch(
            C_pred=p,
            C_true=np.ones((6, n), dtype=np.float32),
            y_true=np.ones(6, dtype=np.int32),
        )
        mask = (
            KFlipInterventionStrategy()
            .propose(
                model,
                batch,
                InterventionConfig(per_instance_budget=n, score_threshold=0.0),
            )
            .mask
        )
        assert mask.sum(axis=1).min() > 12  # subsets grow past twelve concepts
        for i in range(
            6
        ):  # the twelve least certain concepts are always among those chosen
            assert mask[i, np.argsort(p[i])[:12]].all()

    def test_sampled_flip_probability_agrees_with_exact(self):
        # 16 concepts whose subsets grow to all 16: sizes 13-16 are sampled by default and still enumerable
        class _AnyAbsent(_EmbeddingStyleFrontEnd):
            def predict_proba_from_concepts(self, concepts, **kwargs):
                prob1 = (np.asarray(concepts) >= 0.5).all(axis=1).astype(np.float32)
                return np.column_stack([1.0 - prob1, prob1])

        n = 16
        for seed in range(5):
            rng = np.random.default_rng(seed)
            p = rng.uniform(0.85, 0.95, size=(8, n)).astype(np.float32)
            model = ConceptBasedModel(label_predictor=_AnyAbsent(np.zeros(n), 0.0))
            batch = InterventionBatch(
                C_pred=p,
                C_true=np.ones((8, n), dtype=np.float32),
                y_true=np.ones(8, dtype=np.int32),
            )
            config = InterventionConfig(
                per_instance_budget=n, score_threshold=0.0, random_state=seed
            )
            sampled = KFlipInterventionStrategy().propose(model, batch, config)
            exact = KFlipInterventionStrategy(n_samples=2**16).propose(
                model, batch, config
            )
            assert sampled.mask.sum(axis=1).min() > 12  # sampling was used
            truth = 1.0 - np.prod(
                np.where(sampled.mask, p.astype(np.float64), 1.0), axis=1
            )  # exact flip probability
            np.testing.assert_allclose(sampled.details["flip_prob"], truth, atol=0.03)
            # sampling may differ from exact search only over a concept whose gain is close to zero
            assert np.abs(sampled.mask.sum(axis=1) - exact.mask.sum(axis=1)).max() <= 1
            truth_exact = 1.0 - np.prod(
                np.where(exact.mask, p.astype(np.float64), 1.0), axis=1
            )
            np.testing.assert_allclose(truth, truth_exact, atol=0.02)
