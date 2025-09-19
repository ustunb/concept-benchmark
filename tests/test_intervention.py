from __future__ import annotations

import numpy as np
import pytest

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionBatch,
    InterventionConfig,
    InterventionError,
    OrderedCBMStrategy,
    RandomInterventionStrategy,
)
from concept_benchmark.models import ConceptBasedModel, ConceptDetector, FrontEndModel


def _make_sample(concepts: np.ndarray, labels: np.ndarray) -> ConceptDatasetSample:
    n_samples, n_concepts = concepts.shape
    X = np.zeros((n_samples, 1), dtype=np.float32)
    meta = {
        "classes": ["y0", "y1"],
        "concepts": [f"c{i}" for i in range(n_concepts)],
        "data_type": "tabular",
    }
    return ConceptDatasetSample(X=X, C=concepts.astype(np.int8), y=labels.astype(np.int32), meta=meta)


class LinearFrontEndModel(FrontEndModel):
    def __init__(self, weights: np.ndarray, bias: float = 0.0) -> None:
        super().__init__()
        self._weights = np.asarray(weights, dtype=np.float64)
        self._bias = float(bias)

    def fit(self, C: np.ndarray, y: np.ndarray, fit_params=None) -> None:  # pragma: no cover - unused
        return None

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        logits = C @ self._weights + self._bias
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        probs_pos = probs_pos.reshape(-1)
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)


class RecordingConceptDetector(ConceptDetector):
    def __init__(self, predictions: np.ndarray) -> None:
        super().__init__(embedding_model=None, concept_layers=None)
        self._predictions = predictions
        self.calls = 0

    def predict(self, dataset, embed_params=None, calibrate=None):  # type: ignore[override]
        self.calls += 1
        return self._predictions


def _expected_random_mask(config: InterventionConfig, n_samples: int, n_concepts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(config.random_state)
    candidates = np.arange(n_samples)
    rng.shuffle(candidates)
    limit_instances = config.resolve_instance_budget(n_samples)
    selected = candidates[:limit_instances]

    mask = np.zeros((n_samples, n_concepts), dtype=bool)
    total_limit = config.resolve_concept_budget(mask.size)
    total_applied = 0
    per_instance_limit = config.per_instance_limit(n_concepts)

    if config.per_instance_ordering:
        for idx in selected:
            order = rng.permutation(n_concepts)
            for concept in order:
                if total_applied >= total_limit:
                    return mask, selected, None
                if mask[idx, concept]:
                    continue
                if np.sum(mask[idx]) >= per_instance_limit:
                    break
                mask[idx, concept] = True
                total_applied += 1
        return mask, selected, None

    order = rng.permutation(n_concepts)
    for idx in selected:
        for concept in order:
            if total_applied >= total_limit:
                return mask, selected, order
            if mask[idx, concept]:
                continue
            if np.sum(mask[idx]) >= per_instance_limit:
                break
            mask[idx, concept] = True
            total_applied += 1
    return mask, selected, order


def test_intervention_config_budget_resolution():
    config = InterventionConfig(concept_budget=0.25, instance_budget=2, max_concepts_per_instance=1)
    assert config.resolve_concept_budget(12) == 3
    assert config.resolve_instance_budget(5) == 2
    assert config.per_instance_limit(4) == 1

    config = InterventionConfig(concept_budget=3, instance_budget=None, max_concepts_per_instance=None)
    assert config.resolve_concept_budget(10) == 3
    assert config.resolve_instance_budget(8) == 8
    assert config.per_instance_limit(4) == 4

    with pytest.raises(ValueError):
        InterventionConfig(concept_budget=1.5).resolve_concept_budget(-1)


def test_conceptual_safeguards_intervenes_only_abstaining():
    C_true = np.array([[1, 0], [1, 0]], dtype=np.int8)
    C_pred = np.array([[0, 0], [1, 0]], dtype=np.int8)
    y_true = np.array([0, 1], dtype=np.int32)
    dataset = _make_sample(C_true, y_true)

    model = ConceptBasedModel(front_end_model=LinearFrontEndModel(weights=np.array([2.0, 0.0]), bias=0.0))
    runner = ConceptInterventionRunner(model)

    config = InterventionConfig(tau=0.2, concept_order=[0, 1], max_concepts_per_instance=1)
    strategy = ConceptualSafeguardsStrategy()
    result = runner.run(strategy, config, dataset, concept_proba=C_pred, concept_true=C_true, labels=y_true)

    assert np.array_equal(result.proposal.selected_instances, np.array([0]))
    assert result.mask[0, 0]
    assert not result.mask[1].any()
    assert result.y_prob_before[0, 1] != result.y_prob_after[0, 1]
    np.testing.assert_array_equal(result.y_prob_before[1], result.y_prob_after[1])


def test_ordered_cbm_prepare_produces_expected_ordering():
    C_true = np.array([[1, 0], [1, 1], [1, 1]], dtype=np.int8)
    C_pred = np.array([[0, 0], [1, 0], [1, 0]], dtype=np.int8)
    y_true = np.array([0, 1, 1], dtype=np.int32)
    dataset = _make_sample(C_true, y_true)

    model = ConceptBasedModel(front_end_model=LinearFrontEndModel(weights=np.array([0.0, 2.0]), bias=-0.5))
    runner = ConceptInterventionRunner(model)

    strategy = OrderedCBMStrategy()
    config = InterventionConfig()
    runner.prepare(strategy, config, dataset, concept_proba=C_pred, concept_true=C_true, labels=y_true)

    ordering = strategy.state["ordering"]
    assert np.array_equal(ordering, np.array([1, 0]))
    error_deltas = strategy.state["error_deltas"]
    assert error_deltas.shape == C_pred.shape


def test_ordered_cbm_propose_uses_prepared_order_and_budgets():
    C_true = np.array([[1, 0], [1, 1], [1, 1]], dtype=np.int8)
    C_pred = np.array([[0, 0], [1, 0], [1, 0]], dtype=np.int8)
    y_true = np.array([0, 1, 1], dtype=np.int32)
    dataset = _make_sample(C_true, y_true)

    model = ConceptBasedModel(front_end_model=LinearFrontEndModel(weights=np.array([0.0, 2.0]), bias=-0.5))
    runner = ConceptInterventionRunner(model)

    strategy = OrderedCBMStrategy()
    config = InterventionConfig(concept_budget=2, max_concepts_per_instance=1)
    runner.prepare(strategy, config, dataset, concept_proba=C_pred, concept_true=C_true, labels=y_true)

    result = runner.run(strategy, config, dataset, concept_proba=C_pred, concept_true=C_true, labels=y_true)

    assert result.mask.sum() == 2
    assert np.array_equal(result.proposal.ordering_used, np.array([1, 0]))
    assert set(result.proposal.selected_instances.tolist()) == {0, 1, 2}
    assert "validation_error_deltas" in result.proposal.details


def test_ordered_cbm_requires_prepare_or_ordering():
    C_true = np.array([[1, 0]], dtype=np.int8)
    C_pred = np.array([[0, 0]], dtype=np.int8)
    y_true = np.array([0], dtype=np.int32)
    dataset = _make_sample(C_true, y_true)

    model = ConceptBasedModel(front_end_model=LinearFrontEndModel(weights=np.array([1.0, 1.0]), bias=0.0))
    runner = ConceptInterventionRunner(model)

    strategy = OrderedCBMStrategy()
    config = InterventionConfig()

    with pytest.raises(InterventionError):
        runner.run(strategy, config, dataset, concept_proba=C_pred, concept_true=C_true, labels=y_true)


def test_random_strategy_global_order_respects_budgets():
    n_samples, n_concepts = 4, 3
    C_true = np.zeros((n_samples, n_concepts), dtype=np.int8)
    C_pred = np.zeros((n_samples, n_concepts), dtype=np.int8)
    y_true = np.zeros(n_samples, dtype=np.int32)
    dataset = _make_sample(C_true, y_true)

    model = ConceptBasedModel(front_end_model=LinearFrontEndModel(weights=np.array([0.0, 0.0, 0.0]), bias=0.0))
    runner = ConceptInterventionRunner(model)

    config = InterventionConfig(
        concept_budget=2,
        instance_budget=2,
        max_concepts_per_instance=1,
        random_state=42,
        per_instance_ordering=False,
    )
    strategy = RandomInterventionStrategy()
    result = runner.run(strategy, config, dataset, concept_proba=C_pred, concept_true=C_true, labels=y_true)

    expected_mask, expected_selected, expected_order = _expected_random_mask(config, n_samples, n_concepts)
    np.testing.assert_array_equal(result.mask, expected_mask)
    np.testing.assert_array_equal(result.proposal.selected_instances, expected_selected)
    np.testing.assert_array_equal(result.proposal.ordering_used, expected_order)


def test_random_strategy_per_instance_ordering_limits_total():
    n_samples, n_concepts = 4, 3
    C_true = np.zeros((n_samples, n_concepts), dtype=np.int8)
    C_pred = np.zeros((n_samples, n_concepts), dtype=np.int8)
    y_true = np.zeros(n_samples, dtype=np.int32)
    dataset = _make_sample(C_true, y_true)

    model = ConceptBasedModel(front_end_model=LinearFrontEndModel(weights=np.array([0.0, 0.0, 0.0]), bias=0.0))
    runner = ConceptInterventionRunner(model)

    config = InterventionConfig(
        concept_budget=3,
        instance_budget=3,
        max_concepts_per_instance=1,
        random_state=12,
        per_instance_ordering=True,
    )
    strategy = RandomInterventionStrategy()
    result = runner.run(strategy, config, dataset, concept_proba=C_pred, concept_true=C_true, labels=y_true)

    assert result.mask.sum() <= config.resolve_concept_budget(n_samples * n_concepts)
    assert result.proposal.ordering_used is None
    assert (result.mask.sum(axis=1) <= 1).all()


def test_runner_uses_concept_detector_when_predictions_missing():
    C_true = np.array([[0, 1], [1, 0]], dtype=np.int8)
    predicted = np.array([[1, 1], [0, 0]], dtype=np.int8)
    y_true = np.array([1, 0], dtype=np.int32)
    dataset = _make_sample(C_true, y_true)

    detector = RecordingConceptDetector(predictions=predicted)
    model = ConceptBasedModel(concept_detector=detector, front_end_model=LinearFrontEndModel(weights=np.array([1.0, -1.0]), bias=0.0))
    runner = ConceptInterventionRunner(model)

    strategy = RandomInterventionStrategy()
    config = InterventionConfig(concept_budget=0)

    result = runner.run(strategy, config, dataset, concept_true=C_true, labels=y_true)

    assert detector.calls == 1
    np.testing.assert_array_equal(result.C_pred, predicted)
    np.testing.assert_array_equal(result.C_intervened, predicted)


def test_intervention_batch_validates_shapes():
    C_pred = np.zeros((2, 2))
    C_true = np.zeros((1, 2))
    with pytest.raises(ValueError):
        InterventionBatch(C_pred=C_pred, C_true=C_true)

    with pytest.raises(ValueError):
        InterventionBatch(C_pred=np.zeros((2, 2)), C_true=np.zeros((2, 2)), instance_ids=np.array([0]))
