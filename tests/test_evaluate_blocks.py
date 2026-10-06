"""The evaluation blocks of experiments.evaluate on a tiny tabular dataset."""

import numpy as np
import pytest
import torch
from torch import nn

from concept_benchmark.evaluation import intervention_metrics
from concept_benchmark.evaluation import abstention_threshold, selective_kept
from experiments.evaluate import (
    ROBOT_BUDGET_COLUMNS,
    automation_table,
    coverage_at_target,
    intervene,
    intervention_table,
    predict_labels,
    predict_proba_positive,
    train_cbm,
    train_dnn,
)
from experiments.intervention import ConceptInterventionRunner
from tests.conftest import make_tabular_dataset

D, K = 3, 4


@pytest.fixture(scope="module")
def splits():
    dataset, _ = make_tabular_dataset(n=200, d=D, k=K, n_classes=2, with_cv=False)
    dataset.sample(test_size=0.25, val_size=0.25, stratify=dataset.y, seed=0)
    return dataset.train, dataset.val, dataset.test


@pytest.fixture(scope="module")
def cbm(splits):
    train, val, _ = splits
    return train_cbm(train, val, detector=lambda: nn.Linear(D, K), epochs=2, seed=0)


def test_intervention_table_has_the_columns_of_the_pipeline(cbm, splits):
    table = intervention_table(cbm, splits[2], budgets=(1, "max"), seed=0)
    assert list(table.columns) == ROBOT_BUDGET_COLUMNS
    assert list(table["budget"]) == [0, 1, K]
    assert all(table[column].dtype.kind == "i" for column in ROBOT_BUDGET_COLUMNS[2:])


def test_intervention_table_is_repeatable_for_a_seed(cbm, splits):
    first = intervention_table(cbm, splits[2], budgets=(1,), seed=3)
    assert first.equals(intervention_table(cbm, splits[2], budgets=(1,), seed=3))


def test_train_dnn_builds_the_model_after_seeding(splits):
    train, val, _ = splits

    def make():
        return nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())

    first = train_dnn(make, train, val, epochs=1, seed=5)
    second = train_dnn(make, train, val, epochs=1, seed=5)
    assert torch.equal(first[0].weight, second[0].weight)


def test_coverage_at_target_returns_nan_and_zero_when_out_of_reach(cbm, splits):
    _, val, test = splits
    selective_accuracy, coverage = coverage_at_target(cbm, val, test, 1.01)
    assert np.isnan(selective_accuracy) and coverage == 0.0


def test_intervention_metrics_counts_edits_confirmations_and_changed_predictions():
    mask = np.array([[True, True, False], [False, False, False]])
    before = np.array([[0.9, 0.2, 0.6], [0.1, 0.8, 0.3]])
    after = np.array([[0.9, 1.0, 0.6], [0.1, 0.8, 0.3]])  # one answer edits a concept
    y_prob_before = np.array([[0.6, 0.4], [0.3, 0.7]])
    y_prob_after = np.array([[0.2, 0.8], [0.3, 0.7]])  # the first prediction flips
    metrics = intervention_metrics(
        mask, before, after, y_prob_before, y_prob_after, np.array([1, 1]), 0.5
    )
    assert metrics == {
        "accuracy": 1.0,
        "accuracy_gain": 0.5,
        "predictions_intervened_on": 1,
        "interventions_rate": 0.5,
        "predictions_changed": 1,
        "avg_edits_per_intervention": 1.0,  # per intervened-on prediction
        "total_concept_confirmations": 2,
        "total_concept_edits_made": 1,
    }


def test_a_wrong_intervener_flips_every_concept_it_is_asked_about(cbm, splits):
    test = splits[2]
    concept_proba = np.asarray(cbm.concept_detector.predict_proba(test))
    runner = ConceptInterventionRunner(cbm)
    wrong = intervene(
        runner,
        test,
        concept_proba,
        K,
        rng=np.random.default_rng(0),
        seed=0,
        intervention_accuracy=0.0,
    )
    truth = test.C.astype(np.float32)
    assert wrong.mask.any()
    assert np.array_equal(wrong.C_intervened[wrong.mask], 1 - truth[wrong.mask])
    assert np.array_equal(wrong.C_intervened[~wrong.mask], wrong.C_pred[~wrong.mask])


def test_automation_table_rows_and_fallback(cbm, splits):
    _, val, test = splits
    table = automation_table(
        cbm, val, test, budgets=(1, "max"), target_accuracy=0.0, seed=0
    )
    assert list(table["budget"]) == [0, 1, K]
    assert "abstention_threshold" in table.columns
    assert table["coverage_after"].between(0.0, 1.0).all()
    out_of_reach = automation_table(
        cbm, val, test, budgets=(1,), target_accuracy=1.01, seed=0
    )
    assert list(out_of_reach["budget"]) == [0, 1]
    assert out_of_reach["coverage_after"].eq(0.0).all()
    assert out_of_reach["selective_accuracy_after"].isna().all()


def test_automation_table_is_repeatable_for_a_seed(cbm, splits):
    _, val, test = splits
    first = automation_table(cbm, val, test, budgets=(1,), target_accuracy=0.0, seed=2)
    assert first.equals(
        automation_table(cbm, val, test, budgets=(1,), target_accuracy=0.0, seed=2)
    )


def test_budgets_start_at_one_and_are_capped(cbm, splits):
    test = splits[2]
    assert list(intervention_table(cbm, test, budgets=(10,), seed=0)["budget"]) == [
        0,
        K,
    ]
    with pytest.raises(ValueError, match="at least 1"):
        intervention_table(cbm, test, budgets=(0,), seed=0)


def test_dnn_predictions_are_probabilities_and_labels(splits):
    train, val, test = splits
    dnn = train_dnn(
        lambda: nn.Sequential(nn.Linear(D, 1), nn.Sigmoid()),
        train,
        val,
        epochs=1,
        seed=0,
    )
    proba = predict_proba_positive(dnn, test)
    assert proba.shape == (test.n,) and ((0 <= proba) & (proba <= 1)).all()
    assert set(predict_labels(dnn, test)) <= {0, 1}


def test_train_dnn_accepts_a_last_batch_of_one(splits):
    train, val, _ = splits
    loader_config = {"batch_size": train.n - 1, "num_workers": 0, "pin_memory": False}
    train_dnn(
        lambda: nn.Sequential(nn.Linear(D, 1), nn.Sigmoid()),
        train,
        val,
        epochs=1,
        seed=0,
        loader_config=loader_config,
    )


def test_selective_kept_matches_the_fitted_coverage():
    y = np.array([1, 1, 1, 0, 0, 1])
    prob = np.array([0.99, 0.95, 0.9, 0.05, 0.6, 0.45])
    threshold, fitted_coverage = abstention_threshold(y, prob, 1.0)
    accuracy, coverage = selective_kept(y, prob, threshold, 0.5)
    assert (accuracy, coverage) == (1.0, pytest.approx(fitted_coverage))
    assert selective_kept(y, prob, None) == (None, 0.0)


class _UnderconfidentAnd:
    """A sudoku-like model: the label is the AND of the concepts, the detector is right but only 80% sure.

    Propagating 6 concepts at 0.8 gives P ~ 0.26 for every positive, so Platt scaling moves the label
    probabilities a lot; a scaling applied twice to unchecked boards would flip their predictions.
    """

    class _Detector:
        n_concepts = 6

        def predict_proba(self, dataset):
            # 80% sure about every concept, and plainly wrong about one concept of a tenth of the rows
            C = np.asarray(dataset.C)
            p = np.where(C > 0.5, 0.8, 0.2)
            rng = np.random.default_rng(len(C))
            wrong = rng.random(len(C)) < 0.1
            p[wrong, rng.integers(0, 6, len(C))[wrong]] = 0.2
            return p

    def __init__(self):
        self.concept_detector = self._Detector()

    def predict_proba_from_concepts(self, C):
        p = np.prod(np.asarray(C, dtype=float), axis=1)
        return np.column_stack([1 - p, p])


def _and_splits(n=400, seed=0):
    from concept_benchmark.data import ConceptDataset

    rng = np.random.default_rng(seed)
    C = np.ones((n, 6), dtype=np.int32)
    negatives = rng.random(n) < 0.5
    C[negatives, rng.integers(0, 6, n)[negatives]] = 0
    y = C.all(axis=1).astype(int)
    dataset = ConceptDataset(
        inputs=rng.random((n, 3)).astype(np.float32),
        C=C,
        y=y,
        meta={"concepts": [f"c{i}" for i in range(6)]},
        input_type="tabular",
        classes=(0, 1),
    )
    dataset.sample(test_size=0.25, val_size=0.25, stratify=dataset.y, seed=seed)
    return dataset.val, dataset.test


def test_only_checked_boards_can_change_their_prediction():
    val, test = _and_splits()
    table = automation_table(
        _UnderconfidentAnd(),
        val,
        test,
        budgets=(1, "max"),
        target_accuracy=0.95,
        seed=0,
    )
    accuracy_0 = table["accuracy"].iloc[0]
    assert (
        table["coverage_after"].iloc[0] > 0.8
    )  # calibration makes the positives confident
    # only the checked boards (the abstained ones) can change: coverage never drops and at most that
    # many predictions change
    assert table["coverage_after"].is_monotonic_increasing
    for _, row in table.iloc[1:].iterrows():
        changed = round(abs(row["accuracy"] - accuracy_0) * test.n)
        assert changed <= row["predictions_intervened_on"]
