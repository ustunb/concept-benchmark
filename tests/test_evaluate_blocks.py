"""The evaluation blocks of experiments.evaluate on a tiny tabular dataset."""

import numpy as np
import pytest
import torch
from torch import nn

from concept_benchmark.evaluation import intervention_metrics
from experiments.evaluate import (
    ROBOT_BUDGET_COLUMNS,
    coverage_at_target,
    intervention_table,
    train_cbm,
    train_dnn,
)
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
        "avg_edits_per_intervention": 0.5,
        "total_concept_confirmations": 2,
        "total_concept_edits_made": 1,
    }
