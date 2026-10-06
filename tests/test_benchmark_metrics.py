"""Tests for concept_benchmark.evaluation.metrics."""

import numpy as np
import pytest

from concept_benchmark.evaluation.metrics import (
    abstention_threshold,
    decision_threshold,
    selective_at,
    accuracy,
    coverage,
    delta_accuracy,
    gain,
    net_work_automated,
    selective_accuracy,
)


class TestAccuracy:
    def test_perfect(self):
        assert accuracy(np.array([0, 1, 1]), np.array([0, 1, 1])) == 1.0

    def test_all_wrong(self):
        assert accuracy(np.array([0, 0]), np.array([1, 1])) == 0.0

    def test_partial(self):
        assert accuracy(np.array([0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 1])) == 0.8

    def test_empty(self):
        assert np.isnan(accuracy(np.array([]), np.array([])))


class TestDeltaAccuracy:
    def test_improvement(self):
        y_true = np.array([0, 1, 1, 0])
        before = np.array([0, 0, 0, 0])
        after = np.array([0, 1, 1, 0])
        assert delta_accuracy(after, before, y_true) == pytest.approx(0.5)

    def test_no_change(self):
        y = np.array([0, 1])
        assert delta_accuracy(y, y, y) == 0.0

    def test_degradation(self):
        y_true = np.array([0, 1])
        before = np.array([0, 1])
        after = np.array([1, 0])
        assert delta_accuracy(after, before, y_true) == pytest.approx(-1.0)


class TestGain:
    def test_positive_gain(self):
        y_pred = np.array([0, 1, 1, 0, 1])
        y_true = np.array([0, 1, 0, 0, 1])
        assert gain(y_pred, y_true, baseline_accuracy=0.5) == pytest.approx(0.3)

    def test_negative_gain(self):
        y_pred = np.array([1, 1, 1, 1])
        y_true = np.array([0, 0, 0, 0])
        assert gain(y_pred, y_true, baseline_accuracy=0.5) == pytest.approx(-0.5)

    def test_zero_gain(self):
        y_pred = np.array([0, 1])
        y_true = np.array([0, 1])
        assert gain(y_pred, y_true, baseline_accuracy=1.0) == pytest.approx(0.0)


class TestSelectiveAccuracy:
    def test_all_confident(self):
        y_pred = np.array([0, 1, 1])
        y_true = np.array([0, 1, 0])
        confidence = np.array([0.9, 0.8, 0.7])
        assert selective_accuracy(
            y_pred, y_true, confidence, threshold=0.5
        ) == pytest.approx(2 / 3)

    def test_some_abstain(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        confidence = np.array([0.9, 0.8, 0.3, 0.7])
        # Only indices 0, 1, 3 kept (confidence >= 0.5). All correct.
        assert selective_accuracy(
            y_pred, y_true, confidence, threshold=0.5
        ) == pytest.approx(1.0)

    def test_all_abstain(self):
        confidence = np.array([0.1, 0.2])
        assert np.isnan(
            selective_accuracy(
                np.array([0, 1]), np.array([0, 1]), confidence, threshold=0.9
            )
        )


class TestCoverage:
    def test_all_kept(self):
        assert coverage(np.array([0.9, 0.8, 0.7]), threshold=0.5) == 1.0

    def test_some_abstain(self):
        assert coverage(np.array([0.9, 0.3, 0.7, 0.2]), threshold=0.5) == 0.5

    def test_none_kept(self):
        assert coverage(np.array([0.1, 0.2]), threshold=0.5) == 0.0

    def test_empty(self):
        assert np.isnan(coverage(np.array([]), threshold=0.5))


class TestNetWorkAutomated:
    def test_no_interventions(self):
        confidence = np.array([0.9, 0.8, 0.7, 0.6])
        n_interventions = np.array([0, 0, 0, 0])
        result = net_work_automated(
            confidence, threshold=0.5, n_interventions=n_interventions, n_concepts=10
        )
        assert result == pytest.approx(1.0)  # full coverage, zero cost

    def test_with_interventions(self):
        confidence = np.array([0.9, 0.8, 0.7, 0.6])
        n_interventions = np.array([2, 3, 1, 2])
        result = net_work_automated(
            confidence, threshold=0.5, n_interventions=n_interventions, n_concepts=10
        )
        assert result == pytest.approx(1.0 - 2.0 / 10)  # coverage=1, avg_cost=2/10


def test_abstention_threshold_keeps_the_confident_predictions_that_reach_the_target():
    y_true = np.array([1, 1, 1, 0, 0, 1])
    prob = np.array(
        [0.99, 0.95, 0.9, 0.05, 0.6, 0.45]
    )  # the two least confident are wrong
    threshold, kept = abstention_threshold(y_true, prob, target_accuracy=1.0)
    assert kept == pytest.approx(4 / 6)
    assert selective_at(y_true, prob, threshold, 0.5) == (1.0, pytest.approx(4 / 6))


def test_abstention_threshold_is_none_when_the_target_is_out_of_reach():
    assert abstention_threshold([0, 1], [0.9, 0.1], target_accuracy=0.5) == (None, None)


def test_decision_threshold_is_the_midpoint_of_the_best_cuts():
    threshold, accuracy = decision_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert accuracy == 1.0
    assert threshold == pytest.approx(
        0.505
    )  # the cuts 0.21 to 0.80 of the 0.01 grid tie
