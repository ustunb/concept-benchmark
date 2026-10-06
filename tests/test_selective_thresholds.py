"""Thresholds for selective classification in the sudoku pipeline."""

import numpy as np
import pytest

from concept_benchmark.evaluation import (
    abstention_mask,
    abstention_threshold,
    classwise_thresholds,
    decision_threshold,
    selective_at,
    selective_at_classwise,
)


def _two_level_model(n_valid=50, n_invalid=50, n_wrong_low=0, p_valid=0.9, rng=None):
    """Probabilities of a model that gives every valid-looking board one value, as the sudoku models do."""
    rng = rng or np.random.default_rng(0)
    y = np.array([1] * n_valid + [0] * n_invalid)
    p = np.where(y == 1, p_valid, rng.uniform(0.0, 0.05, size=y.size))
    p[:n_wrong_low] = 0.02  # valid boards that the model rejects
    return y, p


@pytest.mark.parametrize("n_wrong_low", [0, 2, 10])
@pytest.mark.parametrize("p_valid", [0.6, 0.9, 0.97])
def test_fitted_threshold_keeps_what_the_fit_kept(n_wrong_low, p_valid):
    y, p = _two_level_model(n_wrong_low=n_wrong_low, p_valid=p_valid)
    decision, _ = decision_threshold(y, p)
    t, fitted_coverage = abstention_threshold(y, p, 0.95, decision)
    assert t is not None
    accuracy, coverage = selective_at(y, p, t, decision)
    assert coverage == pytest.approx(
        fitted_coverage
    )  # ties at the threshold are kept by both rules
    assert accuracy >= 0.95


def test_fitted_threshold_never_keeps_the_next_confidence_level():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    p = np.array(
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2e-7, 4e-7]
    )  # last board is wrong and nearly as confident
    t, fitted_coverage = abstention_threshold(y, p, 1.0, 0.5)
    _, coverage = selective_at(y, p, t, 0.5)
    assert coverage == pytest.approx(fitted_coverage) == pytest.approx(0.9)


def test_classwise_thresholds_keep_the_clean_class():
    # 40 valid boards accepted at 0.9; 10 valid boards wrongly rejected at 0.02; 50 invalid boards rejected at 0.001.
    # The wrong rejections are more confident than the accepted boards (0.02 from 0 vs 0.1 from 1).
    y = np.array([1] * 50 + [0] * 50)
    p = np.array([0.9] * 40 + [0.02] * 10 + [0.001] * 50)
    decision = 0.5
    single_t, _ = abstention_threshold(y, p, 0.95, decision)
    _, single_coverage = selective_at(y, p, single_t, decision)
    t_pos, t_neg = classwise_thresholds(y, p, 0.95, decision)
    accuracy, coverage = selective_at_classwise(y, p, t_pos, t_neg, decision)
    assert single_coverage == pytest.approx(
        0.5
    )  # one threshold must also defer every accepted board
    assert coverage == pytest.approx(
        0.9
    )  # per class: only the wrong rejections are deferred
    assert accuracy == pytest.approx(1.0)


def test_classwise_thresholds_return_none_for_a_side_that_cannot_reach_the_target():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array(
        [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    )  # called valid: 50% right; called invalid: 0% right
    t_pos, t_neg = classwise_thresholds(y, p, 0.95, 0.5)
    assert t_pos is None and t_neg is None
    assert selective_at_classwise(y, p, t_pos, t_neg, 0.5)[1] == 0.0


def test_classwise_threshold_keeps_the_predictions_it_was_fitted_on():
    # 1 - (1 - 0.1) is not 0.1 in floating point: the threshold must be the probability itself
    y = np.array([0, 0, 0, 0, 0, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    t_pos, t_neg = classwise_thresholds(y, p, target_accuracy=1.0)
    accuracy, coverage = selective_at_classwise(y, p, t_pos, t_neg, 0.5)
    assert t_neg == 0.1
    assert accuracy == 1.0
    assert coverage == pytest.approx(5 / 6)


def test_classwise_fit_and_evaluation_agree_on_random_probabilities():
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = rng.integers(0, 2, size=40)
        p = np.round(rng.random(40), 2)
        t_pos, t_neg = classwise_thresholds(y, p, target_accuracy=0.9)
        accuracy, coverage = selective_at_classwise(y, p, t_pos, t_neg, 0.5)
        if coverage > 0:
            assert accuracy >= 0.9 - 1e-12


def test_abstention_mask_agrees_with_the_fit_on_random_probabilities():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=400)
    p = np.round(rng.uniform(0.0, 1.0, size=400), 3)
    t, coverage = abstention_threshold(y, p, target_accuracy=0.0)
    assert coverage == 1.0
    _, coverage_at_t = selective_at(y, p, t, 0.5)
    assert coverage_at_t == coverage
    assert not abstention_mask(p, t).any()
    assert np.array_equal(abstention_mask(p, t), abstention_mask(np.column_stack([1 - p, p]), t))


def test_abstention_mask_uses_the_same_margin_for_both_classes():
    # 1 - (1 - t) is not t in floating point; the margin rule must not depend on the class
    t = 0.3
    p = np.array([t, 1 - t, t - 1e-12, 1 - t + 1e-12])
    assert abstention_mask(p, t).tolist() == [True, True, False, False]


def test_abstention_mask_multiclass_uses_the_top_probability():
    y_prob = np.array([[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]])
    assert abstention_mask(y_prob, 0.3).tolist() == [True, False]


def test_decision_threshold_returns_the_accuracy_at_the_returned_threshold():
    # best accuracy at t in {0.0..0.1} and at t in {0.9..1.0}; the midpoint of all of them (0.5) is worse
    y = np.array([1, 1, 1, 0])
    p = np.array([0.95, 0.15, 0.15, 0.5])
    t, acc = decision_threshold(y, p)
    assert acc == ((p >= t).astype(int) == y).mean()
    assert acc == 0.75
    assert t <= 0.15


def test_decision_threshold_prefers_the_longest_run_of_tied_thresholds():
    y = np.array([1, 0])
    p = np.array([0.9, 0.2])
    t, acc = decision_threshold(y, p, thresholds=np.array([0.1, 0.3, 0.4, 0.5, 0.95]))
    assert (t, acc) == (0.4, 1.0)
