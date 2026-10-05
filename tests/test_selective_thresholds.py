"""Thresholds for selective classification in the sudoku pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sudoku_pipeline = pytest.importorskip("sudoku_pipeline")


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
    decision, _ = sudoku_pipeline._decision_threshold_sweep(y, p)
    t, fitted_coverage = sudoku_pipeline._selective_accuracy_threshold(
        y, p, 0.95, decision
    )
    assert t is not None
    accuracy, coverage = sudoku_pipeline._selective_at_thresholds(y, p, t, decision)
    assert coverage == pytest.approx(
        fitted_coverage
    )  # ties at the threshold are kept by both rules
    assert accuracy >= 0.95


def test_fitted_threshold_never_keeps_the_next_confidence_level():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    p = np.array(
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2e-7, 4e-7]
    )  # last board is wrong and nearly as confident
    t, fitted_coverage = sudoku_pipeline._selective_accuracy_threshold(y, p, 1.0, 0.5)
    _, coverage = sudoku_pipeline._selective_at_thresholds(y, p, t, 0.5)
    assert coverage == pytest.approx(fitted_coverage) == pytest.approx(0.9)


def test_classwise_thresholds_keep_the_clean_class():
    # 40 valid boards accepted at 0.9; 10 valid boards wrongly rejected at 0.02; 50 invalid boards rejected at 0.001.
    # The wrong rejections are more confident than the accepted boards (0.02 from 0 vs 0.1 from 1).
    y = np.array([1] * 50 + [0] * 50)
    p = np.array([0.9] * 40 + [0.02] * 10 + [0.001] * 50)
    decision = 0.5
    single_t, _ = sudoku_pipeline._selective_accuracy_threshold(y, p, 0.95, decision)
    _, single_coverage = sudoku_pipeline._selective_at_thresholds(
        y, p, single_t, decision
    )
    t_pos, t_neg = sudoku_pipeline._classwise_accuracy_thresholds(y, p, 0.95, decision)
    accuracy, coverage = sudoku_pipeline._selective_at_classwise_thresholds(
        y, p, t_pos, t_neg, decision
    )
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
    t_pos, t_neg = sudoku_pipeline._classwise_accuracy_thresholds(y, p, 0.95, 0.5)
    assert t_pos is None and t_neg is None
    assert (
        sudoku_pipeline._selective_at_classwise_thresholds(y, p, t_pos, t_neg, 0.5)[1]
        == 0.0
    )
