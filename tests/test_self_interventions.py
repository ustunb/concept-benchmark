"""Interventions with the model's own answers (`--intervention-sources self`)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
robot_pipeline = pytest.importorskip("robot_pipeline")
from experiments.models import FrontEndModel  # noqa: E402
from tests.conftest import make_tabular_dataset  # noqa: E402


def test_own_answers_change_no_concept_and_no_prediction_of_a_hard_concept_cbm():
    dataset, _ = make_tabular_dataset(n=240, d=3, k=5, n_classes=2, with_cv=False)
    dataset.sample(test_size=0.5, val_size=0.2, stratify=dataset.y, seed=0)
    train, test = dataset.train, dataset.test
    frontend = FrontEndModel()
    frontend.fit(np.asarray(train.C, dtype=np.float32), np.asarray(train.y).astype(int))

    rng = np.random.default_rng(0)
    truth, y = np.asarray(test.C).astype(int), np.asarray(test.y).astype(int)
    proba = np.clip(
        np.where(truth == 1, 0.8, 0.2) + rng.normal(0, 0.25, size=truth.shape),
        0.01,
        0.99,
    ).astype(np.float32)
    baseline = float(
        (
            np.argmax(frontend.predict_proba((proba >= 0.5).astype(int)), axis=1) == y
        ).mean()
    )

    settings = robot_pipeline.InterventionSettings(
        seed=0,
        budgets=[1, truth.shape[1]],
        intervention_accuracy=0.8,
        intervention_threshold=0.05,
    )
    settings.intervention_expert = "self"
    settings.run_dir = ""
    _, _, results = robot_pipeline._test_interventions(
        prob_test=proba,
        settings=settings,
        acc_det=baseline,
        fe=frontend,
        test=test,
        cache_only=True,
    )
    assert len(results) == 2
    assert any(
        r["predictions_intervened_on"] > 0 for r in results.values()
    )  # the policy does select instances
    for r in results.values():
        assert r["total_concept_edits_made"] == 0
        assert r["accuracy"] == pytest.approx(baseline)


class _RecordingFrontEnd:
    """Label predictor that returns what it was asked to read."""

    def predict_proba(self, C):
        self.read = np.asarray(C, dtype=float)
        return np.column_stack([np.zeros(len(C)), np.ones(len(C))])


def _concepts_read_after_intervention(encoding, low_values=None, high_values=None):
    frontend = _RecordingFrontEnd()
    C_after = np.array([[1.0, 0.3, 0.7], [0.0, 0.6, 0.2]])
    mask = np.array([[True, False, False], [True, False, False]])
    robot_pipeline._predict_after_intervention(
        None,
        frontend,
        C_after,
        C_after,
        mask,
        supports_aligned=False,
        encoding=encoding,
        low_values=low_values,
        high_values=high_values,
    )
    return frontend.read


def test_binary_encoding_thresholds_every_concept():
    np.testing.assert_array_equal(
        _concepts_read_after_intervention("binary"), [[1, 0, 1], [0, 1, 0]]
    )


def test_binary_revealed_encoding_keeps_the_other_concepts_continuous():
    np.testing.assert_allclose(
        _concepts_read_after_intervention("binary_revealed"),
        [[1.0, 0.3, 0.7], [0.0, 0.6, 0.2]],
    )


def test_percentile_encoding_sets_revealed_concepts_to_training_percentiles():
    low, high = np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 0.95])
    read = _concepts_read_after_intervention("percentile", low, high)
    np.testing.assert_allclose(read, [[0.8, 0.3, 0.7], [0.1, 0.6, 0.2]])


def test_intervention_records_of_different_runs_get_different_names(tmp_path):
    result = SimpleNamespace(
        mask=np.zeros((2, 2), dtype=bool),
        C_pred=np.zeros((2, 2)),
        C_intervened=np.zeros((2, 2)),
        y_pred_after=np.zeros(2),
    )
    test = SimpleNamespace(C=np.zeros((2, 2)), y=np.zeros(2))
    for family, encoding in (
        ("cbm", "binary"),
        ("cem", "binary"),
        ("cbm", "percentile"),
    ):
        settings = robot_pipeline.InterventionSettings(
            seed=1, budgets=[1], model_family=family, encoding=encoding
        )
        robot_pipeline._save_intervention_records(
            tmp_path, settings, 1, ["a", "b"], result, test, np.zeros(2)
        )
    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_subjective_regime_runs_on_noisy_human_concepts():
    assert robot_pipeline._REGIME_TO_CELL["subjective"] == (
        "noisy_human_concepts",
        "expert",
    )
    assert "noisy_human_concepts" in robot_pipeline.CONCEPT_SOURCES
