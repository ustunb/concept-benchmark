"""Interventions with the model's own answers (`--intervention-sources self`)."""

import sys
from pathlib import Path

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
    proba = np.clip(np.where(truth == 1, 0.8, 0.2) + rng.normal(0, 0.25, size=truth.shape), 0.01, 0.99).astype(np.float32)
    baseline = float((np.argmax(frontend.predict_proba((proba >= 0.5).astype(int)), axis=1) == y).mean())

    settings = robot_pipeline.InterventionSettings(
        seed=0, budgets=[1, truth.shape[1]], intervention_accuracy=0.8, intervention_threshold=0.05
    )
    settings.intervention_expert = "self"
    settings.run_dir = ""
    _, _, results = robot_pipeline._test_interventions(
        prob_test=proba, settings=settings, acc_det=baseline, fe=frontend, test=test, cache_only=True
    )
    assert len(results) == 2
    assert any(r["predictions_intervened_on"] > 0 for r in results.values())  # the policy does select instances
    for r in results.values():
        assert r["total_concept_edits_made"] == 0
        assert r["accuracy"] == pytest.approx(baseline)
