"""Fast mini-pipeline integration tests (no images, no OCR, CPU only).

These exercise real pipeline code paths with tiny datasets and 1-epoch
training. NOT marked slow — expected wall time ~3s total.
"""

from __future__ import annotations

import numpy as np

from concept_benchmark.data import ConceptDataset
from concept_benchmark.models import (
    ConceptBasedModel,
    ConceptDetector,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _tabular_dataset(n=60, k=4, seed=42, concept_names=None):
    """Generate a tiny tabular dataset with CV splits.

    Uses balanced labels and enough samples to guarantee both classes
    appear in every fold.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 8)).astype(np.float32)
    C = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    # Balanced labels: alternate 0 and 1, then shuffle
    y = np.tile([0, 1], n // 2 + 1)[:n].astype(np.int32)
    rng.shuffle(y)
    if concept_names is None:
        concept_names = [f"concept_{i}" for i in range(k)]
    meta = {
        "classes": ["drent", "glorp"],
        "concepts": concept_names,
        "data_type": "tabular",
    }
    ds = ConceptDataset(X=X, C=C, y=y, meta=meta)
    ds.generate_cvindices(seed=seed)
    ds.split("K05N01", fold_num_validation=4, fold_num_test=5)
    return ds


def _train_cbm(ds, epochs=1):
    """Train a 1-epoch CBM on tabular data."""
    cd = ConceptDetector()
    cbm = ConceptBasedModel(concept_detector=cd)
    cbm.fit(
        train_dataset=ds.training,
        valid_dataset=ds.validation,
        freeze=False,
        concept_embed_params={"device": "cpu", "batch_size": 8, "num_workers": 0},
        fit_params={
            "epochs": epochs,
            "lr": 1e-3,
            "patience": 1,
            "device": "cpu",
            "batch_size": 8,
            "num_workers": 0,
        },
    )
    return cbm


# ── Tests ────────────────────────────────────────────────────────────


def test_robot_tabular_end_to_end():
    """Generate tabular robots, split, train CBM 1 epoch, predict."""
    ds = _tabular_dataset(n=40, k=4)
    cbm = _train_cbm(ds)

    preds = cbm.predict(ds.test)
    assert len(preds) == ds.test.n
    assert set(preds).issubset({0, 1})

    concept_preds = cbm.concept_detector.predict(ds.test)
    assert concept_preds.shape == (ds.test.n, ds.test.n_concepts)


def test_robot_intervention_end_to_end():
    """Train CBM + KFlip k=1 intervention."""
    from concept_benchmark.intervention import (
        ConceptInterventionRunner,
        InterventionConfig,
    )
    from concept_benchmark.kflip import KFlipInterventionStrategy

    ds = _tabular_dataset(n=40, k=4)
    cbm = _train_cbm(ds)

    runner = ConceptInterventionRunner(cbm)
    config = InterventionConfig(
        max_concepts_per_instance=1,
        score_threshold=0.1,
        random_state=0,
    )
    strat = KFlipInterventionStrategy()
    result = runner.run(strat, config, ds.test)

    assert result.C_pred.shape == (ds.test.n, ds.test.n_concepts)
    assert result.C_intervened.shape == result.C_pred.shape
    assert result.mask.shape == result.C_pred.shape
    assert result.y_pred_after.shape == (ds.test.n,)


def test_robot_alignment_end_to_end():
    """Train CBM + alignment → result dict has expected keys."""
    from concept_benchmark.alignment import retrain_aligned

    ds = _tabular_dataset(n=50, k=4)
    cbm = _train_cbm(ds)

    h_train = ds.training.C.astype(np.float32)
    h_test = (cbm.concept_detector.predict(ds.test) > 0.5).astype(np.float32)

    result = retrain_aligned(
        h_train=h_train,
        y_train=ds.training.y.astype(int),
        h_test=h_test,
        y_test=ds.test.y.astype(int),
        concept_names=list(ds.test.concepts),
        original_frontend=cbm.front_end_model,
        monotonicity_constraints={ds.test.concepts[0]: 1},
    )

    assert "original_accuracy" in result
    assert "aligned_accuracy" in result
    assert 0 <= result["original_accuracy"] <= 1
    assert 0 <= result["aligned_accuracy"] <= 1


def test_sudoku_tabular_end_to_end():
    """Generate tabular sudoku-like data, train CS 1 epoch, predict."""
    ds = _tabular_dataset(
        n=40,
        k=9,
        seed=171,
        concept_names=[f"rule_{i}" for i in range(9)],
    )
    cbm = _train_cbm(ds)
    preds = cbm.predict(ds.test)
    assert len(preds) == ds.test.n
    assert set(preds).issubset({0, 1})


def test_sudoku_selective():
    """Train CBM + selective prediction (confidence-based abstention)."""
    ds = _tabular_dataset(
        n=50,
        k=9,
        seed=171,
        concept_names=[f"rule_{i}" for i in range(9)],
    )
    cbm = _train_cbm(ds)

    # Selective prediction: keep only confident predictions
    concept_preds = cbm.concept_detector.predict(ds.test)
    proba = cbm.front_end_model.predict_proba((concept_preds > 0.5).astype(float))
    confidence = np.max(proba, axis=1)
    threshold = 0.6
    kept = confidence >= threshold
    coverage = kept.mean()
    if kept.any():
        sel_acc = (proba.argmax(axis=1)[kept] == ds.test.y[kept]).mean()
        assert 0 <= sel_acc <= 1
    assert 0 <= coverage <= 1
