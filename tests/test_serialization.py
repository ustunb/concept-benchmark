"""Tests for model serialization round-trip."""
from __future__ import annotations

import numpy as np

from concept_benchmark.data import ConceptDataset
from concept_benchmark.ext.fileutils import load, save
from experiments.models import ConceptBasedModel, ConceptDetector


def _make_dataset(n=40, k=4, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 8)).astype(np.float32)
    C = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y = np.tile([0, 1], n // 2 + 1)[:n].astype(np.int32)
    rng.shuffle(y)
    meta = {
        "classes": ["c0", "c1"],
        "concepts": [f"z{i}" for i in range(k)],
        "data_type": "tabular",
    }
    ds = ConceptDataset(X=X, C=C, y=y, meta=meta)
    ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=seed)
    return ds


def test_cbm_save_load_roundtrip(tmp_path):
    """Train CBM, save, load, verify predictions match."""
    ds = _make_dataset()
    cd = ConceptDetector()
    cbm = ConceptBasedModel(concept_detector=cd)
    cbm.fit(
        train_dataset=ds.training,
        valid_dataset=ds.validation,
        freeze_backbone=False,
        concept_embed_params={"device": "cpu", "batch_size": 8, "num_workers": 0},
        concept_fit_params={
            "epochs": 2,
            "lr": 1e-3,
            "patience": 1,
            "device": "cpu",
            "batch_size": 8,
            "num_workers": 0,
        },
    )

    preds_before = cbm.predict(ds.test)
    concept_preds_before = cbm.concept_detector.predict(ds.test)

    model_path = tmp_path / "test_cbm.model"
    save(cbm, model_path, overwrite=True)
    cbm_loaded = load(model_path)

    preds_after = cbm_loaded.predict(ds.test)
    concept_preds_after = cbm_loaded.concept_detector.predict(ds.test)

    np.testing.assert_array_equal(preds_before, preds_after)
    np.testing.assert_allclose(concept_preds_before, concept_preds_after, atol=1e-6)
