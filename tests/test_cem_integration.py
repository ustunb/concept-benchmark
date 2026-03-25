from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np
import pytest
import torch

from concept_benchmark.data import ConceptDataset
from concept_benchmark.ext.fileutils import load as load_object, save as save_object
from experiments.cem_integration import (
    CEMDependencyError,
    CEMSampleAdapterDataset,
    _ensure_local_cem_checkout_on_path,
    require_cem_dependencies,
    train_cem_model,
    train_probcbm_model,
)


def _tiny_tabular_dataset(n=40, d=8, k=4, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    C = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y = rng.integers(0, 2, size=(n,)).astype(np.int64)
    meta = {
        "classes": ["drent", "glorp"],
        "concepts": [f"concept_{i}" for i in range(k)],
        "data_type": "tabular",
    }
    ds = ConceptDataset(X=X, C=C, y=y, meta=meta)
    ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=seed)
    return ds


@dataclass
class _TinyCEMConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    patience: int = 1
    epochs: int = 1
    cem_emb_size: int = 8
    cem_training_intervention_prob: float = 0.25
    cem_concept_loss_weight: float = 1.0
    cem_task_loss_weight: float = 1.0
    cem_max_epochs: int | None = 1
    probcbm_hidden_dim: int = 4
    probcbm_class_hidden_dim: int = 8
    probcbm_latent_dim: int = 4
    probcbm_n_samples_inference: int = 1
    probcbm_intervention_prob: float = 0.25
    probcbm_max_epochs: int | None = 1


def test_cem_sample_adapter_reorders_to_x_y_c():
    ds = _tiny_tabular_dataset(n=12, d=6, k=3)
    adapter = CEMSampleAdapterDataset(ds.train)

    x, y, c = adapter[0]

    assert np.allclose(np.asarray(x), ds.train.X[0])
    assert int(y.item()) == int(ds.train.y[0])
    assert c.dtype == torch.float32
    assert np.allclose(c.numpy(), ds.train.C[0])


def test_require_cem_dependencies_raises_clean_error_when_missing(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import(name, package=None):
        if name.startswith("cem"):
            raise ModuleNotFoundError("simulated missing cem")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(CEMDependencyError, match="Optional CEM/ProbCBM support"):
        require_cem_dependencies()


def test_cem_wrapper_smoke_train_predict_and_serialize(tmp_path):
    _ensure_local_cem_checkout_on_path()
    pytest.importorskip("cem")
    pytest.importorskip("pytorch_lightning")

    ds = _tiny_tabular_dataset()
    model = train_cem_model(
        train_dataset=ds.train,
        valid_dataset=ds.validation,
        benchmark="robot",
        config=_TinyCEMConfig(),
        device="cpu",
        num_workers=0,
        pin_memory=False,
    )

    preds = model.predict(ds.test)
    y_prob, c_prob = model.predict_proba(ds.test, return_concepts=True)

    assert preds.shape == (ds.test.n,)
    assert y_prob.shape == (ds.test.n, ds.test.n_classes)
    assert c_prob.shape == (ds.test.n, ds.test.n_concepts)
    assert model.label_predictor.predict_proba(c_prob).shape == (
        ds.test.n,
        ds.test.n_classes,
    )

    path = tmp_path / "cem.model"
    save_object(model, path, overwrite=True)
    loaded = load_object(path)
    loaded_y_prob, loaded_c_prob = loaded.predict_proba(ds.test, return_concepts=True)
    assert loaded_y_prob.shape == y_prob.shape
    assert loaded_c_prob.shape == c_prob.shape


def test_probcbm_wrapper_smoke_train_and_predict():
    _ensure_local_cem_checkout_on_path()
    pytest.importorskip("cem")
    pytest.importorskip("pytorch_lightning")

    ds = _tiny_tabular_dataset(seed=11)
    model = train_probcbm_model(
        train_dataset=ds.train,
        valid_dataset=ds.validation,
        benchmark="robot",
        config=_TinyCEMConfig(),
        device="cpu",
        num_workers=0,
        pin_memory=False,
    )

    preds = model.predict(ds.test)
    y_prob, c_prob = model.predict_proba(ds.test, return_concepts=True)

    assert preds.shape == (ds.test.n,)
    assert y_prob.shape == (ds.test.n, ds.test.n_classes)
    assert c_prob.shape == (ds.test.n, ds.test.n_concepts)
    assert model.label_predictor.predict_proba((c_prob > 0.5).astype(float)).shape == (
        ds.test.n,
        ds.test.n_classes,
    )
