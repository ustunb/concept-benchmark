from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np
import pytest
import torch

from concept_benchmark.data import ConceptDataset, ConceptDatasetSample
from concept_benchmark.ext.fileutils import load as load_object, save as save_object
from experiments.cem_integration import (
    CEMDependencyError,
    CEMSampleAdapterDataset,
    ECBMBenchmarkModel,
    _OfficialBenchmarkModelBase,
    _PredictionCache,
    _ensure_local_cem_checkout_on_path,
    compute_ecbm_interpretation_summary,
    require_cem_dependencies,
    train_cem_model,
    train_ecbm_model,
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
    ecbm_emb_size: int = 4
    ecbm_hid_size: int = 16
    ecbm_lambda_xy: float = 1.0
    ecbm_lambda_xc: float = 1.0
    ecbm_lambda_cy: float = 1.0
    ecbm_weight_decay: float = 1e-4
    ecbm_inference_steps: int = 5
    ecbm_inference_lr: float = 0.1
    ecbm_max_epochs: int | None = 2


def _toy_dataset_sample(n: int, k: int) -> ConceptDatasetSample:
    return ConceptDatasetSample(
        X=np.zeros((n, 2), dtype=np.float32),
        C=np.zeros((n, k), dtype=np.int8),
        y=np.zeros(n, dtype=np.int32),
        meta={
            "classes": ["drent", "glorp"],
            "concepts": [f"concept_{i}" for i in range(k)],
            "data_type": "tabular",
        },
    )

class _ToyOfficialModule(torch.nn.Module):
    def __init__(self, n_concepts: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n_concepts, dtype=torch.float32))


class _ToyOfficialBenchmarkModel(_OfficialBenchmarkModelBase):
    family = "toyofficial"

    def __init__(self, cached_concepts: np.ndarray) -> None:
        cached_concepts = np.asarray(cached_concepts, dtype=np.float32)
        self.fixed_concepts = cached_concepts
        self.last_replay: dict[str, np.ndarray] | None = None
        super().__init__(
            official_model=_ToyOfficialModule(cached_concepts.shape[1]),
            benchmark="robot",
            concept_names=[f"concept_{i}" for i in range(cached_concepts.shape[1])],
            class_names=["drent", "glorp"],
            backbone_spec={"kind": "tabular", "input_dim": 2},
            model_init_kwargs={"n_concepts": cached_concepts.shape[1]},
            eval_config={"batch_size": len(cached_concepts), "device": "cpu"},
            training_summary={},
        )

    @staticmethod
    def _label_probs(concepts: np.ndarray) -> np.ndarray:
        score = 0.7 * concepts[:, 0] - 0.4 * concepts[:, 1]
        if concepts.shape[1] > 2:
            score = score + 0.2 * concepts[:, 2]
        prob1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - prob1, prob1]).astype(np.float32)

    def _run_official_model(self, dataset):
        if dataset.n != self.fixed_concepts.shape[0]:
            raise AssertionError("Dataset length must match cached concepts in test.")
        concept_probs = self.fixed_concepts.copy()
        label_probs = self._label_probs(concept_probs)
        cache = _PredictionCache(
            dataset_id=id(dataset),
            concept_probs=concept_probs,
            label_probs=label_probs,
        )
        return label_probs, concept_probs, cache

    def _predict_from_cached_concepts(
        self,
        concepts: np.ndarray,
        cache: _PredictionCache,
        *,
        baseline_concepts: np.ndarray,
        intervention_mask: np.ndarray | None,
    ) -> np.ndarray:
        effective = (
            np.where(intervention_mask, concepts, baseline_concepts)
            if intervention_mask is not None
            else concepts
        )
        self.last_replay = {
            "concepts": np.asarray(concepts, dtype=np.float32),
            "baseline_concepts": np.asarray(baseline_concepts, dtype=np.float32),
            "intervention_mask": None
            if intervention_mask is None
            else np.asarray(intervention_mask, dtype=bool),
            "effective": np.asarray(effective, dtype=np.float32),
            "cached_concepts": np.asarray(cache.concept_probs, dtype=np.float32),
        }
        return self._label_probs(effective)

    def _rebuild_model(self, *, model_init_kwargs, backbone_spec):
        return _ToyOfficialModule(len(self.concept_names))


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


def test_official_replay_supports_repeated_rows_with_alignment_metadata():
    cached = np.array(
        [
            [0.10, 0.70, 0.40],
            [0.80, 0.20, 0.60],
            [0.35, 0.55, 0.25],
        ],
        dtype=np.float32,
    )
    sample = _toy_dataset_sample(n=3, k=3)

    model = _ToyOfficialBenchmarkModel(cached)
    model.predict_proba(sample)

    row_indices = np.array([2, 0, 2], dtype=int)
    baseline = cached[row_indices]
    intervened = baseline.copy()
    intervened[0, 1] = 0.95
    intervened[2, 0] = 0.05
    intervention_mask = np.zeros_like(intervened, dtype=bool)
    intervention_mask[0, 1] = True
    intervention_mask[2, 0] = True

    y_prob = model.predict_proba_from_concepts(
        intervened,
        row_indices=row_indices,
        baseline_concepts=baseline,
        intervention_mask=intervention_mask,
    )

    assert y_prob.shape == (3, 2)
    assert model.last_replay is not None
    np.testing.assert_allclose(model.last_replay["cached_concepts"], cached[row_indices])
    np.testing.assert_allclose(model.last_replay["baseline_concepts"], baseline)
    np.testing.assert_allclose(
        model.last_replay["effective"],
        np.where(intervention_mask, intervened, baseline),
    )


def test_official_replay_requires_safe_full_dataset_alignment():
    cached = np.array(
        [
            [0.15, 0.65, 0.30],
            [0.75, 0.25, 0.55],
            [0.40, 0.60, 0.20],
        ],
        dtype=np.float32,
    )
    sample = _toy_dataset_sample(n=3, k=3)

    model = _ToyOfficialBenchmarkModel(cached)
    model.predict_proba(sample)

    subset = np.repeat(cached[:1], 2, axis=0)
    with pytest.raises(ValueError, match="row_indices/source_indices"):
        model.predict_proba_from_concepts(subset)

    modified_full = cached.copy()
    modified_full[0, 0] = 0.95
    with pytest.raises(ValueError, match="only safe when replaying the cached concept predictions exactly"):
        model.predict_proba_from_concepts(modified_full)

    with pytest.raises(ValueError, match="baseline_concepts must align with the cached dataset rows"):
        model.predict_proba_from_concepts(
            modified_full,
            baseline_concepts=cached[::-1],
        )

    y_prob = model.predict_proba_from_concepts(
        modified_full,
        baseline_concepts=cached,
        intervention_mask=np.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        ),
    )
    assert y_prob.shape == (3, 2)


def test_official_restore_raises_on_state_mismatch():
    cached = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
        ],
        dtype=np.float32,
    )
    model = _ToyOfficialBenchmarkModel(cached)
    state = model.__getstate__()
    state["model_state_dict"] = {"wrong_key": torch.ones(1)}

    restored = _ToyOfficialBenchmarkModel.__new__(_ToyOfficialBenchmarkModel)
    with pytest.raises(RuntimeError):
        restored.__setstate__(state)


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


def test_probcbm_wrapper_smoke_train_and_predict(tmp_path):
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
    binary_c = (c_prob > 0.5).astype(float)
    changed_mask = ~np.isclose(binary_c, c_prob, atol=1e-6, rtol=1e-6)
    assert model.predict_proba_from_concepts(
        binary_c,
        baseline_concepts=c_prob,
        intervention_mask=changed_mask,
    ).shape == (
        ds.test.n,
        ds.test.n_classes,
    )

    path = tmp_path / "probcbm.model"
    save_object(model, path, overwrite=True)
    loaded = load_object(path)
    loaded_y_prob, loaded_c_prob = loaded.predict_proba(ds.test, return_concepts=True)
    assert loaded_y_prob.shape == y_prob.shape
    assert loaded_c_prob.shape == c_prob.shape


def test_ecbm_wrapper_smoke_train_predict_replay_and_interpret(tmp_path):
    ds = _tiny_tabular_dataset(seed=23)
    model = train_ecbm_model(
        train_dataset=ds.train,
        valid_dataset=ds.validation,
        benchmark="robot",
        config=_TinyCEMConfig(),
        device="cpu",
        num_workers=0,
        pin_memory=False,
    )

    assert isinstance(model, ECBMBenchmarkModel)

    preds = model.predict(ds.test)
    y_prob, c_prob = model.predict_proba(ds.test, return_concepts=True)

    assert preds.shape == (ds.test.n,)
    assert y_prob.shape == (ds.test.n, ds.test.n_classes)
    assert c_prob.shape == (ds.test.n, ds.test.n_concepts)

    replayed = model.predict_proba_from_concepts(c_prob)
    np.testing.assert_allclose(replayed, y_prob, atol=1e-6, rtol=1e-6)

    binary_c = (c_prob > 0.5).astype(np.float32)
    changed_mask = ~np.isclose(binary_c, c_prob, atol=1e-6, rtol=1e-6)
    intervened = model.predict_proba_from_concepts(
        binary_c,
        baseline_concepts=c_prob,
        intervention_mask=changed_mask,
    )
    assert intervened.shape == (ds.test.n, ds.test.n_classes)

    summary = compute_ecbm_interpretation_summary(model, ds.test, top_k=3)
    assert summary["family"] == "ecbm"
    assert len(summary["rows"]) == ds.test.n_concepts * ds.test.n_classes
    assert set(summary["top_concepts_by_class"]) == set(ds.test.classes)

    path = tmp_path / "ecbm.model"
    save_object(model, path, overwrite=True)
    loaded = load_object(path)
    loaded_y_prob, loaded_c_prob = loaded.predict_proba(ds.test, return_concepts=True)
    assert loaded_y_prob.shape == y_prob.shape
    assert loaded_c_prob.shape == c_prob.shape
