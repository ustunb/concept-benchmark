import copy
import numpy as np
import torch
import torch.nn as nn

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.train import (
    train_concept_heads,
    calibrate_trained_heads,
    TorchSKLearnWrapper,
)


def _make_tabular_samples(n=32, d=8, k=2):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, k)).astype(np.float32)
    logits = X @ W
    C = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.int8)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    meta = {"classes": ["a", "b"], "concepts": [f"c{i}" for i in range(k)], "data_type": "tabular"}
    ds = ConceptDatasetSample(X=X, C=C, y=y, meta=meta)
    # simple split
    train = ConceptDatasetSample(X=X[:24], C=C[:24], y=y[:24], meta=meta)
    valid = ConceptDatasetSample(X=X[24:], C=C[24:], y=y[24:], meta=meta)
    return train, valid, d, k


def _any_state_diff(state_a, state_b):
    for k in state_a:
        ta, tb = state_a[k], state_b[k]
        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            return True
        if not torch.allclose(ta, tb):
            return True
    return False


def test_train_heads_no_encoder_builds_heads_architecture():
    train, valid, d, k = _make_tabular_samples()
    heads = train_concept_heads(
        train_dataset=train,
        valid_dataset=valid,
        embedding_model=None,
        input_dim=None,
        l1_size=16,
        freeze=True,
        fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
    )
    assert isinstance(heads, nn.ModuleList)
    assert len(heads) == k
    first = heads[0]
    assert isinstance(first[0], nn.Linear)
    assert first[0].in_features == d
    assert isinstance(first[-1], nn.Linear)
    assert first[-1].out_features == 1


def test_train_heads_with_encoder_freeze_true_keeps_encoder_constant():
    train, valid, d, k = _make_tabular_samples()
    enc = nn.Linear(d, 6)
    before = copy.deepcopy(enc.state_dict())
    heads = train_concept_heads(
        train_dataset=train,
        valid_dataset=valid,
        embedding_model=enc,
        input_dim=None,
        l1_size=8,
        freeze=True,
        fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
    )
    after = enc.state_dict()
    assert not _any_state_diff(before, after), "Encoder should not change when frozen"
    assert heads[0][0].in_features == 6


def test_train_heads_with_encoder_finetunes_encoder_changes():
    train, valid, d, k = _make_tabular_samples()
    enc = nn.Linear(d, 6)
    before = copy.deepcopy(enc.state_dict())
    _ = train_concept_heads(
        train_dataset=train,
        valid_dataset=valid,
        embedding_model=enc,
        input_dim=None,
        l1_size=8,
        freeze=False,
        fit_params={"epochs": 2, "device": "cpu", "batch_size": 8, "lr_encoder": 1e-2, "lr_heads": 1e-2},
    )
    after = enc.state_dict()
    assert _any_state_diff(before, after), "Encoder should update when not frozen"


def test_calibrate_trained_heads_returns_calibrators():
    train, valid, d, k = _make_tabular_samples()
    heads = train_concept_heads(
        train_dataset=train,
        valid_dataset=valid,
        embedding_model=None,
        input_dim=None,
        l1_size=8,
        freeze=True,
        fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
    )
    calibs = calibrate_trained_heads(train, valid, heads)
    assert len(calibs) == k
    pr = calibs[0].predict_proba(valid.X)
    assert pr.shape == (len(valid), 2)
    assert np.all(pr >= 0) and np.all(pr <= 1)
    assert np.allclose(pr.sum(axis=1), 1, atol=1e-6)


def test_torch_sklearn_wrapper_predict_proba_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 4)).astype(np.float32)
    head = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    wrapper = TorchSKLearnWrapper(head)
    wrapper.fit(X, (rng.random(5) > 0.5).astype(int))
    pr = wrapper.predict_proba(X)
    assert pr.shape == (5, 2)
    assert np.allclose(pr.sum(axis=1), 1.0, atol=1e-6)

