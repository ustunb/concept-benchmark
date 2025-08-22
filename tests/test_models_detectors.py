import copy
import numpy as np
import torch
import torch.nn as nn

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.models import ClassicalConceptDetector, CalibratedConceptDetector


def _make_tabular_samples(n=32, d=8, k=2):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, k)).astype(np.float32)
    logits = X @ W
    C = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.int8)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    meta = {"classes": ["a", "b"], "concepts": [f"c{i}" for i in range(k)], "data_type": "tabular"}
    ds = ConceptDatasetSample(X=X, C=C, y=y, meta=meta)
    # split
    train = ConceptDatasetSample(X=X[:24], C=C[:24], y=y[:24], meta=meta)
    valid = ConceptDatasetSample(X=X[24:], C=C[24:], y=y[24:], meta=meta)
    return train, valid, d, k


def _proba_checks(arr: np.ndarray, n: int, k: int):
    assert arr.shape == (n, k)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0) and np.all(arr <= 1)


def _any_state_diff(state_a, state_b):
    for k in state_a:
        ta, tb = state_a[k], state_b[k]
        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            return True
        if not torch.allclose(ta, tb):
            return True
    return False


def test_classical_fit_predict_no_encoder():
    train, valid, d, k = _make_tabular_samples()
    det = ClassicalConceptDetector(embedding_model=None)
    det.fit(train, valid, freeze=True, fit_params={"epochs": 1, "device": "cpu", "batch_size": 16})
    assert isinstance(det.concept_layers, nn.ModuleList)
    assert len(det.concept_layers) == k
    pr = det.predict(valid)
    _proba_checks(pr, len(valid), k)


def test_classical_fit_predict_with_encoder_freeze_variants():
    train, valid, d, k = _make_tabular_samples()
    enc = nn.Linear(d, 6)
    # freeze=True keeps encoder unchanged
    det_frozen = ClassicalConceptDetector(embedding_model=copy.deepcopy(enc))
    before = copy.deepcopy(det_frozen.embedding_model.state_dict())
    det_frozen.fit(train, valid, freeze=True, fit_params={"epochs": 1, "device": "cpu"})
    after = det_frozen.embedding_model.state_dict()
    assert not _any_state_diff(before, after)
    pr = det_frozen.predict(valid)
    _proba_checks(pr, len(valid), k)

    # freeze=False updates encoder
    det_ft = ClassicalConceptDetector(embedding_model=nn.Linear(d, 6))
    before = copy.deepcopy(det_ft.embedding_model.state_dict())
    det_ft.fit(
        train,
        valid,
        freeze=False,
        fit_params={"epochs": 2, "device": "cpu", "lr_encoder": 1e-2, "lr_heads": 1e-2, "batch_size": 8},
    )
    after = det_ft.embedding_model.state_dict()
    assert _any_state_diff(before, after)
    pr = det_ft.predict(valid)
    _proba_checks(pr, len(valid), k)


def test_calibrated_fit_predict_with_encoder_finetune():
    train, valid, d, k = _make_tabular_samples()
    det = CalibratedConceptDetector(embedding_model=nn.Linear(d, 6))
    det.fit(
        train,
        valid,
        freeze=False,
        fit_params={"epochs": 2, "device": "cpu", "lr_encoder": 1e-2, "lr_heads": 1e-2, "batch_size": 8},
    )
    from sklearn.calibration import CalibratedClassifierCV

    assert len(det.concept_layers) == k
    assert all(isinstance(m, CalibratedClassifierCV) for m in det.concept_layers)
    pr = det.predict(valid)
    _proba_checks(pr, len(valid), k)


def test_calibrated_fit_predict_no_encoder():
    train, valid, d, k = _make_tabular_samples()
    det = CalibratedConceptDetector(embedding_model=None)
    det.fit(train, valid, freeze=True, fit_params={"epochs": 1, "device": "cpu"})
    from sklearn.calibration import CalibratedClassifierCV

    assert len(det.concept_layers) == k
    assert all(isinstance(m, CalibratedClassifierCV) for m in det.concept_layers)
    pr = det.predict(valid)
    _proba_checks(pr, len(valid), k)

