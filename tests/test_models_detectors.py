import copy
import numpy as np
import torch
import torch.nn as nn

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.models import ConceptDetector


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


def test_detector_fit_predict_no_encoder():
    train, valid, d, k = _make_tabular_samples()
    det = ConceptDetector(embedding_model=None)
    det.fit(train, valid, freeze=True, fit_params={"epochs": 1, "device": "cpu", "batch_size": 16})
    assert isinstance(det.concept_layers, nn.ModuleList)
    assert len(det.concept_layers) == k
    pr = det.predict(valid, calibrate=False)
    _proba_checks(pr, len(valid), k)


def test_detector_fit_predict_with_encoder_freeze_variants():
    train, valid, d, k = _make_tabular_samples()
    enc = nn.Linear(d, 6)
    # freeze=True keeps encoder unchanged
    det_frozen = ConceptDetector(embedding_model=copy.deepcopy(enc))
    before = copy.deepcopy(det_frozen.embedding_model.state_dict())
    det_frozen.fit(train, valid, freeze=True, fit_params={"epochs": 1, "device": "cpu"})
    after = det_frozen.embedding_model.state_dict()
    assert not _any_state_diff(before, after)
    pr = det_frozen.predict(valid, calibrate=False)
    _proba_checks(pr, len(valid), k)

    # freeze=False updates encoder
    det_ft = ConceptDetector(embedding_model=nn.Linear(d, 6))
    before = copy.deepcopy(det_ft.embedding_model.state_dict())
    det_ft.fit(
        train,
        valid,
        freeze=False,
        fit_params={"epochs": 2, "device": "cpu", "lr_encoder": 1e-2, "lr_heads": 1e-2, "batch_size": 8},
    )
    after = det_ft.embedding_model.state_dict()
    assert _any_state_diff(before, after)
    pr = det_ft.predict(valid, calibrate=False)
    _proba_checks(pr, len(valid), k)


def test_detector_calibration_with_encoder_finetune():
    train, valid, d, k = _make_tabular_samples()
    det = ConceptDetector(embedding_model=nn.Linear(d, 6))
    det.fit(
        train,
        valid,
        freeze=False,
        calibrate=True,
        fit_params={"epochs": 2, "device": "cpu", "lr_encoder": 1e-2, "lr_heads": 1e-2, "batch_size": 8},
    )
    assert len(det.concept_layers) == k
    # Calibrated by default when calibration params exist
    pr = det.predict(valid)
    _proba_checks(pr, len(valid), k)
    # Uncalibrated override
    pr_uncal = det.predict(valid, calibrate=False)
    _proba_checks(pr_uncal, len(valid), k)
    # In general, calibration should change probabilities for at least some items
    assert not np.allclose(pr, pr_uncal)


def test_detector_predict_raises_when_calibrate_requested_but_not_fitted():
    train, valid, d, k = _make_tabular_samples()
    det = ConceptDetector(embedding_model=None)
    det.fit(train, valid, freeze=True, fit_params={"epochs": 1, "device": "cpu"})
    # Calibration requested but not fitted should raise
    import pytest
    with pytest.raises(RuntimeError):
        det.predict(valid, calibrate=True)


def test_detector_calibrate_after_fit_changes_predictions():
    train, valid, d, k = _make_tabular_samples()
    det = ConceptDetector(embedding_model=nn.Linear(d, 6))
    # Fit without calibration
    det.fit(
        train,
        valid,
        freeze=False,
        calibrate=False,
        fit_params={"epochs": 2, "device": "cpu", "lr_encoder": 1e-2, "lr_heads": 1e-2, "batch_size": 8},
    )
    pr_uncal = det.predict(valid, calibrate=False)
    # Calibrate afterwards
    det.calibrate(valid)
    pr_cal = det.predict(valid, calibrate=True)
    _proba_checks(pr_uncal, len(valid), k)
    _proba_checks(pr_cal, len(valid), k)
    assert not np.allclose(pr_uncal, pr_cal)
