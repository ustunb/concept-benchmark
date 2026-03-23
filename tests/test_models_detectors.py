import copy
import numpy as np
import torch.nn as nn

from experiments.models import ConceptDetector
from tests.conftest import _any_state_diff


import pytest


def _proba_checks(arr: np.ndarray, n: int, k: int):
    assert arr.shape == (n, k)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0) and np.all(arr <= 1)


def test_detector_fit_predict_with_encoder_freeze_variants(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    enc = nn.Linear(d, 6)
    # freeze_backbone=True keeps encoder unchanged
    det_frozen = ConceptDetector(embedding_model=copy.deepcopy(enc))
    before = copy.deepcopy(det_frozen.embedding_model.state_dict())
    det_frozen.fit(
        train, valid, freeze_backbone=True, fit_params={"epochs": 1, "device": "cpu"}
    )
    after = det_frozen.embedding_model.state_dict()
    assert not _any_state_diff(before, after)
    pr = det_frozen.predict(valid, should_calibrate=False)
    _proba_checks(pr, len(valid), k)

    # freeze_backbone=False updates encoder
    det_ft = ConceptDetector(embedding_model=nn.Linear(d, 6))
    before = copy.deepcopy(det_ft.embedding_model.state_dict())
    det_ft.fit(
        train,
        valid,
        freeze_backbone=False,
        fit_params={
            "epochs": 2,
            "device": "cpu",
            "lr_encoder": 1e-2,
            "lr_heads": 1e-2,
            "batch_size": 8,
        },
    )
    after = det_ft.embedding_model.state_dict()
    assert _any_state_diff(before, after)
    pr = det_ft.predict(valid, should_calibrate=False)
    _proba_checks(pr, len(valid), k)


def test_detector_predict_raises_when_calibrate_requested_but_not_fitted(
    tabular_train_valid,
):
    train, valid, d, k = tabular_train_valid
    det = ConceptDetector(embedding_model=None)
    det.fit(
        train, valid, freeze_backbone=True, fit_params={"epochs": 1, "device": "cpu"}
    )
    # Calibration requested but not fitted should raise
    with pytest.raises(RuntimeError):
        det.predict(valid, should_calibrate=True)


def test_detector_calibrate_after_fit_changes_predictions(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    det = ConceptDetector(embedding_model=nn.Linear(d, 6))
    # Fit without calibration
    det.fit(
        train,
        valid,
        freeze_backbone=False,
        should_calibrate=False,
        fit_params={
            "epochs": 2,
            "device": "cpu",
            "lr_encoder": 1e-2,
            "lr_heads": 1e-2,
            "batch_size": 8,
        },
    )
    pr_uncal = det.predict_proba(valid, should_calibrate=False)
    # Calibrate afterwards
    det.calibrate(valid)
    pr_cal = det.predict_proba(valid, should_calibrate=True)
    _proba_checks(pr_uncal, len(valid), k)
    _proba_checks(pr_cal, len(valid), k)
    assert not np.allclose(pr_uncal, pr_cal)
