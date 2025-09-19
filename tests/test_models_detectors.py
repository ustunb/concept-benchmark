import copy
import numpy as np
import torch
import torch.nn as nn

from concept_benchmark.models import ConceptBasedModel, ConceptDetector, JointConceptModel
from concept_benchmark.train import TrainerResult


import pytest


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


def test_detector_fit_predict_no_encoder(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    det = ConceptDetector(embedding_model=None)
    det.fit(train, valid, freeze=False, fit_params={"epochs": 1, "device": "cpu", "batch_size": 16})
    assert isinstance(det.model, nn.Module)
    assert det.n_concepts == k
    assert isinstance(det.training_result, TrainerResult)
    pr = det.predict(valid, calibrate=False)
    _proba_checks(pr, len(valid), k)


def test_detector_fit_predict_with_encoder_freeze_variants(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    enc = nn.Linear(d, 6)
    # freeze=True keeps encoder unchanged
    det_frozen = ConceptDetector(embedding_model=copy.deepcopy(enc))
    before = copy.deepcopy(det_frozen.embedding_model.state_dict())
    det_frozen.fit(train, valid, freeze=True, fit_params={"epochs": 1, "device": "cpu"})
    assert isinstance(det_frozen.model, JointConceptModel)
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


def test_detector_calibration_with_encoder_finetune(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    det = ConceptDetector(embedding_model=nn.Linear(d, 6))
    det.fit(
        train,
        valid,
        freeze=False,
        calibrate=True,
        fit_params={"epochs": 2, "device": "cpu", "lr_encoder": 1e-2, "lr_heads": 1e-2, "batch_size": 8},
    )
    assert det.n_concepts == k
    # Calibrated by default when calibration params exist
    pr = det.predict(valid)
    _proba_checks(pr, len(valid), k)
    # Uncalibrated override
    pr_uncal = det.predict(valid, calibrate=False)
    _proba_checks(pr_uncal, len(valid), k)
    # In general, calibration should change probabilities for at least some items
    assert not np.allclose(pr, pr_uncal)


def test_detector_predict_raises_when_calibrate_requested_but_not_fitted(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    det = ConceptDetector(embedding_model=None)
    det.fit(train, valid, freeze=False, fit_params={"epochs": 1, "device": "cpu"})
    # Calibration requested but not fitted should raise
    with pytest.raises(RuntimeError):
        det.predict(valid, calibrate=True)


def test_detector_calibrate_after_fit_changes_predictions(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
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


def test_detector_save_load_round_trip(tabular_train_valid, tmp_path):
    train, valid, d, k = tabular_train_valid
    det = ConceptDetector(embedding_model=nn.Linear(d, 6))
    det.fit(
        train,
        valid,
        freeze=False,
        fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
    )
    path = tmp_path / "detector.pkl"
    det.save(path, overwrite=True, msg=False)
    det_loaded = ConceptDetector.load(path, map_location="cpu")
    pr_orig = det.predict(valid, calibrate=False)
    pr_loaded = det_loaded.predict(valid, calibrate=False)
    _proba_checks(pr_loaded, len(valid), k)
    assert np.allclose(pr_orig, pr_loaded)
    param = next(det_loaded.model.parameters())
    assert param.device.type == "cpu"


def test_concept_based_model_save_load(tabular_train_valid, tmp_path):
    train, valid, d, k = tabular_train_valid
    model = ConceptBasedModel()
    model.fit(
        train,
        valid,
        freeze=False,
        concept_fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
        calibrate=False,
    )
    path = tmp_path / "cbm.pkl"
    model.save(path, overwrite=True, msg=False)
    loaded = ConceptBasedModel.load(path, map_location="cpu")
    preds_original = model.predict(valid)
    preds_loaded = loaded.predict(valid)
    assert np.array_equal(preds_original, preds_loaded)
    assert type(loaded.front_end_model.model) is type(model.front_end_model.model)
