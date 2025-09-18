import copy

import numpy as np
import torch
import torch.nn as nn

from concept_benchmark.train import DefaultConceptTrainer, TrainerResult


def test_default_trainer_returns_trainer_result(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    model = nn.Sequential(nn.Linear(d, k))
    trainer = DefaultConceptTrainer()
    result = trainer(
        model,
        train,
        valid,
        num_concepts=k,
        params={"epochs": 1, "device": "cpu", "batch_size": 16},
    )
    assert isinstance(result, TrainerResult)
    assert isinstance(result.model, nn.Module)
    assert not result.model.training
    assert "train_loss" in result.history


def test_default_trainer_accepts_optimizer_factory(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    model = nn.Sequential(nn.Linear(d, k))

    called = {"flag": False}

    def _opt_factory(params):
        called["flag"] = True
        return torch.optim.Adam(params, lr=1e-2)

    trainer = DefaultConceptTrainer()
    _ = trainer(
        model,
        train,
        valid,
        num_concepts=k,
        params={"epochs": 1, "device": "cpu", "batch_size": 16, "optimizer_factory": _opt_factory},
    )
    assert called["flag"] is True


def test_default_trainer_masks_missing_concepts(tabular_train_valid):
    train, valid, d, k = tabular_train_valid

    train_missing = copy.deepcopy(train)
    valid_missing = copy.deepcopy(valid)

    train_missing.concept_missing = True
    valid_missing.concept_missing = True

    train_mask = np.zeros_like(train_missing.base_concepts, dtype=bool)
    train_mask[::2, 0] = True
    valid_mask = np.zeros_like(valid_missing.base_concepts, dtype=bool)
    valid_mask[::3, 0] = True

    train_missing.set_concept_missing_mask(train_mask, fill_value=np.nan)
    valid_missing.set_concept_missing_mask(valid_mask, fill_value=np.nan)

    model = nn.Sequential(nn.Linear(d, k))
    trainer = DefaultConceptTrainer()
    result = trainer(
        model,
        train_missing,
        valid_missing,
        num_concepts=k,
        params={"epochs": 1, "device": "cpu", "batch_size": 16},
    )

    assert isinstance(result, TrainerResult)
    assert np.isfinite(np.asarray(result.history["train_loss"]))
