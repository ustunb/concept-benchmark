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
