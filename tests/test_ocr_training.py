"""The digit recognizer stops training when its validation accuracy stops improving and keeps the best epoch."""

import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from concept_benchmark.synthetic.sudoku.ocr.train_ocr_fast import train_resnet_tiny


def _loaders(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(64, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (64,), generator=g)
    train = DataLoader(TensorDataset(x, y), batch_size=32)
    val = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=32)
    return train, val


def test_training_stops_after_patience_epochs_without_improvement(monkeypatch):
    accuracies = iter([0.5, 0.6, 0.6, 0.6, 0.6, 0.9, 0.9])
    monkeypatch.setattr(
        "concept_benchmark.synthetic.sudoku.ocr.train_ocr_fast.eval_model",
        lambda model, loader, device: (0.0, next(accuracies), None),
    )
    train, val = _loaders()
    result = train_resnet_tiny(
        train,
        val,
        "cpu",
        epochs=7,
        class_weights=None,
        logger=logging.getLogger("t"),
        patience=3,
    )
    assert result["best_epoch"] == 2
    assert result["best_val_acc"] == 0.6


def test_without_patience_training_runs_every_epoch_and_keeps_the_best(monkeypatch):
    accuracies = iter([0.5, 0.6, 0.4, 0.7])
    monkeypatch.setattr(
        "concept_benchmark.synthetic.sudoku.ocr.train_ocr_fast.eval_model",
        lambda model, loader, device: (0.0, next(accuracies), None),
    )
    train, val = _loaders()
    result = train_resnet_tiny(
        train, val, "cpu", epochs=4, class_weights=None, logger=logging.getLogger("t")
    )
    assert result["best_epoch"] == 4
