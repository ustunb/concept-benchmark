"""Tests for concept_benchmark.utils module."""

from __future__ import annotations

import numpy as np
import torch

from concept_benchmark.utils import (
    compute_accuracy,
    determine_device,
    get_loader_config,
    set_deterministic_seed,
)


# ── set_deterministic_seed ───────────────────────────────────────────


class TestSeed:
    def test_numpy_deterministic(self):
        set_deterministic_seed(123)
        a = np.random.rand(5)
        set_deterministic_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self):
        set_deterministic_seed(456)
        a = torch.rand(5)
        set_deterministic_seed(456)
        b = torch.rand(5)
        torch.testing.assert_close(a, b)


# ── determine_device ─────────────────────────────────────────────────


class TestDetermineDevice:
    def test_returns_device(self):
        dev = determine_device()
        assert isinstance(dev, torch.device)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_DEVICE", "cpu")
        dev = determine_device()
        assert dev == torch.device("cpu")


# ── compute_accuracy ─────────────────────────────────────────────────


class TestComputeAccuracy:
    def test_perfect(self):
        """All-correct predictions → 1.0."""

        class _Model(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 1)

        model = _Model()
        data = [
            (torch.rand(4, 2), torch.zeros(4), torch.ones(4, dtype=torch.long)),
        ]
        acc = compute_accuracy(model, data, torch.device("cpu"))
        assert acc == 1.0

    def test_empty(self):
        class _Model(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(0, 1)

        model = _Model()
        acc = compute_accuracy(model, [], torch.device("cpu"))
        assert acc == 0


# ── get_loader_config ────────────────────────────────────────────────


class TestGetLoaderConfig:
    def test_keys(self):
        cfg = get_loader_config()
        assert "batch_size" in cfg
        assert "num_workers" in cfg
        assert "pin_memory" in cfg

    def test_macos(self, monkeypatch):
        monkeypatch.setattr("concept_benchmark.utils.platform.system", lambda: "Darwin")
        cfg = get_loader_config()
        assert cfg["num_workers"] == 0
        assert cfg["pin_memory"] is False

    def test_linux(self, monkeypatch):
        monkeypatch.setattr("concept_benchmark.utils.platform.system", lambda: "Linux")
        cfg = get_loader_config()
        assert cfg["num_workers"] > 0
        assert cfg["pin_memory"] is torch.cuda.is_available()
