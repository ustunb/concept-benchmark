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
        assert cfg["pin_memory"] is True


# ── create_skewed_splits_full ────────────────────────────────────────

class TestSkewedSplits:
    def _make_dataset(self, n=200, k=3):
        from tests.conftest import make_tabular_dataset

        ds, _ = make_tabular_dataset(n=n, d=5, k=k, n_classes=2)
        return ds

    def test_produces_splits(self):
        from concept_benchmark.utils import create_skewed_splits_full

        ds = self._make_dataset(n=200, k=3)
        ds.generate_cvindices(seed=42)
        rng = np.random.default_rng(42)
        result = create_skewed_splits_full(
            dataset=ds,
            skew_specs=[],
            test_size=50,
            train_skew_size=100,
            val_fraction=0.5,
            rng=rng,
            drop_concepts=[],
        )
        assert result.training is not None
        assert result.validation is not None
        assert result.test is not None
        assert result.training.n > 0
        assert result.validation.n > 0
        assert result.test.n > 0

    def test_test_size(self):
        from concept_benchmark.utils import create_skewed_splits_full

        ds = self._make_dataset(n=200, k=3)
        ds.generate_cvindices(seed=42)
        rng = np.random.default_rng(42)
        result = create_skewed_splits_full(
            dataset=ds,
            skew_specs=[],
            test_size=60,
            train_skew_size=80,
            val_fraction=0.5,
            rng=rng,
            drop_concepts=[],
        )
        assert result.test.n == 60

    def test_drops_concepts(self):
        from concept_benchmark.utils import create_skewed_splits_full

        ds = self._make_dataset(n=200, k=3)
        ds.generate_cvindices(seed=42)
        original_concepts = list(ds.concepts)
        rng = np.random.default_rng(42)
        result = create_skewed_splits_full(
            dataset=ds,
            skew_specs=[],
            test_size=50,
            train_skew_size=100,
            val_fraction=0.5,
            rng=rng,
            drop_concepts=[original_concepts[0]],
        )
        assert original_concepts[0] not in result.training.concepts
