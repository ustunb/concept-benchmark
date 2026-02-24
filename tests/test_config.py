"""Tests for config validation."""
from __future__ import annotations

import pytest
from concept_benchmark.config import (
    RobotBenchmarkConfig,
    SudokuBenchmarkConfig,
    RobotTextBenchmarkConfig,
)


class TestRobotConfigValidation:
    def test_default_config_is_valid(self):
        cfg = RobotBenchmarkConfig()
        assert cfg.seed == 1014

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            RobotBenchmarkConfig(seed=-1)

    def test_rejects_negative_budget(self):
        with pytest.raises(ValueError, match="intervention_budgets must be non-negative"):
            RobotBenchmarkConfig(intervention_budgets=[-1, 3])

    def test_rejects_unknown_regime(self):
        with pytest.raises(ValueError, match="unknown intervention regimes"):
            RobotBenchmarkConfig(intervention_regimes=["baseline", "bogus"])

    def test_rejects_bad_strategy(self):
        with pytest.raises(ValueError, match="intervention_strategy must be one of"):
            RobotBenchmarkConfig(intervention_strategy="random")

    def test_accepts_all_valid_regimes(self):
        cfg = RobotBenchmarkConfig(
            intervention_regimes=["baseline", "expert", "subjective", "machine", "llm", "clip"]
        )
        assert len(cfg.intervention_regimes) == 6

    def test_exact_k_strategy(self):
        cfg = RobotBenchmarkConfig(intervention_strategy="exact_k")
        assert cfg.intervention_strategy == "exact_k"


class TestSudokuConfigValidation:
    def test_default_config_is_valid(self):
        cfg = SudokuBenchmarkConfig()
        assert cfg.seed == 171

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            SudokuBenchmarkConfig(seed=-1)

    def test_rejects_zero_samples(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            SudokuBenchmarkConfig(n_samples=0)


class TestRobotTextConfigValidation:
    def test_default_config_is_valid(self):
        cfg = RobotTextBenchmarkConfig()
        assert cfg.seed == 1337

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            RobotTextBenchmarkConfig(seed=-1)

    def test_rejects_unknown_regime(self):
        with pytest.raises(ValueError, match="unknown intervention regimes"):
            RobotTextBenchmarkConfig(intervention_regimes=["llm"])

    def test_rejects_bad_strategy(self):
        with pytest.raises(ValueError, match="intervention_strategy must be one of"):
            RobotTextBenchmarkConfig(intervention_strategy="random")
