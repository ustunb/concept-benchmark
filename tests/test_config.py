"""Tests for config validation."""
from __future__ import annotations

from dataclasses import fields

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
            RobotBenchmarkConfig(intervention_budgets=[-2, 3])

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

    def test_default_data_type_is_image(self):
        cfg = SudokuBenchmarkConfig()
        assert cfg.data_type == "image"

    def test_explicit_tabular_data_type(self):
        cfg = SudokuBenchmarkConfig(data_type="tabular")
        assert cfg.data_type == "tabular"


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


class TestYAMLRoundTrip:
    """YAML serialization/deserialization preserves all public fields."""

    def _assert_roundtrip(self, cfg, cls, tmp_path):
        path = tmp_path / "config.yaml"
        cfg.to_yaml(path)
        restored = cls.from_yaml(path)
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            assert getattr(restored, f.name) == getattr(cfg, f.name), (
                f"Field {f.name!r} differs after YAML round-trip"
            )

    def test_robot_config_roundtrip(self, tmp_path):
        cfg = RobotBenchmarkConfig(
            seed=42,
            subconcept=True,
            intervention_regimes=["baseline", "expert"],
            concept_missing=0.3,
            concept_missing_mech="mcar",
        )
        self._assert_roundtrip(cfg, RobotBenchmarkConfig, tmp_path)

    def test_sudoku_config_roundtrip(self, tmp_path):
        cfg = SudokuBenchmarkConfig(
            seed=99,
            max_corrupt=21,
            target_accuracy=0.99,
        )
        self._assert_roundtrip(cfg, SudokuBenchmarkConfig, tmp_path)

    def test_robot_text_config_roundtrip(self, tmp_path):
        cfg = RobotTextBenchmarkConfig(
            seed=7,
            difficulty="medium",
            intervention_regimes=["baseline", "subjective"],
        )
        self._assert_roundtrip(cfg, RobotTextBenchmarkConfig, tmp_path)


class TestYAMLSecretExclusion:
    """Ensure sensitive fields are excluded from YAML output."""

    def test_robot_yaml_excludes_llm_api_key(self, tmp_path):
        cfg = RobotBenchmarkConfig(llm_api_key="super_secret_key")
        path = tmp_path / "robot.yaml"
        cfg.to_yaml(path)
        content = path.read_text()
        # The secret value should not appear; llm_api_key_env is a different field
        assert "super_secret_key" not in content
        # The exact key "llm_api_key:" should not be a YAML key
        # (llm_api_key_env is fine — it's not the secret)
        lines = content.splitlines()
        yaml_keys = [l.split(":")[0].strip() for l in lines if ":" in l]
        assert "llm_api_key" not in yaml_keys

    def test_robot_text_yaml_excludes_llm_api_key(self, tmp_path):
        cfg = RobotTextBenchmarkConfig()
        path = tmp_path / "robot_text.yaml"
        cfg.to_yaml(path)
        content = path.read_text()
        lines = content.splitlines()
        yaml_keys = [l.split(":")[0].strip() for l in lines if ":" in l]
        assert "llm_api_key" not in yaml_keys

    def test_sudoku_yaml_includes_all_fields(self, tmp_path):
        cfg = SudokuBenchmarkConfig(seed=99, n_samples=500)
        path = tmp_path / "sudoku.yaml"
        cfg.to_yaml(path)
        content = path.read_text()
        assert "seed" in content
        assert "n_samples" in content
