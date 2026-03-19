"""Tests for config validation."""

from __future__ import annotations

from dataclasses import fields

import pytest
from concept_benchmark.config import (
    RobotBenchmarkConfig,
    SudokuBenchmarkConfig,
)


class TestRobotConfigValidation:
    def test_default_config_is_valid(self):
        cfg = RobotBenchmarkConfig()
        assert cfg.seed == 1014

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            RobotBenchmarkConfig(seed=-1)

    def test_rejects_negative_budget(self):
        with pytest.raises(
            ValueError, match="intervention_budgets must be non-negative"
        ):
            RobotBenchmarkConfig(intervention_budgets=[-2, 3])

    def test_rejects_unknown_regime(self):
        with pytest.raises(ValueError, match="unknown intervention regimes"):
            RobotBenchmarkConfig(intervention_regimes=["baseline", "bogus"])

    def test_rejects_bad_strategy(self):
        with pytest.raises(ValueError, match="intervention_strategy must be one of"):
            RobotBenchmarkConfig(intervention_strategy="random")

    def test_accepts_all_valid_regimes(self):
        cfg = RobotBenchmarkConfig(
            intervention_regimes=[
                "baseline",
                "expert",
                "subjective",
                "machine",
                "llm",
                "clip",
            ]
        )
        assert len(cfg.intervention_regimes) == 6

    def test_exactly_k_strategy(self):
        cfg = RobotBenchmarkConfig(intervention_strategy="exactly_k")
        assert cfg.intervention_strategy == "exactly_k"

    def test_concept_preset_foot_subtypes(self):
        cfg = RobotBenchmarkConfig(concept_preset="foot_subtypes")
        assert cfg.concept_preset == "foot_subtypes"

    def test_concept_preset_ground_truth_is_default(self):
        cfg = RobotBenchmarkConfig()
        assert cfg.concept_preset == "ground_truth"

    def test_rejects_bad_concept_preset(self):
        with pytest.raises(ValueError, match="concept_preset must be"):
            RobotBenchmarkConfig(concept_preset="invalid")

    def test_image_size_small(self):
        cfg = RobotBenchmarkConfig(image_size="small")
        assert cfg.image_size == "small"
        assert cfg.pixel_resolution == 8

    def test_image_size_large(self):
        cfg = RobotBenchmarkConfig(image_size="large")
        assert cfg.image_size == "large"
        assert cfg.pixel_resolution == 600

    def test_image_size_medium_is_default(self):
        cfg = RobotBenchmarkConfig()
        assert cfg.image_size == "medium"
        assert cfg.pixel_resolution == 32

    def test_rejects_bad_image_size(self):
        with pytest.raises(ValueError, match="image_size must be"):
            RobotBenchmarkConfig(image_size="huge")

    def test_label_formula_nested_format(self):
        cfg = RobotBenchmarkConfig(
            label_formula={
                "terms": {
                    "mouth_type": {"value": "closed", "weight": 5.0},
                    "has_knees": {"value": "true", "weight": -2.0},
                },
                "intercept": 1.0,
                "temperature": 3.0,
            }
        )
        assert cfg.label_features == {"mouth_type": "closed", "has_knees": "true"}
        assert cfg.label_weights == {"mouth_type": 5.0, "has_knees": -2.0}
        assert cfg.label_intercept == 1.0
        assert cfg.label_temperature == 3.0

    def test_label_formula_rejects_missing_terms(self):
        with pytest.raises(ValueError, match='"terms" key'):
            RobotBenchmarkConfig(
                label_formula={"mouth_type": 5.0}
            )

    def test_label_formula_rejects_bad_term_structure(self):
        with pytest.raises(ValueError, match="'value' and 'weight' keys"):
            RobotBenchmarkConfig(
                label_formula={
                    "terms": {"mouth_type": {"val": "closed"}},
                }
            )


class TestSudokuConfigValidation:
    def test_default_config_is_valid(self):
        cfg = SudokuBenchmarkConfig()
        assert cfg.seed == 171

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            SudokuBenchmarkConfig(seed=-1)

    def test_rejects_zero_boards(self):
        with pytest.raises(ValueError, match="n_boards must be positive"):
            SudokuBenchmarkConfig(n_boards=0)

    def test_default_data_type_is_image(self):
        cfg = SudokuBenchmarkConfig()
        assert cfg.data_type == "image"

    def test_explicit_tabular_data_type(self):
        cfg = SudokuBenchmarkConfig(data_type="tabular")
        assert cfg.data_type == "tabular"


class TestRobotTextConfigValidation:
    def test_text_config_is_valid(self):
        cfg = RobotBenchmarkConfig(data_type="text", seed=1337)
        assert cfg.seed == 1337
        assert cfg.data_type == "text"

    def test_rejects_negative_seed(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            RobotBenchmarkConfig(data_type="text", seed=-1)

    def test_rejects_unknown_regime(self):
        with pytest.raises(ValueError, match="unknown intervention regimes"):
            RobotBenchmarkConfig(data_type="text", intervention_regimes=["llm"])

    def test_rejects_bad_strategy(self):
        with pytest.raises(ValueError, match="intervention_strategy must be one of"):
            RobotBenchmarkConfig(data_type="text", intervention_strategy="random")

    def test_auto_switches_concepts_to_text(self):
        from concept_benchmark.config import TEXT_CONCEPTS
        cfg = RobotBenchmarkConfig(data_type="text")
        assert cfg.concepts == TEXT_CONCEPTS

    def test_clears_sampling_constraints(self):
        cfg = RobotBenchmarkConfig(data_type="text")
        assert cfg.sampling_constraints == []

    def test_label_formula_with_prefix_value(self):
        cfg = RobotBenchmarkConfig(
            data_type="text",
            label_formula={
                "terms": {
                    "mouth_type": {"value": "open", "weight": 3.0},
                    "foot_shape": {"value": "pointy", "weight": 8.0},
                },
                "intercept": 1.0,
            },
        )
        assert cfg.label_features == {"mouth_type": "open", "foot_shape": "pointy"}
        assert cfg.label_weights == {"mouth_type": 3.0, "foot_shape": 8.0}
        assert cfg.label_intercept == 1.0


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
            concept_preset="foot_subtypes",
            intervention_regimes=["baseline", "expert"],
            missing_fraction=0.3,
            missing_mechanism="mcar",
        )
        self._assert_roundtrip(cfg, RobotBenchmarkConfig, tmp_path)

    def test_sudoku_config_roundtrip(self, tmp_path):
        cfg = SudokuBenchmarkConfig(
            seed=99,
            max_cell_swaps=21,
            target_accuracy=0.99,
        )
        self._assert_roundtrip(cfg, SudokuBenchmarkConfig, tmp_path)

    def test_robot_text_config_roundtrip(self, tmp_path):
        cfg = RobotBenchmarkConfig(
            data_type="text",
            seed=7,
            template_complexity="medium",
            intervention_regimes=["baseline", "subjective"],
        )
        self._assert_roundtrip(cfg, RobotBenchmarkConfig, tmp_path)


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
        yaml_keys = [line.split(":")[0].strip() for line in lines if ":" in line]
        assert "llm_api_key" not in yaml_keys

    def test_robot_text_yaml_excludes_llm_api_key(self, tmp_path):
        cfg = RobotBenchmarkConfig(data_type="text")
        path = tmp_path / "robot_text.yaml"
        cfg.to_yaml(path)
        content = path.read_text()
        lines = content.splitlines()
        yaml_keys = [line.split(":")[0].strip() for line in lines if ":" in line]
        assert "llm_api_key" not in yaml_keys

    def test_sudoku_yaml_includes_all_fields(self, tmp_path):
        cfg = SudokuBenchmarkConfig(seed=99, n_boards=500)
        path = tmp_path / "sudoku.yaml"
        cfg.to_yaml(path)
        content = path.read_text()
        assert "seed" in content
        assert "n_boards" in content
