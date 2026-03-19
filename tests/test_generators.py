"""Tests for the unified DatasetGenerator API."""

from __future__ import annotations

import numpy as np
import pytest

from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.data import ConceptDataset
from concept_benchmark.generators import DatasetGenerator, generate_robot_dataset


# ── DatasetGenerator meta ────────────────────────────────────────────


class TestDatasetGeneratorMeta:
    def test_available_benchmarks(self):
        assert DatasetGenerator.available_benchmarks() == ["robot", "sudoku"]

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark 'nonexistent'"):
            DatasetGenerator("nonexistent")

    def test_wrong_param_raises_with_message(self):
        with pytest.raises(
            ValueError, match="Invalid parameter.*'sudoku'.*concept_preset"
        ):
            DatasetGenerator("sudoku", concept_preset="foot_subtypes")

    def test_wrong_param_for_robot_raises(self):
        with pytest.raises(ValueError, match="Invalid parameter.*'robot'.*block_size"):
            DatasetGenerator("robot", block_size=3)

    def test_imports_from_top_level(self):
        from concept_benchmark import DatasetGenerator as DG

        assert DG is DatasetGenerator

    def test_text_routing_constructs_correctly(self):
        gen = DatasetGenerator("robot", data_type="text", seed=1337)
        assert isinstance(gen.config, RobotBenchmarkConfig)
        assert gen.config.data_type == "text"
        assert gen.config.seed == 1337
        assert gen.benchmark == "robot"

    def test_text_wrong_param_raises(self):
        with pytest.raises(ValueError, match="Invalid parameter.*'robot'.*block_size"):
            DatasetGenerator("robot", data_type="text", block_size=3)


# ── Robot via DatasetGenerator ───────────────────────────────────────


class TestRobotDatasetGenerator:
    def test_default_generates_ideal_shapes(self):
        ds = DatasetGenerator("robot", render_images=False).generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.training.C.shape == (3800, 7)
        assert ds.training.y.shape == (3800,)
        assert ds.test.C.shape[1] == 7
        assert ds.validation is not None

    def test_subconcept_generates_12_concepts(self):
        ds = DatasetGenerator(
            "robot", render_images=False, concept_preset="foot_subtypes"
        ).generate()
        assert ds.training.C.shape == (3800, 12)

    def test_label_formula_decomposes_correctly(self):
        gen = DatasetGenerator(
            "robot",
            render_images=False,
            label_formula={
                "terms": {
                    "mouth_type": {"value": "open", "weight": 3.0},
                    "has_knees": {"value": "true", "weight": -2.0},
                },
                "intercept": 1.0,
            },
        )
        assert gen.config.label_features == {
            "mouth_type": "open",
            "has_knees": "true",
        }
        assert gen.config.label_weights == {
            "mouth_type": 3.0,
            "has_knees": -2.0,
        }
        assert gen.config.label_intercept == 1.0

    def test_label_formula_without_intercept(self):
        gen = DatasetGenerator(
            "robot",
            render_images=False,
            label_formula={
                "terms": {
                    "mouth_type": {"value": "closed", "weight": 1.0},
                },
            },
        )
        assert gen.config.label_intercept == 0.0

    def test_label_formula_via_config_matches_generator(self):
        """Verify that label_formula via RobotBenchmarkConfig matches DatasetGenerator."""
        formula = {
            "terms": {
                "mouth_type": {"value": "closed", "weight": 5.0},
                "has_knees": {"value": "true", "weight": -5.0},
            },
            "intercept": 2.0,
        }
        ds_generator = DatasetGenerator(
            "robot", seed=1014, render_images=False, label_formula=formula
        ).generate()
        cfg = RobotBenchmarkConfig(
            seed=1014,
            render_images=False,
            label_formula=formula,
        )
        ds_config = generate_robot_dataset(cfg)
        np.testing.assert_array_equal(ds_generator.training.y, ds_config.training.y)
        np.testing.assert_array_equal(ds_generator.training.C, ds_config.training.C)

    def test_reproducible_with_same_seed(self):
        ds1 = DatasetGenerator("robot", seed=42, render_images=False).generate()
        ds2 = DatasetGenerator("robot", seed=42, render_images=False).generate()
        np.testing.assert_array_equal(ds1.training.y, ds2.training.y)
        np.testing.assert_array_equal(ds1.training.C, ds2.training.C)

    def test_different_seeds_differ(self):
        ds1 = DatasetGenerator("robot", seed=1, render_images=False).generate()
        ds2 = DatasetGenerator("robot", seed=2, render_images=False).generate()
        assert not np.array_equal(ds1.training.y, ds2.training.y)

    def test_config_is_accessible(self):
        gen = DatasetGenerator("robot", seed=99, render_images=False)
        assert gen.config.seed == 99
        assert gen.config.render_images is False
        assert gen.benchmark == "robot"

    def test_label_formula_rejects_unknown_feature(self):
        with pytest.raises(ValueError, match="Unknown feature 'nonexistent'"):
            DatasetGenerator(
                "robot",
                render_images=False,
                label_formula={
                    "terms": {
                        "nonexistent": {"value": "x", "weight": 1.0},
                    },
                },
            )

    def test_label_formula_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid value 'nonexistent'"):
            DatasetGenerator(
                "robot",
                render_images=False,
                label_formula={
                    "terms": {
                        "mouth_type": {"value": "nonexistent", "weight": 1.0},
                    },
                },
            )

    def test_label_formula_temperature(self):
        gen = DatasetGenerator(
            "robot",
            render_images=False,
            label_formula={
                "terms": {
                    "mouth_type": {"value": "open", "weight": 3.0},
                },
                "intercept": 1.0,
                "temperature": 10.0,
            },
        )
        assert gen.config.label_temperature == 10.0

    def test_label_formula_accepts_prefix_value(self):
        gen = DatasetGenerator(
            "robot",
            render_images=False,
            label_formula={
                "terms": {
                    "foot_shape": {"value": "pointy", "weight": 8.0},
                },
            },
        )
        assert gen.config.label_features == {"foot_shape": "pointy"}
        assert gen.config.label_weights == {"foot_shape": 8.0}


# ── Sudoku via DatasetGenerator ──────────────────────────────────────


class TestSudokuDatasetGenerator:
    def test_default_generates_correct_shapes(self):
        ds = DatasetGenerator("sudoku", data_type="tabular").generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.training.C.shape[1] == 27
        assert ds.training.y.ndim == 1
        assert ds.test is not None
        assert ds.validation is not None

    def test_split_sizes_sum_to_total(self):
        ds = DatasetGenerator("sudoku", n_boards=100, data_type="tabular").generate()
        total = len(ds.training.y) + len(ds.validation.y) + len(ds.test.y)
        assert total == 100

    def test_reproducible_with_same_seed(self):
        ds1 = DatasetGenerator("sudoku", seed=7, data_type="tabular").generate()
        ds2 = DatasetGenerator("sudoku", seed=7, data_type="tabular").generate()
        np.testing.assert_array_equal(ds1.training.y, ds2.training.y)
        np.testing.assert_array_equal(ds1.training.C, ds2.training.C)

    def test_different_seeds_differ(self):
        ds1 = DatasetGenerator(
            "sudoku", seed=1, n_boards=200, data_type="tabular"
        ).generate()
        ds2 = DatasetGenerator(
            "sudoku", seed=2, n_boards=200, data_type="tabular"
        ).generate()
        # y is stratified 50/50, so compare concepts (actual board content)
        assert not np.array_equal(ds1.training.C, ds2.training.C)

    def test_custom_params(self):
        ds = DatasetGenerator(
            "sudoku", seed=42, n_boards=50, max_cell_swaps=3, data_type="tabular"
        ).generate()
        assert len(ds.training.y) + len(ds.validation.y) + len(ds.test.y) == 50

    def test_image_generates_png_boards(self):
        ds = DatasetGenerator(
            "sudoku", seed=42, n_boards=10, data_type="image"
        ).generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.training.C.shape[1] == 27
        # Image data: X should contain image arrays (H, W, C) or paths
        assert ds.training.X is not None
        assert len(ds.training.y) + len(ds.validation.y) + len(ds.test.y) == 10

    def test_boards_stored_in_meta(self):
        ds = DatasetGenerator(
            "sudoku", seed=42, n_boards=20, data_type="tabular"
        ).generate()
        boards = ds.meta.get("boards")
        assert boards is not None
        assert boards.shape == (20, 9, 9)
        # All values should be valid sudoku digits 1-9
        assert boards.min() >= 1
        assert boards.max() <= 9

    def test_config_is_accessible(self):
        gen = DatasetGenerator("sudoku", seed=55)
        assert gen.config.seed == 55
        assert gen.benchmark == "sudoku"
