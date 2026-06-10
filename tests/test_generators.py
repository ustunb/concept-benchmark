"""Tests for the unified DatasetGenerator API."""

from __future__ import annotations

import numpy as np
import pytest

from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS, RobotBenchmarkConfig
from concept_benchmark.data import ConceptDataset
from concept_benchmark.generators import DatasetGenerator

SMALL_ROBOT_CONCEPTS = {
    "head_shape": ["square", "round"],
    "body_shape": ["square", "round"],
    "has_knees": ["false", "true"],
    "has_elbows": ["true"],
    "has_antennae": ["false"],
    "ears_shape": ["square"],
    "mouth_type": ["closed"],
    "hand_shape": ["round_circle"],
    "foot_shape": ["pointy_4sided"],
}


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
        gen = DatasetGenerator("robot", render_images=False)
        ds = gen.generate()
        assert isinstance(ds, ConceptDataset)
        ds.drop_concepts(PRESET_EXCLUDED_CONCEPTS["ground_truth"])
        ds.sample(test_size=10000, val_size=0.2, train_size=3800, seed=1014)
        assert ds.train.C.shape == (3800, 7)
        assert ds.train.y.shape == (3800,)
        assert ds.test.C.shape[1] == 7
        assert ds.validation.n > 0

    def test_subconcept_generates_12_concepts(self):
        gen = DatasetGenerator(
            "robot", render_images=False, concept_preset="foot_subtypes"
        )
        ds = gen.generate()
        ds.drop_concepts(PRESET_EXCLUDED_CONCEPTS["foot_subtypes"])
        ds.sample(test_size=10000, val_size=0.2, train_size=3800, seed=1014)
        assert ds.train.C.shape == (3800, 12)

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
        assert gen.config.label_formula.features == {
            "mouth_type": "open",
            "has_knees": "true",
        }
        assert gen.config.label_formula.weights == {
            "mouth_type": 3.0,
            "has_knees": -2.0,
        }
        assert gen.config.label_formula.intercept == 1.0

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
        assert gen.config.label_formula.intercept == 0.0

    def test_label_formula_via_config_matches_kwargs(self):
        """Verify that label_formula via config matches DatasetGenerator kwargs."""
        formula = {
            "terms": {
                "mouth_type": {"value": "closed", "weight": 5.0},
                "has_knees": {"value": "true", "weight": -5.0},
            },
            "intercept": 2.0,
        }
        ds1 = DatasetGenerator(
            "robot", seed=1014, render_images=False, label_formula=formula
        ).generate()
        cfg = RobotBenchmarkConfig(
            seed=1014,
            render_images=False,
            label_formula=formula,
        )
        ds2 = DatasetGenerator.from_config(cfg).generate()
        np.testing.assert_array_equal(ds1.y, ds2.y)
        np.testing.assert_array_equal(ds1.C, ds2.C)

    def test_reproducible_with_same_seed(self):
        ds1 = DatasetGenerator("robot", seed=42, render_images=False).generate()
        ds2 = DatasetGenerator("robot", seed=42, render_images=False).generate()
        np.testing.assert_array_equal(ds1.y, ds2.y)
        np.testing.assert_array_equal(ds1.C, ds2.C)

    def test_different_seeds_differ(self):
        ds1 = DatasetGenerator("robot", seed=1, render_images=False).generate()
        ds2 = DatasetGenerator("robot", seed=2, render_images=False).generate()
        assert not np.array_equal(ds1.y, ds2.y)

    def test_config_is_accessible(self):
        gen = DatasetGenerator("robot", seed=99, render_images=False)
        assert gen.config.seed == 99
        assert gen.config.render_images is False
        assert gen.benchmark == "robot"

    def test_label_formula_rejects_unknown_feature(self):
        with pytest.raises(ValueError, match="not found in concepts"):
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
        with pytest.raises(ValueError, match="not valid for feature"):
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
        assert gen.config.label_formula.temperature == 10.0

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
        assert gen.config.label_formula.features == {"foot_shape": "pointy"}
        assert gen.config.label_formula.weights == {"foot_shape": 8.0}


# ── Sudoku via DatasetGenerator ──────────────────────────────────────


class TestSudokuDatasetGenerator:
    def test_default_generates_correct_shapes(self):
        ds = DatasetGenerator("sudoku", data_type="tabular").generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.C.shape[1] == 27
        assert ds.y.ndim == 1
        ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=42)
        assert ds.test.n > 0
        assert ds.validation.n > 0

    def test_split_sizes_sum_to_total(self):
        ds = DatasetGenerator("sudoku", n_boards=100, data_type="tabular").generate()
        ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=42)
        total = len(ds.train.y) + len(ds.validation.y) + len(ds.test.y)
        assert total == 100

    def test_reproducible_with_same_seed(self):
        ds1 = DatasetGenerator("sudoku", seed=7, data_type="tabular").generate()
        ds2 = DatasetGenerator("sudoku", seed=7, data_type="tabular").generate()
        np.testing.assert_array_equal(ds1.y, ds2.y)
        np.testing.assert_array_equal(ds1.C, ds2.C)

    def test_different_seeds_differ(self):
        ds1 = DatasetGenerator(
            "sudoku", seed=1, n_boards=200, data_type="tabular"
        ).generate()
        ds2 = DatasetGenerator(
            "sudoku", seed=2, n_boards=200, data_type="tabular"
        ).generate()
        assert not np.array_equal(ds1.C, ds2.C)

    def test_custom_params(self):
        ds = DatasetGenerator(
            "sudoku", seed=42, n_boards=50, max_cell_swaps=3, data_type="tabular"
        ).generate()
        ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=42)
        assert len(ds.train.y) + len(ds.validation.y) + len(ds.test.y) == 50

    def test_image_generates_png_boards(self):
        ds = DatasetGenerator(
            "sudoku", seed=42, n_boards=10, data_type="image"
        ).generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.C.shape[1] == 27
        assert ds.inputs is not None
        ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=42)
        assert len(ds.train.y) + len(ds.validation.y) + len(ds.test.y) == 10

    def test_boards_stored_in_meta(self):
        ds = DatasetGenerator(
            "sudoku", seed=42, n_boards=20, data_type="tabular"
        ).generate()
        boards = ds.meta.get("boards")
        assert boards is not None
        assert boards.shape == (20, 9, 9)
        assert boards.min() >= 1
        assert boards.max() <= 9

    def test_config_is_accessible(self):
        gen = DatasetGenerator("sudoku", seed=55)
        assert gen.config.seed == 55
        assert gen.benchmark == "sudoku"
