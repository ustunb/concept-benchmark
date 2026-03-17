"""Tests for RobotDatasetGenerator and SudokuDatasetGenerator."""

from __future__ import annotations

import numpy as np

from concept_benchmark.data import ConceptDataset
from concept_benchmark.generators import RobotDatasetGenerator, SudokuDatasetGenerator


# ── Robot Generator ──────────────────────────────────────────────────


class TestRobotDatasetGenerator:
    def test_default_generates_ideal_shapes(self):
        ds = RobotDatasetGenerator(draw=False).generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.training.C.shape == (3800, 7)
        assert ds.training.y.shape == (3800,)
        assert ds.test.C.shape[1] == 7
        assert ds.validation is not None

    def test_subconcept_generates_12_concepts(self):
        ds = RobotDatasetGenerator(draw=False, subconcept=True).generate()
        assert ds.training.C.shape == (3800, 12)

    def test_label_formula_decomposes_correctly(self):
        gen = RobotDatasetGenerator(
            draw=False,
            label_formula={
                ("mouth_type", "open"): 3.0,
                ("has_knees", "true"): -2.0,
                "intercept": 1.0,
            },
        )
        assert gen.config.model_features == {
            "mouth_type": "open",
            "has_knees": "true",
        }
        assert gen.config.model_weights == {
            "mouth_type": 3.0,
            "has_knees": -2.0,
        }
        assert gen.config.model_intercept == 1.0

    def test_label_formula_without_intercept(self):
        gen = RobotDatasetGenerator(
            draw=False,
            label_formula={("mouth_type", "closed"): 1.0},
        )
        assert gen.config.model_intercept == 0.0

    def test_label_formula_matches_explicit_params(self):
        ds_formula = RobotDatasetGenerator(
            seed=1014,
            draw=False,
            label_formula={
                ("mouth_type", "closed"): 5.0,
                ("foot_shape", "pointy"): 8.0,
                ("has_knees", "true"): -5.0,
                "intercept": 2.0,
            },
        ).generate()
        ds_explicit = RobotDatasetGenerator(seed=1014, draw=False).generate()
        np.testing.assert_array_equal(ds_formula.training.y, ds_explicit.training.y)
        np.testing.assert_array_equal(ds_formula.training.C, ds_explicit.training.C)

    def test_reproducible_with_same_seed(self):
        ds1 = RobotDatasetGenerator(seed=42, draw=False).generate()
        ds2 = RobotDatasetGenerator(seed=42, draw=False).generate()
        np.testing.assert_array_equal(ds1.training.y, ds2.training.y)
        np.testing.assert_array_equal(ds1.training.C, ds2.training.C)

    def test_different_seeds_differ(self):
        ds1 = RobotDatasetGenerator(seed=1, draw=False).generate()
        ds2 = RobotDatasetGenerator(seed=2, draw=False).generate()
        assert not np.array_equal(ds1.training.y, ds2.training.y)

    def test_config_is_accessible(self):
        gen = RobotDatasetGenerator(seed=99, draw=False)
        assert gen.config.seed == 99
        assert gen.config.draw is False

    def test_imports_from_top_level(self):
        from concept_benchmark import RobotDatasetGenerator as RDG

        assert RDG is RobotDatasetGenerator


# ── Sudoku Generator ─────────────────────────────────────────────────


class TestSudokuDatasetGenerator:
    def test_default_generates_correct_shapes(self):
        ds = SudokuDatasetGenerator(data_type="tabular").generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.training.C.shape[1] == 27
        assert ds.training.y.ndim == 1
        assert ds.test is not None
        assert ds.validation is not None

    def test_split_sizes_sum_to_total(self):
        ds = SudokuDatasetGenerator(n_samples=100, data_type="tabular").generate()
        total = len(ds.training.y) + len(ds.validation.y) + len(ds.test.y)
        assert total == 100

    def test_reproducible_with_same_seed(self):
        ds1 = SudokuDatasetGenerator(seed=7, data_type="tabular").generate()
        ds2 = SudokuDatasetGenerator(seed=7, data_type="tabular").generate()
        np.testing.assert_array_equal(ds1.training.y, ds2.training.y)
        np.testing.assert_array_equal(ds1.training.C, ds2.training.C)

    def test_different_seeds_differ(self):
        ds1 = SudokuDatasetGenerator(seed=1, n_samples=200, data_type="tabular").generate()
        ds2 = SudokuDatasetGenerator(seed=2, n_samples=200, data_type="tabular").generate()
        # y is stratified 50/50, so compare concepts (actual board content)
        assert not np.array_equal(ds1.training.C, ds2.training.C)

    def test_custom_params(self):
        ds = SudokuDatasetGenerator(
            seed=42, n_samples=50, max_corrupt=3, data_type="tabular"
        ).generate()
        assert len(ds.training.y) + len(ds.validation.y) + len(ds.test.y) == 50

    def test_image_generates_png_boards(self):
        ds = SudokuDatasetGenerator(
            seed=42, n_samples=10, data_type="image"
        ).generate()
        assert isinstance(ds, ConceptDataset)
        assert ds.training.C.shape[1] == 27
        # Image data: X should contain image arrays (H, W, C) or paths
        assert ds.training.X is not None
        assert len(ds.training.y) + len(ds.validation.y) + len(ds.test.y) == 10

    def test_config_is_accessible(self):
        gen = SudokuDatasetGenerator(seed=55)
        assert gen.config.seed == 55

    def test_imports_from_top_level(self):
        from concept_benchmark import SudokuDatasetGenerator as SDG

        assert SDG is SudokuDatasetGenerator
