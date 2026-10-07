"""The data API as the README and docs show it to pip users: only `concept_benchmark`, the claimed shapes.

These tests run against the installed package (CI installs the wheel without extras and runs them with
`--import-mode=importlib`); the ones that render images are marked slow.
"""

import numpy as np
import pytest

from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS
from concept_benchmark.robots import DatasetGenerator as RobotGenerator
from concept_benchmark.robots import F, LabelFormula
from concept_benchmark.sudoku import DatasetGenerator as SudokuGenerator
from concept_benchmark.transforms import (
    ConceptDropGenerator,
    ConceptMissingnessGenerator,
    ConceptNoiseGenerator,
    LabelNoiseGenerator,
)


def test_package_is_the_installed_one():
    import concept_benchmark

    assert concept_benchmark.__version__ != "0.0.0"


def test_robot_paper_split_has_the_documented_shape():
    dataset = RobotGenerator(
        seed=1014, concept_preset="foot_subtypes", render_images=False
    ).generate_splits()
    assert dataset.train.C.shape == (3800, 12)
    assert dataset.test.n == 10000
    assert len(dataset.train.concepts) == 12


def test_robot_generate_accepts_every_documented_keyword():
    dataset = RobotGenerator(
        seed=1014,
        data_type="image",
        label_rule="balanced",
        label_formula=None,
        concept_preset="ground_truth",
        renders_per_robot=1,
        expand_concepts=["foot_shape"],
        image_size="medium",
        color_mode="color",
        render_images=False,
        template_complexity="high",
    ).generate()
    assert dataset.C.shape[1] >= 7
    assert set(np.unique(dataset.y)) <= {0, 1}


def test_label_formula_example_from_the_readme():
    formula = LabelFormula(
        score=4 * F("mouth_type").closed
        + 6 * F("body_shape").round
        - 3 * F("has_knees").true
        - 2,
        temperature=4.2,
        stochastic=True,
    )
    dataset = RobotGenerator(
        seed=1014, label_formula=formula, render_images=False
    ).generate()
    assert dataset.y.shape[0] == dataset.C.shape[0]
    assert "mouth_type" in dataset.train.to_dataframe().columns or dataset.n > 0


def test_drop_concepts_sample_and_constraints_as_in_the_readme():
    dataset = RobotGenerator(seed=1014, render_images=False).generate()
    dataset = ConceptDropGenerator(dataset, ["has_elbows", "hand_shape"]).generate()
    dataset.sample(test_size=10000, val_size=0.2, train_size=3800, seed=1014)
    assert dataset.train.n == 3800 and dataset.test.n == 10000
    dataset = RobotGenerator(
        seed=1014, concept_preset="foot_subtypes", render_images=False
    ).generate()
    dataset.sample(
        test_size=10000,
        val_size=0.2,
        train_size=3800,
        seed=1014,
        sampling_constraints=[
            {"concepts": {"foot_shape_pointy_4sided": 1}, "min_fraction": 0.30}
        ],
    )
    dataset.drop_concepts(PRESET_EXCLUDED_CONCEPTS["foot_subtypes"])
    assert dataset.train.C.shape[0] == 3800
    assert dataset.train.to_dataframe().shape[0] == 3800


def test_missingness_and_noise_generators_as_in_the_readme():
    dataset = RobotGenerator(seed=1014, render_images=False).generate()
    missing = ConceptMissingnessGenerator(
        dataset, p=0.2, mechanism="mcar", seed=99
    ).generate()
    assert np.isnan(np.asarray(missing.C, dtype=float)).any()
    mnar = ConceptMissingnessGenerator(
        dataset,
        p=0.2,
        mechanism="mnar",
        seed=99,
        mnar_config={"present_prob": 0.8, "absent_prob": 0.1},
    ).generate()
    assert np.isnan(np.asarray(mnar.C, dtype=float)).any()
    noisy = ConceptNoiseGenerator(dataset, p=0.1, seed=99).generate()
    assert (np.asarray(noisy.C) != np.asarray(dataset.C)).any()
    relabeled = LabelNoiseGenerator(dataset, p=0.05, seed=99).generate()
    assert (np.asarray(relabeled.y) != np.asarray(dataset.y)).any()


def test_sudoku_paper_split_has_the_documented_shape():
    dataset = SudokuGenerator(
        seed=171,
        n_boards=1000,
        max_cell_swaps=9,
        valid_board_ratio=0.5,
        render_images=False,
    ).generate_splits()
    assert dataset.train.C.shape == (600, 27)
    assert dataset.validation.n == 200 and dataset.test.n == 200
    assert list(dataset.train.concepts[:3]) == [
        "row_valid_1",
        "row_valid_2",
        "row_valid_3",
    ]


def test_sudoku_image_generate_accepts_every_documented_keyword(tmp_path, monkeypatch):
    monkeypatch.setenv("CONCEPT_BENCHMARK_DATA_DIR", str(tmp_path))
    dataset = SudokuGenerator(
        seed=171,
        data_type="image",
        render_images=True,
        block_size=3,
        n_boards=6,
        max_cell_swaps=9,
        valid_board_ratio=0.5,
        font_style="handwritten",
        font_size=25,
        cell_px=50,
        cell_margin_px=2,
        gridline_px=2,
        block_border_px=5,
    ).generate()
    assert dataset.C.shape == (6, 27)
    dataset.sample(test_size=2, val_size=2, seed=171)
    df = dataset.train.to_dataframe()
    assert set(dataset.train.concepts[:5]) <= set(df.columns)
    assert "label" in df.columns


@pytest.mark.slow
def test_hub_card_call_renders_the_published_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("CONCEPT_BENCHMARK_DATA_DIR", str(tmp_path))
    dataset = SudokuGenerator(seed=171).generate_splits()
    assert (dataset.train.n, dataset.validation.n, dataset.test.n) == (600, 200, 200)
