"""The two robot label rules and the paper's train/validation/test split."""

import numpy as np
import pytest
from scipy.special import expit

from concept_benchmark.config import (
    ROBOT_LABEL_RULES,
    ROBOT_TEST_SIZE,
    ROBOT_TRAIN_SIZE,
    RobotBenchmarkConfig,
)
from concept_benchmark.generators import DatasetGenerator
from concept_benchmark.synthetic.robot.catalog import generate_robot_catalog


def _glorp_probability(rule: str) -> np.ndarray:
    config = RobotBenchmarkConfig(label_rule=rule)
    settings = config.to_dict()
    catalog, _ = generate_robot_catalog(
        concepts=settings["concepts"],
        additional_features=settings["additional_features"],
        draw=False,
    )
    score = catalog.apply(config.label_formula.score, axis=1).to_numpy(float)
    return expit(config.label_formula.temperature * score)


def test_balanced_is_the_default_rule():
    assert RobotBenchmarkConfig().label_rule == "balanced"


def test_balanced_rule_makes_both_classes_equally_likely():
    assert _glorp_probability("balanced").mean() == pytest.approx(0.5, abs=1e-6)


def test_sparse_rule_makes_one_class_rare():
    assert _glorp_probability("sparse").mean() == pytest.approx(0.875, abs=1e-3)


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="label_rule"):
        RobotBenchmarkConfig(label_rule="uniform")


def test_custom_formula_overrides_the_rule():
    formula = ROBOT_LABEL_RULES["sparse"].build_formula()
    config = RobotBenchmarkConfig(label_rule="balanced", label_formula=formula)
    assert config.label_formula is formula


def test_sparse_rule_files_do_not_share_names_with_the_default():
    balanced, sparse = RobotBenchmarkConfig(), RobotBenchmarkConfig(label_rule="sparse")
    assert balanced.get_dataset_path() != sparse.get_dataset_path()
    assert balanced.get_model_path("cbm") != sparse.get_model_path("cbm")


@pytest.mark.parametrize("rule", sorted(ROBOT_LABEL_RULES))
@pytest.mark.parametrize(
    "preset, n_concepts", [("ground_truth", 7), ("foot_subtypes", 12)]
)
def test_generate_splits_follows_the_rule(rule, preset, n_concepts):
    data = DatasetGenerator(
        "robot", seed=1014, label_rule=rule, concept_preset=preset, render_images=False
    ).generate_splits()
    assert data.train.n == ROBOT_TRAIN_SIZE
    assert data.test.n == ROBOT_TEST_SIZE
    assert data.train.C.shape[1] == n_concepts
    catalog = data.meta["catalog_df"]
    subtype = (
        catalog["foot_shape"].astype(str)
        + "_"
        + catalog["foot_shape_subtype"].astype(str)
    )
    train_shares = subtype.iloc[
        catalog.index.get_indexer(data.train.meta["df_indices"])
    ].value_counts(normalize=True)
    for constraint in ROBOT_LABEL_RULES[rule].sampling_constraints:
        (name,) = constraint["concepts"]
        assert (
            train_shares[name.removeprefix("foot_shape_")]
            >= constraint["min_fraction"] - 1e-3
        )
