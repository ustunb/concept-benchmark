"""Tests for the additive robot hierarchy overlay."""

from __future__ import annotations

import numpy as np

from concept_benchmark.config import ROBOT_CONCEPTS
from concept_benchmark.robots import (
    DatasetGenerator,
    HierarchyImplication,
    RobotConceptHierarchy,
)


def test_from_dataset_infers_image_foot_hierarchy():
    ds = DatasetGenerator(render_images=False, concept_preset="foot_subtypes").generate()
    hierarchy = RobotConceptHierarchy.from_dataset(ds)

    expected_children = tuple(
        f"foot_shape_{value}" for value in ROBOT_CONCEPTS["foot_shape"]
    )
    assert hierarchy.children("foot_shape") == expected_children
    assert hierarchy.parents("foot_shape_pointy_square") == ("foot_shape",)
    assert hierarchy.group("foot_shape_pointy_square") == "foot_shape"
    assert hierarchy.implied_parent_value("foot_shape_pointy_square") == (
        "foot_shape",
        "pointy",
    )
    assert "foot_shape_flat_trapezoid" in hierarchy.siblings(
        "foot_shape_pointy_square"
    )
    assert HierarchyImplication(
        source="foot_shape_pointy_square",
        target="foot_shape",
        value=1,
    ) in hierarchy.implies("foot_shape_pointy_square")
    assert hierarchy.validate_matrix(ds.C) == []


def test_from_dataset_infers_text_foot_hierarchy_without_mutating_dataset():
    ds = DatasetGenerator(data_type="text", concept_preset="foot_subtypes").generate()
    original_names = tuple(ds.meta["concepts"])

    hierarchy = RobotConceptHierarchy.from_dataset(ds)

    assert tuple(ds.meta["concepts"]) == original_names
    assert hierarchy.children("foot_shape") == tuple(
        f"foot_shape_{value}" for value in ROBOT_CONCEPTS["foot_shape"]
    )
    assert hierarchy.implied_parent_value("foot_shape_flat_square") == (
        "foot_shape",
        "flat",
    )
    assert hierarchy.implies("foot_shape_flat_square") == ()
    assert hierarchy.validate_matrix(ds.C) == []


def test_validate_matrix_detects_group_violations():
    hierarchy = RobotConceptHierarchy.from_concept_names(
        [
            "foot_shape_flat_square",
            "foot_shape_pointy_square",
            "mouth_is_open",
        ]
    )
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)

    violations = hierarchy.validate_matrix(matrix)

    assert len(violations) == 1
    assert violations[0].relation == "group"
    assert violations[0].concepts == (
        "foot_shape_flat_square",
        "foot_shape_pointy_square",
    )


def test_validate_matrix_detects_leaf_to_parent_implication_violations():
    hierarchy = RobotConceptHierarchy.from_concept_names(
        [
            "foot_shape",
            "foot_shape_flat_square",
            "foot_shape_pointy_square",
        ]
    )
    matrix = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int8)

    violations = hierarchy.validate_matrix(matrix)

    assert len(violations) == 1
    assert violations[0].relation == "implies"
    assert violations[0].concepts == ("foot_shape_pointy_square", "foot_shape")


def test_implies_can_target_text_binary_parent_when_present():
    hierarchy = RobotConceptHierarchy.from_concept_names(
        [
            "foot_is_pointy",
            "foot_shape_flat_square",
            "foot_shape_pointy_square",
        ]
    )

    assert hierarchy.implies("foot_shape_pointy_square") == (
        HierarchyImplication(
            source="foot_shape_pointy_square",
            target="foot_is_pointy",
            value=1,
        ),
    )
    assert hierarchy.implies("foot_shape_flat_square") == (
        HierarchyImplication(
            source="foot_shape_flat_square",
            target="foot_is_pointy",
            value=0,
        ),
    )
