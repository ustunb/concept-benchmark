"""Hierarchy helpers for robot benchmark concept names.

This module is intentionally additive: it infers hierarchical structure
from the existing robot concept naming conventions without modifying
dataset generation, configs, or metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from concept_benchmark.config import ROBOT_CONCEPTS

__all__ = [
    "HierarchyImplication",
    "HierarchyViolation",
    "RobotConceptHierarchy",
]


_BINARY_PARENT_CONCEPTS: dict[str, str] = {
    "head_shape": "head_is_square",
    "body_shape": "body_is_square",
    "has_knees": "has_knees",
    "has_elbows": "has_elbows",
    "foot_shape": "foot_is_pointy",
    "has_antennae": "has_antennae",
    "ears_shape": "ears_is_triangle",
    "mouth_type": "mouth_is_open",
    "hand_shape": "hands_are_pointy",
}


def _positive_value_for_feature(
    feature: str,
    concept_pos_value: Mapping[str, str] | None = None,
) -> str | None:
    if concept_pos_value and feature in concept_pos_value:
        return str(concept_pos_value[feature])

    values = ROBOT_CONCEPTS.get(feature)
    if not values:
        return None

    coarse_values = list(dict.fromkeys(str(value).split("_", 1)[0] for value in values))
    if not coarse_values:
        return None
    if len(coarse_values) == 1:
        return coarse_values[0]
    return coarse_values[1]


@dataclass(frozen=True)
class HierarchyImplication:
    """A concept-level implication relation inferred from hierarchy."""

    source: str
    target: str
    value: int
    relation: Literal["implies"] = "implies"


@dataclass(frozen=True)
class HierarchyViolation:
    """A failed group or implication constraint for one row."""

    row_index: int
    relation: Literal["group", "implies"]
    concepts: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class _LeafInfo:
    concept: str
    feature: str
    feature_value: str
    coarse_value: str


class RobotConceptHierarchy:
    """Read-only hierarchy overlay inferred from existing concept names.

    Supports three relation types over the current robots benchmark:

    - ``refines``: subtype concepts such as ``foot_shape_pointy_square``
      refine a feature family such as ``foot_shape``.
    - ``group``: subtype concepts in the same feature family are siblings.
    - ``implies``: a subtype can imply a coarse binary concept when that
      concept is present in the current concept set.
    """

    def __init__(
        self,
        concept_names: Sequence[str],
        *,
        concept_pos_value: Mapping[str, str] | None = None,
    ) -> None:
        self.concept_names = tuple(str(name) for name in concept_names)
        self._concept_name_set = frozenset(self.concept_names)
        self._concept_pos_value = (
            {str(k): str(v) for k, v in concept_pos_value.items()}
            if concept_pos_value
            else {}
        )
        self._leaf_infos = self._infer_leaf_infos()
        self._group_members = self._build_group_members()

    @classmethod
    def from_concept_names(
        cls,
        concept_names: Sequence[str],
        *,
        concept_pos_value: Mapping[str, str] | None = None,
    ) -> "RobotConceptHierarchy":
        """Build a hierarchy overlay from a concept-name sequence."""

        return cls(concept_names, concept_pos_value=concept_pos_value)

    @classmethod
    def from_dataset(cls, dataset: Any) -> "RobotConceptHierarchy":
        """Build a hierarchy overlay from a dataset or dataset sample."""

        meta = getattr(dataset, "meta", None)
        concept_names = None
        concept_pos_value = None
        if isinstance(meta, Mapping):
            concept_names = meta.get("concepts")
            concept_pos_value = meta.get("concept_pos_value")
        if concept_names is None:
            concept_names = getattr(dataset, "concepts", None)
        if concept_names is None:
            raise ValueError("dataset must expose concept names via .meta or .concepts")
        return cls(concept_names, concept_pos_value=concept_pos_value)

    def parents(self, concept: str) -> tuple[str, ...]:
        """Return the feature family that concept refines, if any."""

        info = self._leaf_infos.get(concept)
        if info is None:
            return ()
        return (info.feature,)

    def children(self, concept: str) -> tuple[str, ...]:
        """Return leaf concepts in the given feature family."""

        return self._group_members.get(concept, ())

    def siblings(self, concept: str) -> tuple[str, ...]:
        """Return other leaf concepts in the same feature family."""

        group = self.group(concept)
        if group is None:
            return ()
        return tuple(name for name in self._group_members[group] if name != concept)

    def group(self, concept: str) -> str | None:
        """Return the feature family for a leaf concept, if any."""

        info = self._leaf_infos.get(concept)
        if info is None:
            return None
        return info.feature

    def implied_parent_value(self, concept: str) -> tuple[str, str] | None:
        """Return the coarse feature/value implied by a leaf concept."""

        info = self._leaf_infos.get(concept)
        if info is None:
            return None
        return (info.feature, info.coarse_value)

    def implies(self, concept: str) -> tuple[HierarchyImplication, ...]:
        """Return concept-level implications available in this concept set."""

        info = self._leaf_infos.get(concept)
        if info is None:
            return ()

        positive_value = _positive_value_for_feature(
            info.feature,
            concept_pos_value=self._concept_pos_value,
        )
        if positive_value is None:
            return ()

        value = 1 if info.coarse_value == positive_value else 0
        implications: list[HierarchyImplication] = []

        if info.feature in self._concept_name_set:
            implications.append(
                HierarchyImplication(source=concept, target=info.feature, value=value)
            )

        binary_target = _BINARY_PARENT_CONCEPTS.get(info.feature)
        if binary_target and binary_target in self._concept_name_set:
            implications.append(
                HierarchyImplication(source=concept, target=binary_target, value=value)
            )

        return tuple(implications)

    def validate_matrix(
        self,
        C: Any,
        *,
        concept_names: Sequence[str] | None = None,
        threshold: float = 0.5,
        max_violations: int | None = 100,
    ) -> list[HierarchyViolation]:
        """Validate group and implication consistency for a concept matrix.

        Group validation enforces an ``at most one active leaf`` rule within
        each subtype family. Implication validation enforces leaf-to-parent
        consistency only when the relevant parent concept is present.
        """

        names = tuple(str(name) for name in (concept_names or self.concept_names))
        validator = (
            self
            if names == self.concept_names
            else RobotConceptHierarchy.from_concept_names(
                names,
                concept_pos_value=self._concept_pos_value,
            )
        )
        matrix = np.asarray(C)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("C must be a 2D array or a 1D single-row array")
        if matrix.shape[1] != len(names):
            raise ValueError(
                "Number of matrix columns must match number of concept names"
            )

        name_to_idx = {name: idx for idx, name in enumerate(names)}
        active = matrix >= float(threshold)
        violations: list[HierarchyViolation] = []

        for feature, members in validator._group_members.items():
            idxs = [name_to_idx[name] for name in members if name in name_to_idx]
            if len(idxs) < 2:
                continue
            counts = active[:, idxs].sum(axis=1)
            bad_rows = np.where(counts > 1)[0]
            for row_idx in bad_rows.tolist():
                active_members = tuple(
                    member
                    for member, idx in zip(members, idxs, strict=False)
                    if bool(active[row_idx, idx])
                )
                violations.append(
                    HierarchyViolation(
                        row_index=int(row_idx),
                        relation="group",
                        concepts=active_members,
                        message=(
                            f"row {row_idx} activates multiple concepts in group "
                            f"{feature!r}: {active_members}"
                        ),
                    )
                )
                if max_violations is not None and len(violations) >= max_violations:
                    return violations

        for concept in validator._leaf_infos:
            if concept not in name_to_idx:
                continue
            source_idx = name_to_idx[concept]
            implications = validator.implies(concept)
            for implication in implications:
                target_idx = name_to_idx.get(implication.target)
                if target_idx is None:
                    continue
                source_active = active[:, source_idx]
                target_matches = active[:, target_idx] == bool(implication.value)
                bad_rows = np.where(source_active & ~target_matches)[0]
                for row_idx in bad_rows.tolist():
                    violations.append(
                        HierarchyViolation(
                            row_index=int(row_idx),
                            relation="implies",
                            concepts=(concept, implication.target),
                            message=(
                                f"row {row_idx} violates implication "
                                f"{concept!r} -> {implication.target!r}={implication.value}"
                            ),
                        )
                    )
                    if (
                        max_violations is not None
                        and len(violations) >= max_violations
                    ):
                        return violations

        return violations

    def _infer_leaf_infos(
        self,
        concept_names: Sequence[str] | None = None,
    ) -> dict[str, _LeafInfo]:
        names = tuple(str(name) for name in (concept_names or self.concept_names))
        name_set = frozenset(names)
        leaf_infos: dict[str, _LeafInfo] = {}

        for feature, values in ROBOT_CONCEPTS.items():
            for raw_value in values:
                value = str(raw_value)
                if "_" not in value:
                    continue
                concept_name = f"{feature}_{value}"
                if concept_name not in name_set:
                    continue
                coarse_value = value.split("_", 1)[0]
                leaf_infos[concept_name] = _LeafInfo(
                    concept=concept_name,
                    feature=feature,
                    feature_value=value,
                    coarse_value=coarse_value,
                )
        return leaf_infos

    def _build_group_members(
        self,
        concept_names: Sequence[str] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        leaf_infos = self._infer_leaf_infos(concept_names)
        grouped: dict[str, list[str]] = {}

        for feature, values in ROBOT_CONCEPTS.items():
            ordered = []
            for raw_value in values:
                concept_name = f"{feature}_{raw_value}"
                if concept_name in leaf_infos:
                    ordered.append(concept_name)
            if ordered:
                grouped[feature] = ordered

        return {feature: tuple(members) for feature, members in grouped.items()}
