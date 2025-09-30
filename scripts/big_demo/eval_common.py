"""Shared helpers for big demo evaluation scripts.

This module provides utilities for loading datasets/models and exporting
metric records so that specialised evaluation scripts (e.g. conceptual
safeguards, score interventions) can stay focused on their strategy-specific
logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from concept_benchmark.ext.fileutils import load
from concept_benchmark.intervention import ConceptInterventionRunner
from concept_benchmark.models import ConceptBasedModel, ConceptDetector, FrontEndModel

import utils as big_demo_utils

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    import torch


# Lightweight defaults that cover the parameters required to locate datasets
# and models for the sudoku and robot tasks.
BASE_DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sudoku": {
        "data_name": "sudoku",
        "data_type": "tabular",
        "n": 3,
        "max_corrupt": 21,
    },
    "robot": {
        "data_name": "robot",
        "data_type": "image",
        "n": 1,
    },
}


# INTERVENTION_SPLITS = {"validation", "test"}
INTERVENTION_SPLITS = {"test"}


@dataclass
class MetricRecord:
    """Row-oriented metric container ready for CSV export."""

    strategy: str
    metric: str
    value: float
    split: str
    data_name: str
    data_type: str
    concept_noise: float
    concept_missing: float
    concept_missing_mech: str
    target_accuracy_label: str
    target_accuracy_value: float
    params: Dict[str, Any] = field(default_factory=dict)


def iter_splits(dataset) -> Iterator[Tuple[str, Any]]:
    """Yield available dataset splits while preserving train/val/test ordering."""

    yield "train", dataset.training
    validation = getattr(dataset, "validation", None)
    if validation is not None:
        yield "validation", validation
    test = getattr(dataset, "test", None)
    if test is not None:
        yield "test", test


def load_dataset(settings: Mapping[str, Any], *, fold_id: str, fold_val: int, fold_test: int):
    """Load a dataset specified by ``settings`` and materialise default splits."""

    dataset_path = big_demo_utils.get_dataset_file(**settings)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    dataset = load(dataset_path)
    dataset.split(fold_id=fold_id, fold_num_validation=fold_val, fold_num_test=fold_test)
    return dataset


def apply_missingness(dataset, mechanism: str, missing_rate: float, *, seed: int = 42) -> None:
    """Apply concept missingness in-place according to the provided mechanism."""

    if mechanism == "none":
        dataset.sample_concept_missingness(enable=False)
        return

    dataset.sample_concept_missingness(
        p=missing_rate,
        mechanism=mechanism,
        rng=seed,
        enable=True,
    )


def _load_concept_detector(path: Path, device: "torch.device") -> ConceptDetector:
    detector = load(path)
    if not isinstance(detector, ConceptDetector):
        raise TypeError(f"Expected ConceptDetector at {path}, got {type(detector)!r}")
    detector.to(device)
    return detector


def _load_front_end(path: Path) -> FrontEndModel:
    model = load(path)
    if not isinstance(model, FrontEndModel):
        raise TypeError(f"Expected FrontEndModel at {path}, got {type(model)!r}")
    return model


def build_cbm(settings: Mapping[str, Any], device: "torch.device", *, propagate: bool) -> Optional[ConceptBasedModel]:
    """Load a concept bottleneck model (concept detector + front-end)."""

    cd_path = big_demo_utils.get_model_file(model_type="cd", **settings)
    fe_path = big_demo_utils.get_model_file(model_type="fe", **settings)

    if not cd_path.is_file() or not fe_path.is_file():
        return None

    concept_detector = _load_concept_detector(cd_path, device)
    front_end_model = _load_front_end(fe_path)

    return ConceptBasedModel(
        concept_detector=concept_detector,
        front_end_model=front_end_model,
        propagate=propagate,
    )


def default_target_options(labels: Optional[Sequence[str]] = None) -> List[Tuple[str, float]]:
    """Resolve target accuracy labels into (label, value) pairs."""

    items = list(big_demo_utils.DIFFICULTY.items())
    if labels is None:
        return items

    allowed = {label: value for label, value in items}
    resolved: List[Tuple[str, float]] = []
    for label in labels:
        if label not in allowed:
            raise ValueError(f"Unknown target accuracy label: {label}")
        resolved.append((label, allowed[label]))
    return resolved


def default_concept_noise(values: Optional[Sequence[float]] = None) -> List[float]:
    """Return concept noise sweep values, optionally filtered by user input."""

    defaults = [float(v) for v in big_demo_utils.CONCEPT_NOISE]
    if values is None:
        return defaults
    return [float(v) for v in values]


def default_missingness_levels(values: Optional[Sequence[float]] = None) -> List[float]:
    """Return concept missingness sweep values (excluding zero)."""

    defaults = [float(v) for v in big_demo_utils.CONCEPT_MISSING]
    if values is None:
        return defaults
    return [float(v) for v in values]


def build_settings(
    *,
    data_name: str,
    concept_noise: float,
    target_accuracy: float,
    concept_missing: float,
    concept_missing_mech: str,
) -> Dict[str, Any]:
    """Compose dataset/model settings for a specific configuration."""

    if data_name not in BASE_DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {data_name}")

    base = dict(BASE_DATASET_CONFIGS[data_name])
    base.update(
        {
            "concept_noise": float(concept_noise),
            "target_accuracy": float(target_accuracy),
            "concept_missing": float(concept_missing),
            "concept_missing_mech": concept_missing_mech,
        }
    )
    return base


def write_metrics_csv(records: Sequence[MetricRecord], output_path: Path) -> None:
    """Write metric records to CSV using a melted representation."""

    if not records:
        print("No metrics to write; skipping CSV export.")
        return

    extra_keys = sorted({key for record in records for key in record.params.keys()})

    header = [
        "strategy",
        "metric",
        "value",
        "split",
        "data_name",
        "data_type",
        "concept_noise",
        "concept_missing",
        "concept_missing_mech",
        "target_accuracy_label",
        "target_accuracy_value",
    ] + extra_keys

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        # Defer import to avoid import cycles when scripts only need helpers.
        import csv  # noqa: WPS433 - local import keeps csv optional for callers

        writer = csv.writer(handle)
        writer.writerow(header)
        for record in records:
            row = [
                record.strategy,
                record.metric,
                record.value,
                record.split,
                record.data_name,
                record.data_type,
                f"{record.concept_noise:.6f}",
                f"{record.concept_missing:.6f}",
                record.concept_missing_mech,
                record.target_accuracy_label,
                f"{record.target_accuracy_value:.6f}",
            ]
            for key in extra_keys:
                value = record.params.get(key, "")
                if isinstance(value, float):
                    row.append(f"{value:.6f}")
                else:
                    row.append(value)
            writer.writerow(row)


__all__ = [
    "BASE_DATASET_CONFIGS",
    "MetricRecord",
    "apply_missingness",
    "build_cbm",
    "build_settings",
    "default_concept_noise",
    "default_missingness_levels",
    "default_target_options",
    "iter_splits",
    "load_dataset",
    "write_metrics_csv",
    "ConceptInterventionRunner",
    "INTERVENTION_SPLITS",
]
