"""Robot text data generation utilities."""

from concept_benchmark.synthetic.robot_text.catalog import (
    CORE_CONCEPT_NAMES,
    enumerate_robot_concepts,
    compute_label,
)
from concept_benchmark.synthetic.robot_text.corpus import (
    load_jsonl,
    render_from_corpus,
    core_vector_from_row,
    compute_text_concept_names,
    concept_vector_from_row,
)
from concept_benchmark.synthetic.robot_text.dataset import (
    build_text_dataset,
    kfold_by_robot_identity,
)

__all__ = [
    "CORE_CONCEPT_NAMES",
    "enumerate_robot_concepts",
    "compute_label",
    "load_jsonl",
    "render_from_corpus",
    "core_vector_from_row",
    "compute_text_concept_names",
    "concept_vector_from_row",
    "build_text_dataset",
    "kfold_by_robot_identity",
]
