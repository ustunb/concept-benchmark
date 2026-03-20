"""Synthetic benchmarks for evaluating Concept Bottleneck Models (CBMs).

This package provides dataset generators, data containers, and benchmark
configurations for two synthetic tasks:

* **Robot Classification** -- classify fictional robots from images or text
  using interpretable body-part concepts.
* **Sudoku Validation** -- determine whether a 9x9 Sudoku board is valid
  using row/column/block validity concepts.

Quick start::

    from concept_benchmark import DatasetGenerator

    from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS

    gen = DatasetGenerator("robot", seed=1014, render_images=False)
    dataset = gen.generate()
    dataset.drop_concepts(PRESET_EXCLUDED_CONCEPTS["ground_truth"])
    dataset.sample(test_size=0.2, val_size=0.2, seed=1014)
    print(dataset)
"""

from . import config, synthetic, utils
from .data import ConceptDataset, ConceptDatasetSample
from .generators import DatasetGenerator

utils.patch_macos_dataloader()

__all__ = [
    "config",
    "synthetic",
    "utils",
    "ConceptDataset",
    "ConceptDatasetSample",
    "DatasetGenerator",
]
