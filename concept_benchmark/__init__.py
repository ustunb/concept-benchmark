"""Synthetic benchmarks for evaluating Concept Bottleneck Models (CBMs).

This package provides dataset generators, data containers, and benchmark
configurations for two synthetic tasks:

* **Robot Classification** -- classify fictional robots from images or text
  using interpretable body-part concepts.
* **Sudoku Validation** -- determine whether a 9x9 Sudoku board is valid
  using row/column/block validity concepts.

Quick start::

    from concept_benchmark.robots import DatasetGenerator

    dataset = DatasetGenerator(seed=1014, render_images=False).generate()
    dataset.sample(test_size=0.2, val_size=0.2, seed=1014)
    print(dataset)
"""

from . import config, robots, sudoku, synthetic, transforms, utils
from .data import ConceptDataset, ConceptDatasetSample
from .formula import F, LabelFormula
from .generators import DatasetGenerator

utils.patch_macos_dataloader()

__all__ = [
    "config",
    "robots",
    "sudoku",
    "synthetic",
    "transforms",
    "utils",
    "ConceptDataset",
    "ConceptDatasetSample",
    "DatasetGenerator",
    "F",
    "LabelFormula",
]
