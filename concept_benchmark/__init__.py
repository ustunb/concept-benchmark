"""Synthetic benchmarks for evaluating Concept Bottleneck Models (CBMs).

This package provides dataset generators, data containers, and benchmark
configurations for two synthetic tasks:

* **Robot Classification** -- classify fictional robots from images or text
  using interpretable body-part concepts.
* **Sudoku Validation** -- determine whether a 9x9 Sudoku board is valid
  using row/column/block validity concepts.

Quick start::

    from concept_benchmark import RobotDatasetGenerator

    gen = RobotDatasetGenerator(seed=1014, subconcept=True, draw=True)
    dataset = gen.generate()
    print(dataset)
"""

from . import config, synthetic, utils
from .data import ConceptDataset, ConceptDatasetSample
from .generators import RobotDatasetGenerator, SudokuDatasetGenerator

__all__ = [
    "config",
    "synthetic",
    "utils",
    "ConceptDataset",
    "ConceptDatasetSample",
    "RobotDatasetGenerator",
    "SudokuDatasetGenerator",
]
