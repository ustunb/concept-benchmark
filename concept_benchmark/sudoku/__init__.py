"""Sudoku validation benchmark.

Generate datasets of 9x9 Sudoku boards with row/column/block validity concepts::

    from concept_benchmark.sudoku import DatasetGenerator

    dataset = DatasetGenerator(seed=171, n_boards=1000).generate()
"""

from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.generators import DatasetGenerator as _BaseGenerator


class DatasetGenerator(_BaseGenerator):
    """Sudoku benchmark dataset generator.

    All keyword arguments are forwarded to
    :class:`~concept_benchmark.config.SudokuBenchmarkConfig`.

    Example::

        from concept_benchmark.sudoku import DatasetGenerator

        ds = DatasetGenerator(seed=171, n_boards=1000).generate()
    """

    def __init__(self, **kwargs):
        super().__init__("sudoku", **kwargs)


__all__ = ["DatasetGenerator", "SudokuBenchmarkConfig"]
