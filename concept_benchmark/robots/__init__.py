"""Robot classification benchmark.

Generate datasets of fictional robots (Glorps vs. Drents) from body features::

    from concept_benchmark.robots import DatasetGenerator

    dataset = DatasetGenerator(seed=1014).generate()
"""

from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.formula import F, LabelFormula
from concept_benchmark.generators import DatasetGenerator as _BaseGenerator
from concept_benchmark.robots.hierarchy import (
    HierarchyImplication,
    HierarchyViolation,
    RobotConceptHierarchy,
)


class DatasetGenerator(_BaseGenerator):
    """Robot benchmark dataset generator.

    All keyword arguments are forwarded to
    :class:`~concept_benchmark.config.RobotBenchmarkConfig`.

    Example::

        from concept_benchmark.robots import DatasetGenerator

        ds = DatasetGenerator(seed=1014, concept_preset="foot_subtypes").generate()
    """

    def __init__(self, **kwargs):
        super().__init__("robot", **kwargs)


__all__ = [
    "DatasetGenerator",
    "F",
    "LabelFormula",
    "RobotBenchmarkConfig",
    "HierarchyImplication",
    "HierarchyViolation",
    "RobotConceptHierarchy",
]
