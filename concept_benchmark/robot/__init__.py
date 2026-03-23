"""Robot classification benchmark.

Generate datasets of fictional robots (Glorps vs. Drents) from body features::

    from concept_benchmark.robot import DatasetGenerator

    dataset = DatasetGenerator(seed=1014).generate()
"""

from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.formula import LabelFormula
from concept_benchmark.generators import DatasetGenerator as _BaseGenerator


class DatasetGenerator(_BaseGenerator):
    """Robot benchmark dataset generator.

    All keyword arguments are forwarded to
    :class:`~concept_benchmark.config.RobotBenchmarkConfig`.

    Example::

        from concept_benchmark.robot import DatasetGenerator

        ds = DatasetGenerator(seed=1014, concept_preset="foot_subtypes").generate()
    """

    def __init__(self, **kwargs):
        super().__init__("robot", **kwargs)


__all__ = ["DatasetGenerator", "RobotBenchmarkConfig", "LabelFormula"]
