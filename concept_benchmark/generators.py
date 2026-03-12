"""High-level dataset generators for concept benchmarks.

Provides a simple ``generate()`` API that wraps data creation, splitting,
and seed management into a single call:

    >>> from concept_benchmark import RobotDatasetGenerator
    >>> dataset = RobotDatasetGenerator(seed=1014, draw=False).generate()
    >>> dataset.training.C.shape
    (3800, 7)

    >>> from concept_benchmark import SudokuDatasetGenerator
    >>> dataset = SudokuDatasetGenerator(seed=171).generate()
    >>> dataset.training.C.shape
    (600, 27)
"""

from __future__ import annotations

import numpy as np

from concept_benchmark.config import RobotBenchmarkConfig, SudokuBenchmarkConfig
from concept_benchmark.synthetic.robot import create_synthetic_dataset
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset
from concept_benchmark.utils import create_skewed_splits_full, set_deterministic_seed

__all__ = ["RobotDatasetGenerator", "SudokuDatasetGenerator"]


class RobotDatasetGenerator:
    """Generate a robot classification dataset with train/val/test splits.

    Wraps :class:`RobotBenchmarkConfig`, data generation, CV indexing, and
    skewed splitting into a single ``generate()`` call.

    Parameters
    ----------
    label_formula : dict, optional
        Combined labeling function specification.  Maps ``(feature, value)``
        tuples to weights; use ``"intercept"`` key for the bias term::

            label_formula={
                ("mouth_type", "closed"): 5.0,
                ("foot_shape", "pointy"): 8.0,
                ("has_knees", "true"): -5.0,
                "intercept": 2.0,
            }

        This is equivalent to the score equation::

            score = 5·1[mouth=closed] + 8·1[foot=pointy] - 5·1[knees=true] + 2

        If not provided, defaults to the paper's labeling rule.
    **kwargs
        Forwarded to :class:`RobotBenchmarkConfig`.  Common options:

        seed : int
            Random seed (default ``1014``).
        subconcept : bool
            Use 12 fine-grained subconcepts instead of 7 ideal concepts.
        draw : bool
            Whether to render robot PNGs to disk (default ``True``).
            Set to ``False`` for faster generation when images aren't needed.
        model_type : str
            ``"stochastic"`` (default) or ``"deterministic"``.
        drop_concepts : list[str]
            Concepts to exclude from the concept matrix.

    Examples
    --------
    >>> gen = RobotDatasetGenerator(seed=1014, draw=False)
    >>> dataset = gen.generate()
    >>> dataset.training.C.shape
    (3800, 7)

    >>> gen = RobotDatasetGenerator(
    ...     label_formula={
    ...         ("mouth_type", "closed"): 5.0,
    ...         ("foot_shape", "pointy"): 8.0,
    ...         ("has_knees", "true"): -5.0,
    ...         "intercept": 2.0,
    ...     },
    ...     draw=False,
    ... )
    """

    def __init__(self, *, label_formula=None, **kwargs):
        if label_formula is not None:
            model_features = {}
            model_weights = {}
            intercept = 0.0
            for key, weight in label_formula.items():
                if key == "intercept":
                    intercept = weight
                else:
                    feature, value = key
                    model_features[feature] = value
                    model_weights[feature] = weight
            kwargs.setdefault("model_features", model_features)
            kwargs.setdefault("model_weights", model_weights)
            kwargs.setdefault("model_intercept", intercept)
        self.config = RobotBenchmarkConfig(**kwargs)

    def generate(self):
        """Generate the dataset and return it with train/val/test splits.

        Returns
        -------
        ConceptDataset
            Dataset with ``.training``, ``.validation``, and ``.test`` set.
        """
        set_deterministic_seed(self.config.seed)
        settings = self.config.to_dict()
        data = create_synthetic_dataset(**settings)
        data.generate_cvindices(seed=self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        data = create_skewed_splits_full(dataset=data, rng=rng, **settings)
        return data


class SudokuDatasetGenerator:
    """Generate a Sudoku validation dataset with train/val/test splits.

    Wraps :class:`SudokuBenchmarkConfig` and tabular data generation into a
    single ``generate()`` call.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`SudokuBenchmarkConfig`.  Common options:

        seed : int
            Random seed (default ``171``).
        n : int
            Block size (default ``3`` for 9x9 boards).
        n_samples : int
            Number of boards to generate (default ``1000``).
        max_corrupt : int
            Maximum number of cell swaps for invalid boards (default ``9``).
        valid_ratio : float
            Fraction of valid boards (default ``0.5``).

    Examples
    --------
    >>> gen = SudokuDatasetGenerator(seed=171)
    >>> dataset = gen.generate()
    >>> dataset.training.C.shape
    (600, 27)
    """

    def __init__(self, **kwargs):
        self.config = SudokuBenchmarkConfig(**kwargs)

    def generate(self):
        """Generate the dataset and return it with train/val/test splits.

        Returns
        -------
        ConceptDataset
            Dataset with ``.training``, ``.validation``, and ``.test`` set.
            Uses tabular representation (X = flattened board digits).
        """
        set_deterministic_seed(self.config.seed)
        data = create_sudoku_dataset(
            n=self.config.n,
            n_samples=self.config.n_samples,
            valid_ratio=self.config.valid_ratio,
            max_corrupt=self.config.max_corrupt,
            seed=self.config.seed,
            data_type="tabular",
        )
        data.generate_cvindices(
            strata=data.y, total_folds_for_cv=[5], seed=self.config.seed
        )
        data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)
        return data
