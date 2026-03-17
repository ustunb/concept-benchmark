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

from concept_benchmark.config import (
    ROBOT_CONCEPTS,
    RobotBenchmarkConfig,
    SudokuBenchmarkConfig,
)
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

    _VALID_FEATURES = frozenset(ROBOT_CONCEPTS.keys())

    def __init__(self, *, label_formula=None, **kwargs):
        if label_formula is not None:
            model_features = {}
            model_weights = {}
            intercept = 0.0
            for key, weight in label_formula.items():
                if key == "intercept":
                    intercept = weight
                else:
                    if not (isinstance(key, tuple) and len(key) == 2):
                        raise ValueError(
                            f"label_formula key must be 'intercept' or a "
                            f"(feature, value) tuple, got {key!r}. "
                            f"Valid features: {sorted(self._VALID_FEATURES)}"
                        )
                    feature, value = key
                    if feature not in self._VALID_FEATURES:
                        raise ValueError(
                            f"Unknown feature {feature!r} in label_formula. "
                            f"Valid features: {sorted(self._VALID_FEATURES)}"
                        )
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
        return generate_robot_dataset(self.config)


def generate_robot_dataset(config: RobotBenchmarkConfig):
    """Generate a robot dataset from a config, with train/val/test splits."""
    set_deterministic_seed(config.seed)
    settings = config.to_dict()
    data = create_synthetic_dataset(**settings)
    data.generate_cvindices(seed=config.seed)
    rng = np.random.default_rng(config.seed)
    return create_skewed_splits_full(dataset=data, rng=rng, **settings)


class SudokuDatasetGenerator:
    """Generate a Sudoku validation dataset with train/val/test splits.

    Wraps :class:`SudokuBenchmarkConfig` and data generation into a single
    ``generate()`` call.  By default ``data_type="image"`` renders boards
    as PNG images so that ``.explore()`` shows the actual images the model
    sees.  Use ``data_type="tabular"`` for a lightweight representation
    where ``X`` contains flattened digit vectors (much faster).

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
        data_type : str
            ``"image"`` (default) renders board PNGs; ``"tabular"`` stores
            flattened digit vectors (much faster).

    Examples
    --------
    >>> gen = SudokuDatasetGenerator(seed=171)
    >>> dataset = gen.generate()
    >>> dataset.training.C.shape
    (600, 27)
    """

    def __init__(self, **kwargs):
        """Create a generator with the given benchmark configuration.

        Parameters
        ----------
        **kwargs
            Passed directly to :class:`SudokuBenchmarkConfig`.
        """
        self.config = SudokuBenchmarkConfig(**kwargs)

    def generate(self):
        """Generate the dataset and return it with train/val/test splits.

        Returns
        -------
        ConceptDataset
            Dataset with ``.training``, ``.validation``, and ``.test`` set.
        """
        from functools import partial

        set_deterministic_seed(self.config.seed)
        cfg = self.config

        data_type = cfg.data_type
        kwargs = {}
        if data_type == "image":
            from concept_benchmark.synthetic.sudoku import image_transform

            kwargs["transform"] = partial(
                image_transform,
                cell_px=cfg.cell_px,
                margin_px=cfg.margin_px,
                line_px=cfg.line_px,
                bold_px=cfg.bold_px,
                font_size=cfg.font_size,
                handwriting=cfg.handwriting,
            )

        data = create_sudoku_dataset(
            n=cfg.n,
            n_samples=cfg.n_samples,
            valid_ratio=cfg.valid_ratio,
            max_corrupt=cfg.max_corrupt,
            seed=cfg.seed,
            data_type=data_type,
            **kwargs,
        )
        data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=cfg.seed)
        data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)
        return data
