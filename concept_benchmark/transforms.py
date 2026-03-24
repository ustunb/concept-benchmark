"""Dataset transform generators.

Each generator takes a :class:`~concept_benchmark.data.ConceptDataset` and
configuration, then ``.generate()`` returns a **new** dataset with the
transformation applied.  The original dataset is never modified.

Example::

    from concept_benchmark.robots import DatasetGenerator
    from concept_benchmark.transforms import (
        ConceptDropGenerator,
        ConceptNoiseGenerator,
        ConceptMissingnessGenerator,
        LabelNoiseGenerator,
    )

    dataset = DatasetGenerator(seed=1014).generate()
    dataset.sample(test_size=0.2, val_size=0.2, seed=1014)

    dataset = ConceptDropGenerator(dataset, ["has_elbows"]).generate()
    noisy = ConceptNoiseGenerator(dataset, p=0.1, seed=42).generate()
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from concept_benchmark.data import ConceptDataset


class ConceptDropGenerator:
    """Remove concepts from a dataset.

    Parameters
    ----------
    dataset : ConceptDataset
        Input dataset.
    concepts_to_drop : list of str
        Concept names to remove.

    Example::

        smaller = ConceptDropGenerator(dataset, ["has_elbows", "hand_shape"]).generate()
    """

    def __init__(self, dataset: ConceptDataset, concepts_to_drop: list[str]) -> None:
        self.dataset = dataset
        self.concepts_to_drop = list(concepts_to_drop)

    def __repr__(self) -> str:
        remaining = self.dataset.n_concepts - len(self.concepts_to_drop)
        lines = [
            f"ConceptDropGenerator(dropping={self.concepts_to_drop})",
            f"  input: {self.dataset.n} samples, {self.dataset.n_concepts} concepts \u2192 {remaining} concepts after drop",
        ]
        return "\n".join(lines)

    def generate(self) -> ConceptDataset:
        """Return a new dataset with the specified concepts removed."""
        ds = copy.deepcopy(self.dataset)
        ds.drop_concepts(self.concepts_to_drop)
        return ds


class ConceptNoiseGenerator:
    """Add bit-flip noise to concept labels.

    Parameters
    ----------
    dataset : ConceptDataset
        Input dataset.
    p : float
        Probability of flipping each concept bit (default 0.1).
    seed : int, optional
        Random seed for reproducibility.
    config : dict, optional
        Asymmetric noise config with keys ``"p01"`` and ``"p10"``.
    splits : set of str, optional
        Apply noise only to these splits (e.g. ``{"train"}``).
        If ``None``, applies to all splits.

    Example::

        noisy = ConceptNoiseGenerator(dataset, p=0.1, seed=42).generate()
        noisy_train = ConceptNoiseGenerator(dataset, p=0.2, seed=42, splits={"train"}).generate()
    """

    def __init__(
        self,
        dataset: ConceptDataset,
        *,
        p: float = 0.1,
        seed: int | None = None,
        config: Mapping[str, object] | None = None,
        splits: set[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.p = p
        self.seed = seed
        self.config = config
        self.splits = splits

    def __repr__(self) -> str:
        ds = self.dataset
        params = [f"p={self.p}"]
        if self.seed is not None:
            params.append(f"seed={self.seed}")
        if self.splits:
            params.append(f"splits={self.splits}")
        lines = [
            f"ConceptNoiseGenerator({', '.join(params)})",
            f"  input: {ds.n} samples, {ds.n_concepts} concepts",
        ]
        if ds.train.n > 0 and ds.test.n > 0:
            lines.append(
                f"  splits: train={ds.train.n}, val={ds.validation.n}, test={ds.test.n}"
            )
        return "\n".join(lines)

    def generate(self) -> ConceptDataset:
        """Return a new dataset with concept noise baked in."""
        ds = copy.deepcopy(self.dataset)
        ds.sample_concept_noise(
            p=self.p,
            rng=self.seed,
            config=self.config,
            enable=True,
        )
        return ds


class ConceptMissingnessGenerator:
    """Add missing concept labels.

    Parameters
    ----------
    dataset : ConceptDataset
        Input dataset.
    p : float
        Fraction of concept values to mark as missing (default 0.1).
    mechanism : str
        ``"mcar"`` (missing completely at random) or ``"mnar"``
        (missing not at random).
    seed : int, optional
        Random seed.
    mnar_config : dict, optional
        Config for MNAR mechanism (keys ``"present_prob"``, ``"absent_prob"``).
    fill_value : float
        Value used to replace missing concepts (default ``NaN``).
    splits : set of str, optional
        Apply only to these splits. If ``None``, applies to all.

    Example::

        missing = ConceptMissingnessGenerator(dataset, p=0.2, mechanism="mcar", seed=99).generate()
    """

    def __init__(
        self,
        dataset: ConceptDataset,
        *,
        p: float = 0.1,
        mechanism: str = "mcar",
        seed: int | None = None,
        mnar_config: Mapping[str, object] | None = None,
        fill_value: float = np.nan,
        splits: set[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.p = p
        self.mechanism = mechanism
        self.seed = seed
        self.mnar_config = mnar_config
        self.fill_value = fill_value
        self.splits = splits

    def __repr__(self) -> str:
        ds = self.dataset
        params = [f"p={self.p}", f"mechanism={self.mechanism!r}"]
        if self.seed is not None:
            params.append(f"seed={self.seed}")
        if self.splits:
            params.append(f"splits={self.splits}")
        lines = [
            f"ConceptMissingnessGenerator({', '.join(params)})",
            f"  input: {ds.n} samples, {ds.n_concepts} concepts",
        ]
        if ds.train.n > 0 and ds.test.n > 0:
            lines.append(
                f"  splits: train={ds.train.n}, val={ds.validation.n}, test={ds.test.n}"
            )
        return "\n".join(lines)

    def generate(self) -> ConceptDataset:
        """Return a new dataset with concept missingness baked in."""
        ds = copy.deepcopy(self.dataset)
        ds.sample_concept_missingness(
            p=self.p,
            mechanism=self.mechanism,
            rng=self.seed,
            mnar_config=self.mnar_config,
            fill_value=self.fill_value,
            enable=True,
            splits=self.splits,
        )
        return ds


class LabelNoiseGenerator:
    """Add noise to labels.

    Parameters
    ----------
    dataset : ConceptDataset
        Input dataset.
    p : float
        Probability of flipping each label (default 0.1).
    seed : int, optional
        Random seed.
    config : dict, optional
        Label noise config with optional ``"flip_matrix"`` key.

    Example::

        label_noisy = LabelNoiseGenerator(dataset, p=0.05, seed=7).generate()
    """

    def __init__(
        self,
        dataset: ConceptDataset,
        *,
        p: float = 0.1,
        seed: int | None = None,
        config: Mapping[str, object] | None = None,
    ) -> None:
        self.dataset = dataset
        self.p = p
        self.seed = seed
        self.config = config

    def __repr__(self) -> str:
        ds = self.dataset
        params = [f"p={self.p}"]
        if self.seed is not None:
            params.append(f"seed={self.seed}")
        lines = [
            f"LabelNoiseGenerator({', '.join(params)})",
            f"  input: {ds.n} samples, {ds.n_classes} classes",
        ]
        if ds.train.n > 0 and ds.test.n > 0:
            lines.append(
                f"  splits: train={ds.train.n}, val={ds.validation.n}, test={ds.test.n}"
            )
        return "\n".join(lines)

    def generate(self) -> ConceptDataset:
        """Return a new dataset with label noise baked in."""
        ds = copy.deepcopy(self.dataset)
        ds.sample_label_noise(
            p=self.p,
            rng=self.seed,
            label_noise_config=self.config,
            enable=True,
        )
        return ds


__all__ = [
    "ConceptDropGenerator",
    "ConceptNoiseGenerator",
    "ConceptMissingnessGenerator",
    "LabelNoiseGenerator",
]
