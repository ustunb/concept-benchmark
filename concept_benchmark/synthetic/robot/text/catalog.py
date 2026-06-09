"""Robot text concept catalog and label computation."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from concept_benchmark.config import ROBOT_CONCEPTS
from concept_benchmark.formula import LabelFormula

CORE_CONCEPT_NAMES = [
    "head_is_square",
    "body_is_square",
    "has_knees",
    "has_elbows",
    "foot_is_pointy",
    "has_antennae",
    "ears_is_triangle",
    "mouth_is_open",
    "hands_are_pointy",
]


def enumerate_robot_concepts(
    concepts: dict | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Enumerate all combinations of robot concept values.

    Returns a DataFrame with one row per unique robot identity.
    """
    if concepts is None:
        concepts = ROBOT_CONCEPTS
    cols = list(concepts.keys())
    grids = [concepts[c] for c in cols]
    combos = list(product(*grids))
    df = pd.DataFrame(combos, columns=cols)
    if shuffle:
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df


def compute_label(
    df: pd.DataFrame,
    formula: LabelFormula,
    *,
    stochastic: bool = False,
    seed: int = 0,
) -> pd.Series:
    """Compute labels for a catalog DataFrame using a :class:`LabelFormula`.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with one row per robot identity.
    formula : LabelFormula
        The labeling rule (features, weights, intercept, temperature).
    stochastic : bool
        If *True*, sample labels via Bernoulli(σ(temperature × score)).
        If *False*, use the deterministic threshold (score ≥ 0 → Glorp).
    seed : int
        Random seed for stochastic sampling.
    """
    rng = np.random.default_rng(int(seed)) if stochastic else None

    def _label_one(sr):
        row = sr.to_dict()
        return formula.label(row, rng=rng)

    return df.apply(_label_one, axis=1).astype(str)
