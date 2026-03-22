"""Robot text concept catalog and label computation."""

from __future__ import annotations

import re
from itertools import product

import numpy as np
import pandas as pd

from concept_benchmark.config import ROBOT_CONCEPTS

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
    model_expr: str,
    label_model_type: str = "deterministic",
    alpha: float = 1.0,
    bias: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Compute labels for a catalog DataFrame using a model expression.

    For deterministic: evaluates the expression directly.
    For stochastic: extracts a numeric score, applies sigmoid, samples.
    """
    SAFE_GLOBALS = {
        "__builtins__": None,
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
        "any": any,
        "all": all,
        "np": np,
        "min": min,
        "max": max,
    }
    rng = np.random.default_rng(int(seed))

    def _cond_to_score(expr: str) -> str | None:
        m = re.search(r"\bif\s+(?P<cond>.+?)\s+else\b", expr)
        cond = m.group("cond").strip() if m else expr.strip()
        m2 = re.search(
            r"^(?P<lhs>.+?)(?:\s*(?:>=|<=|>|<)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$",
            cond,
            flags=re.IGNORECASE,
        )
        lhs = m2.group("lhs").strip() if m2 else cond
        while lhs.startswith("(") and lhs.endswith(")"):
            lvl = 0
            ok = True
            for ch in lhs:
                if ch == "(":
                    lvl += 1
                elif ch == ")":
                    lvl -= 1
                    if lvl < 0:
                        ok = False
                        break
            if ok and lvl == 0:
                lhs = lhs[1:-1].strip()
            else:
                break
        return lhs or None

    score_expr = (
        _cond_to_score(model_expr) if label_model_type == "stochastic" else None
    )

    def eval_one(sr):
        row = sr.to_dict()
        if label_model_type is None or label_model_type == "deterministic":
            return eval(model_expr, SAFE_GLOBALS, {"row": row})
        score = None
        if score_expr:
            try:
                score = float(eval(score_expr, SAFE_GLOBALS, {"row": row}))
            except (TypeError, ValueError, NameError, SyntaxError):
                score = None
        if score is None:
            try:
                hard = eval(model_expr, SAFE_GLOBALS, {"row": row})
                score = 1.0 if str(hard).strip().lower() == "glorp" else 0.0
            except (TypeError, ValueError, NameError, SyntaxError):
                score = 0.0
        p = 1.0 / (1.0 + float(np.exp(-float(alpha) * (float(score) - float(bias)))))
        return "glorp" if rng.random() < p else "drent"

    return df.apply(eval_one, axis=1).astype(str)
