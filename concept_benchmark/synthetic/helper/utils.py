"""
This file contains helper functions that we use throughout the project
"""

import re
import numpy as np
import colorir
from colorir import StackPalette
from colorir import simplified_dist as colordist


# Color Schemes
def generate_color_schemes(shuffle=True, random_seed=123456, include_flipped=True):
    """
    generate color schemes used to color distinct Robots
    each color scheme is a tuple that contains the fill color for the (left, right) halves of a Robot
    :param shuffle: set to True to shuffle the schemes
    :param random_seed: random seed used to shuffle
    :param include_flipped: set to True to include (right, left) color scheme in output
    :return:
    """

    # use spectral since we can easily drop similar colors
    pal = StackPalette.load("spectral", palettes_dir=colorir.config.DEFAULT_PALETTES_DIR)

    # add in "dark" colors from paired for extra colors
    qal = StackPalette.load("paired", palettes_dir=colorir.config.DEFAULT_PALETTES_DIR)
    qal = StackPalette([qal[i] for i in range(1, len(qal), 2)])

    # we want the color schemes for the robot halves to be distinguishable
    schemas = [(pal[i], pal[-1 - i]) for i in range(len(pal)//2)]
    schemas += [(s, d) for s in pal for d in qal if colordist(s, d) > 0.175]

    if include_flipped:
        schemas_flipped = [(b, a) for (a, b) in schemas]
        schemas = schemas + schemas_flipped

    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(schemas)

    return schemas


def subset_colors(colors, N=5):
    return [c for i, c in enumerate(colors) if i % N == 0]


def get_color_generator(colors, N=5, n=10**7):
    rng = np.random.default_rng(123456)
    rng.random((n, 10))
    colors = subset_colors(colors, N)
    for i in range(n):
        yield colors[i % len(colors)]


def unlist0(obj):
    if type(obj) in [list, tuple]:
        return obj[0]
    else:
        return obj


def build_model_expression(
    model_features: dict,
    *,
    model_type: str = "stochastic",
    weights: dict | None = None,
    intercept: float = 0.0,
    scalar: float = 1.0,
) -> str:
    """Build a row-level label expression from structured feature spec.

    Both modes compute:  score = Σ w_i · 1[f_i = v_i] + intercept

    Deterministic: ``'glorp' if score >= 0 else 'drent'``
    Stochastic:    ``expit(scalar * score)``  (probability for Bernoulli sampling)
    """
    weights = weights or {}
    terms = []
    for feat, val in model_features.items():
        w = weights.get(feat, 1.0)
        terms.append(f"{w}*int(row['{feat}']=='{val}')")
    score_expr = " + ".join(terms) + f" + {intercept}"

    if model_type == "deterministic":
        return f"'glorp' if ({score_expr}) >= 0 else 'drent'"
    elif model_type == "stochastic":
        if scalar != 1.0:
            score_expr = f"({score_expr}) * {scalar}"
        return f"expit({score_expr})"
    else:
        raise ValueError(f"Invalid model_type: {model_type!r}")


def model_to_logistic(model: str, weights: dict, scalar = 1.0, intercept = None) -> str:
    """
    Extracts the arithmetic expression inside the first (...) before a comparison operator and the number after the
    comparison operator >= then constructs a new expression that subtracts the number from the arithmetic expression
    and returns it wrapped in the logistic function.
    """
    # find whatever is inside (...) before a comparison operator
    match = re.search(r"\((.+?)\s*>=", model)
    if not match:
        raise ValueError("Model string not in expected format.")

    num = re.search(r">=\s*([-\d.]+)", model)
    expr = match.group(1).strip()[:-1]

    # extract feature names, that are stored inside row parentheses row["feature"]
    matches = re.findall(r"row\['(.*?)'\]==(['\"].*?['\"])", expr)
    for feature, value in matches:
        if feature not in weights:
            import logging
            logging.getLogger(__name__).warning("Weight for feature '%s' not found in weights dictionary.", feature)
            weight = 1
        else:
            weight = weights[feature]
        old_pattern = f"int(row['{feature}']=={value})"
        new_pattern = f"{weight}*int(row['{feature}']=={value})"
        expr = expr.replace(old_pattern, new_pattern)

    intercept = intercept if intercept is not None else num.group(1)
    expr += ' - ' + str(intercept)

    if scalar != 1.0:
        expr = f"({expr}) * {scalar}"
    return f"expit({expr})"


def apply_subjective_noise(C, rate=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(C).astype(np.int32, copy=True)
    if rate <= 0:
        return X
    M = rng.random(X.shape) < float(rate)
    X = (X ^ M.astype(np.int32))
    X[X < 0] = 0
    X[X > 1] = 1
    return X


def apply_machine_noise(C, confusion=None, seed=0):
    if isinstance(confusion, dict):
        p01 = float(confusion.get("p01", 0.2))
        p10 = float(confusion.get("p10", 0.05))
    elif isinstance(confusion, (list, tuple)) and len(confusion) == 2:
        p01, p10 = float(confusion[0]), float(confusion[1])
    else:
        p01, p10 = 0.2, 0.05
    rng = np.random.default_rng(seed)
    X = np.asarray(C).astype(np.int32, copy=True)
    R = rng.random(X.shape)
    m01 = (X == 0) & (R < p01)
    m10 = (X == 1) & (R < p10)
    X[m01] = 1
    X[m10] = 0
    return X
