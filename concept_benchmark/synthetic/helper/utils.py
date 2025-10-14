"""
This file contains helper functions that we use throughout the project
"""

import re
import numpy as np
from colorir import StackPalette
from colorir import simplified_dist as colordist
from pero import Color


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
    pal = StackPalette.load("spectral")

    # add in "dark" colors from paired for extra colors
    qal = StackPalette.load("paired")
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


def model_to_logistic(model: str, scalar = 1.0, intercept = None) -> str:
    """
    Extracts the arithmetic expression inside the first (...) before a comparison operator and the number after the
    comparison operator >= then constructs a new expression that subtracts the number from the arithmetic expression
    and returns it wrapped in the logistic function.
    """
    # find whatever is inside (...) before a comparison operator
    match = re.search(r"\((.+?)\s*>=", model)
    if not match:
        raise ValueError("Model string not in expected format.")

    #extract the number after >=
    num = re.search(r">=\s*([-\d.]+)", model)

    expr = match.group(1).strip()[:-1]
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
