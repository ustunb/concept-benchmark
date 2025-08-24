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

    schemes = []
    for i, a in enumerate(pal):
        drop_idx = np.arange(i - 2, i + 3)  # similar colors in spectral are nearby
        keep_idx = np.setdiff1d(np.arange(len(pal)), drop_idx).tolist()
        pool = StackPalette(pal[keep_idx]) & qal
        pairs = [(a, b) for b in pool if 250 < colordist(a, b)]
        schemes += pairs

    # add flipped pairs
    if include_flipped:
        schemes += [(b, a) for a, b in schemes]

    # convert to Pero colors
    schemes = [(Color(a.hex()), Color(b.hex())) for a, b in schemes]

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(schemes)

    return schemes


def unlist0(obj):
    if type(obj) in [list, tuple]:
        return obj[0]
    else:
        return obj


def model_to_logistic(model: str):
    """
    Extracts the arithmetic expression inside the first (...) before a comparison
    and returns it wrapped in the logistic function.
    """
    # find whatever is inside (...) before a comparison operator
    match = re.search(r"\((.+?)\s*>=", model)
    if not match:
        raise ValueError("Model string not in expected format.")

    expr = match.group(1).strip()
    # print(f"Returning: expit({expr})")
    return f"expit({expr})"
