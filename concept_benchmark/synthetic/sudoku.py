"""
Generate Sudoku ConceptDataset
"""
from typing import Callable, Optional
import numpy as np
import random
from concept_benchmark.data import ConceptDataset
from concept_benchmark.synthetic.sudoku_helper import (
    generate_valid_board,
    generate_invalid_board,
    get_concepts,
)

# TODO: label noise, concept noise, concept masking toggles
def create_sudoku_dataset(
    n_samples: int = 1000,
    valid_ratio: float = 0.5,
    max_corrupt: int = 3,
    seed: int = 42,
    transform: Optional[Callable] = None,
) -> ConceptDataset:
    """
    Create a synthetic dataset of Sudoku boards with concepts.

    Args:
        n_samples (int): Number of samples to generate.
        valid_ratio (float): Ratio of valid to invalid boards.
        max_corrupt (int): Maximum number of changes to make an invalid board.
        seed (int): Random seed for reproducibility.
        transform (callable, optional): Optional transformation to apply to the boards.
                                        Should take a board (9 x 9 numpy array) and return
                                        a transformed representation as a np.ndarray.

    Returns:
        ConceptDataset
    """
    random.seed(seed)
    np.random.seed(seed)

    transform = transform or default_transform

    n_valid = int(round(n_samples * float(valid_ratio)))
    n_invalid = n_samples - n_valid

    X_list = []   # features
    C_list = []   # concept vectors (27,)
    y_list = []   # labels: board_valid (0/1)

    # Generate valid boards
    for _ in range(n_valid):
        b = generate_valid_board()
        X_list.append(transform(b))
        C_list.append(np.ones(27, dtype=np.int32))  # all concepts valid
        y_list.append(1)

    # Generate invalid boards by corrupting valid ones
    for _ in range(n_invalid):
        # one change = turns off at most 3 concepts
        num_changes = random.randint(1, max_corrupt // 3)
        b = generate_invalid_board(num_changes=num_changes)
        concepts = get_concepts(b, return_label=False)
        c_arr = np.array(list(concepts.values()), dtype=np.int32).flatten()

        X_list.append(transform(b))
        C_list.append(c_arr)
        y_list.append(0)

    X = np.stack(X_list, axis=0)
    C = np.stack(C_list, axis=0)
    y = np.array(y_list, dtype=np.int32)

    concept_names = (
        [f"row_valid_{i+1}" for i in range(9)]
        + [f"col_valid_{i+1}" for i in range(9)]
        + [f"block_valid_{i+1}" for i in range(9)]
    )

    # TODO: decide whether to store original boards in metadata
    meta = {
        "classes": [0, 1], # 0 for invalid, 1 for valid
        "concepts": concept_names,
        "data_type": "tabular", # TODO: consider changing to "image" if using image transforms
        "transform": transform.__name__ if transform else "default",
        "max_corrupt": max_corrupt,
        "seed": seed,
    }

    return ConceptDataset(X=X, C=C, y=y, meta=meta)
    

# default transform flattens the 9x9 board to 81-dim vector
def default_transform(board: np.ndarray) -> np.ndarray:
    return board.astype(np.float32).reshape(-1)