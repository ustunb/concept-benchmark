"""
Generate Sudoku ConceptDataset
"""

import random
from collections.abc import Callable

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from concept_benchmark.data import ConceptDataset
from concept_benchmark.paths import data_dir
from concept_benchmark.synthetic.sudoku_helper import (
    generate_invalid_board,
    generate_valid_board,
    get_concepts,
)

SUDOKU_DIR = data_dir / "sudoku"


# TODO: label noise, concept noise, concept masking toggles
def create_sudoku_dataset(
    *,
    n_samples: int = 1000,
    valid_ratio: float = 0.5,
    max_corrupt: int = 3,
    data_type: str = "tabular",
    seed: int = 42,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ds_name: str | None = None,
) -> ConceptDataset:
    """Create a synthetic dataset of Sudoku boards with concepts.

    Args:
        n_samples (int, optional): Number of samples to generate.
            Defaults to 1000.
        valid_ratio (float, optional): Ratio of valid to invalid boards.
            Defaults to 0.5.
        max_corrupt (int, optional): Maximum number of changes to
            make an invalid board. Defaults to 3.
        data_type (str, optional): Type of data representation 
            (e.g., "tabular" or "image"). Defaults to "tabular".
        seed (int, optional): Random seed for reproducibility. 
            Defaults to 42.
        transform (Callable[[np.ndarray], np.ndarray], optional): 
            Should take a board (9 x 9 numpy array) and 
            return a transformed representation as a np.ndarray. 
            Default is None, which uses a simple flattening transform.
        ds_name (str, optional): name of the dataset, used as folder name
            for saving images.

    Returns:
        ConceptDataset
    """
    # Ensure ds_name is set for image datasets
    if data_type == "image":
        ds_name = ds_name if ds_name else \
            datetime.now().strftime("%Y%m%d_%H%M%S")
        ds_path = SUDOKU_DIR / ds_name
        ds_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    transform = transform or default_transform

    n_valid = int(round(n_samples * float(valid_ratio)))
    n_invalid = n_samples - n_valid

    X_list = []  # features
    C_list = []  # concept vectors (27,)
    y_list = []  # labels: board_valid (0/1)

    # Generate valid boards
    for _ in range(n_valid):
        b = generate_valid_board()

        if data_type == "image":
            img_path = ds_path / f"valid_{_}.png"
            transform(b, outfile=img_path)
            X_list.append(img_path)
        else:
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

        if data_type == "image":
            img_path = ds_path / f"invalid_{_}.png"
            transform(b, outfile=img_path)
            X_list.append(img_path)
        else:
            X_list.append(transform(b))

        C_list.append(c_arr)
        y_list.append(0)

    X = np.stack(X_list, axis=0)
    C = np.stack(C_list, axis=0, dtype=np.int32)
    y = np.array(y_list, dtype=np.int32)

    if data_type == "image":
        # save concepts and labels to same folder as csv
        np.savetxt(ds_path / "concepts.csv", C, delimiter=",")
        np.savetxt(ds_path / "labels.csv", y, delimiter=",")

    concept_names = (
        [f"row_valid_{i + 1}" for i in range(9)]
        + [f"col_valid_{i + 1}" for i in range(9)]
        + [f"block_valid_{i + 1}" for i in range(9)]
    )

    # TODO: decide whether to store original boards in metadata
    meta = {
        "classes": [0, 1],  # 0 for invalid, 1 for valid
        "concepts": concept_names,
        "data_type": data_type,
        "transform": transform.__name__ if transform else "default",
        "max_corrupt": max_corrupt,
        "seed": seed,
    }
    
    if data_type == "image":
        kwargs = {"preprocess": sudoku_image_preprocess}
    else:
        kwargs = {}

    return ConceptDataset(X=X, C=C, y=y, meta=meta, **kwargs)


def default_transform(board: np.ndarray) -> np.ndarray:
    """Flattens the 9x9 board to 81-dim vector.

    Args:
        board (np.ndarray): 9x9 Sudoku board with values in {0, 1..9}.

    Returns:
        np.ndarray: Flattened vector.
        Output dimensions = (81,).
    """
    return board.astype(np.float32).reshape(-1)


def onehot_transform(board: np.ndarray) -> np.ndarray:
    """Convert a 9x9 board to a one-hot encoded representation.
    
    Args:
        board (np.ndarray): 9x9 Sudoku board with values in {0, 1..9}.
        
    Returns:
        np.ndarray: One-hot encoded representation of the board.
        Output dimensions = (9, 9, 9).
    """
    x = board.astype(np.int64) - 1  # zero indexing
    return np.eye(9, dtype=np.float32)[x]  # (9,9,9)


def histogram_transform(board: np.ndarray) -> np.ndarray:
    """Convert a 9x9 board to a histogram representation.
    
    Each row, column, and block is represented as a histogram of digit counts.
    
    Args:
        board (np.ndarray): 9x9 Sudoku board with values in {0, 1..9}.
        
    Returns:
        np.ndarray: Histogram representation of the board.
        Output dimensions = (27, 9).
    """
    # One-hot encode the board
    oh = onehot_transform(board)  # (9,9,9)
    row_h = oh.sum(axis=1)  # (9,9)
    col_h = oh.sum(axis=0)  # (9,9)
    blocks = []
    for br in range(3):
        for bc in range(3):
            blk = oh[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3, :].sum(axis=(0, 1))
            blocks.append(blk)
    blk_h = np.stack(blocks, axis=0)  # (9,9)
    feats = np.concatenate([row_h, col_h, blk_h], axis=0)  # (27,9)
    return feats.astype(np.float32)

def image_transform(
    board: np.ndarray,
    *,
    cell_px: int = 16,
    margin_px: int = 3,
    line_px: int = 1,
    bold_px: int = 1,
    font_size: int = 10,
    standardize: bool = True,
    font_path: str | None = None,
    outfile: str | None = None,
) -> np.ndarray:
    """Render a 9x9 Sudoku board to a grayscale image.

    Args:
        board (np.ndarray): 9x9 array with values in {0, 1..9}. Use 0 for blank
            cells.
        cell_px (int, optional): Pixel size of each cell. Defaults to 16.
        margin_px (int, optional): Outer padding around the grid. Defaults
            to 3.
        line_px (int, optional): Width of thin lines. Defaults to 1.
        bold_px (int, optional): Width of 3x3 divider lines. Defaults to 1.
        font_size (int, optional): Digit font size. Defaults to 10.
        standardize (bool, optional): If True, standardize pixel values to
            [0, 1]. Defaults to True.
        font_path (str | None, optional): Path to a .ttf font. If None,
            use default font. Defaults to None.
        outfile (str | None, optional): Path to save the image (e.g.,
            "board.png"). If None, do not write to disk. Defaults to None.

    Returns:
        np.ndarray: Grayscale image array of the Sudoku board.
        Output dimensions = (1, H, W) where H = W = margin_px * 2 + cell_px * 9.
    """
    assert board.shape == (9, 9), "board must be 9x9"
    W = H = margin_px * 2 + cell_px * 9
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default()
        )
    except Exception:
        font = ImageFont.load_default()

    # Helpers
    def cell_rect(r, c):
        x0 = margin_px + c * cell_px
        y0 = margin_px + r * cell_px
        x1 = x0 + cell_px
        y1 = y0 + cell_px
        return x0, y0, x1, y1

    # Grid lines
    # Thin lines
    for i in range(10):
        x = margin_px + i * cell_px
        y = margin_px + i * cell_px
        width_v = bold_px if i % 3 == 0 else line_px
        width_h = bold_px if i % 3 == 0 else line_px
        # Vertical
        draw.line([(x, margin_px), (x, H - margin_px)], fill="black", width=width_v)
        # Horizontal
        draw.line([(margin_px, y), (W - margin_px, y)], fill="black", width=width_h)

    # Numbers (centered in each cell)
    for r in range(9):
        for c in range(9):
            val = board[r, c]
            if val is None or int(val) == 0:
                continue
            text = str(int(val))
            x0, y0, x1, y1 = cell_rect(r, c)
            # Centering
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            tx = x0 + (cell_px - tw) / 2
            ty = y0 + (cell_px - th) / 2
            draw.text((tx, ty), text, fill="black", font=font)

    # Outer bold border (to ensure corners look crisp)
    draw.rectangle(
        [margin_px, margin_px, W - margin_px, H - margin_px],
        outline="black",
        width=bold_px,
    )

    if outfile:
        img.save(outfile)
        return outfile

    img_arr = sudoku_image_preprocess(img, standardize=standardize, to_tensor=False)

    return img_arr

def sudoku_image_preprocess(
    image: Image.Image,
    standardize: bool = True,
    to_tensor: bool = True
) -> np.ndarray:
    image = image.convert("L")  # Convert to grayscale
    img_arr = np.array(image)

    if standardize:
        # Standardize pixel values to [0, 1]
        img_arr = img_arr.astype(np.float32) / 255.0
    
    # add channel dimension (since its grayscale)
    img_arr = np.expand_dims(img_arr, axis=0)

    if to_tensor:
        out = torch.from_numpy(img_arr).float()
    else:
        out = img_arr
        
    return out