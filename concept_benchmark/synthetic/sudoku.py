"""
Generate Sudoku ConceptDataset
"""

import random
from collections.abc import Callable
import math

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from tqdm.auto import tqdm
from typing import Sequence, Tuple, Optional 

from concept_benchmark.data import ConceptDataset
from concept_benchmark.paths import data_dir
from concept_benchmark.synthetic.helper.sudoku_helper import (
    generate_invalid_board,
    generate_valid_board,
    get_concepts,
    normalize_positions,
    normalize_digits,
    cell_digit_concept_vector
)

SUDOKU_DIR = data_dir / "sudoku"


# TODO: concept masking toggles
def create_sudoku_dataset(
    *,
    n: int = 3,
    n_samples: int = 1000,
    valid_ratio: float = 0.5,
    max_corrupt: int = 3,
    data_type: str = "image",
    seed: int = 42,
    target_accuracy: float | None = None,
    concept_noise: float | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    dataset_name: str | None = None,
    add_cell_digit_concepts: bool = False,
    positions_subset: Sequence[tuple[int, int]] | None = None,
    digits_subset: Sequence[int] | None = None,
    cell_concept_prefix: str = "cell",
    **kwargs
) -> ConceptDataset:
    """Create a synthetic dataset of Sudoku boards with concepts.

    Args:
        n (int, optional): block size of Sudoku (default 3 for 9x9 board).
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
            Should take a board (N x N numpy array) and 
            return a transformed representation as a np.ndarray. 
            Default is None, which uses a simple flattening transform.
        dataset_name (str, optional): name of the dataset, used as folder name
            for saving images.
        target_accuracy (float, optional): Desired Bayes accuracy of the noisy
            labeler. Must be in [0.5, 1.0]. When provided, symmetric Bernoulli
            label noise is injected via ``ConceptDataset.sample_label_noise``.
        concept_noise (float, optional): Probability of flipping each concept bit
            independently. Must be in [0.0, 1.0].
        add_cell_digit_concepts: If True, append per-cell digit concepts
            of the form f"{cell_concept_prefix}({r+1},{c+1})_is_{d}".
        positions_subset: Optional subset of 0-indexed (row, col) pairs.
            None ⇒ all cells.
        digits_subset: Optional subset of digits to include. None ⇒ [1..N].
        cell_concept_prefix: Concept name prefix ("cell" by default).
    """

    # (existing image folder setup unchanged)
    if data_type == "image":
        dataset_name = dataset_name if dataset_name else \
            datetime.now().strftime("%Y%m%d_%H%M%S")
        ds_path = SUDOKU_DIR / dataset_name
        ds_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    if target_accuracy is not None:
        if not 0.5 <= target_accuracy <= 1.0:
            raise ValueError("target_accuracy must be within [0.5, 1.0]")
        epsilon = float(np.clip(1.0 - target_accuracy, 0.0, 0.5))
    else:
        epsilon = 0.0

    if concept_noise is not None:
        if not 0.0 <= concept_noise <= 1.0:
            raise ValueError("concept_noise must be within [0.0, 1.0]")
        concept_noise_p = float(concept_noise)
    else:
        concept_noise_p = 0.0

    N = n * n
    transform = transform or default_transform

    # Prep optional cell-digit concept config
    if add_cell_digit_concepts:
        _pos = normalize_positions(N, positions_subset)
        _digs = normalize_digits(N, digits_subset)
    else:
        _pos, _digs = [], []

    n_valid = int(round(n_samples * float(valid_ratio)))
    n_invalid = n_samples - n_valid

    X_list, C_list, y_list = [], [], []

    pbar = tqdm(total=n_valid + n_invalid, desc="Generating Sudoku dataset") if tqdm else None

    # ---- valid boards
    for i in range(n_valid):
        b = generate_valid_board(n=n)
        if data_type == "image":
            img_path = ds_path / f"valid_{i}.png"
            transform(b, outfile=img_path)
            X_list.append(img_path)
        else:
            X_list.append(transform(b))

        # Base (row/col/block) validity concepts
        c_base = np.ones(3 * N, dtype=np.int32)

        # Optional per-cell digit concepts
        if add_cell_digit_concepts:
            c_cell = cell_digit_concept_vector(b, _pos, _digs)
            c_vec = np.concatenate([c_base, c_cell], axis=0)
        else:
            c_vec = c_base

        C_list.append(c_vec)
        y_list.append(1)
        if pbar: pbar.update(1)

    # ---- invalid boards
    for i in range(n_invalid):
        num_actions = max(1, int(random.randint(1, max_corrupt)))
        b = generate_invalid_board(base_board=generate_valid_board(n=n), num_actions=num_actions)
        concepts = get_concepts(b, return_label=False)
        c_base = np.array(list(concepts.values()), dtype=np.int32).flatten()

        if data_type == "image":
            img_path = ds_path / f"invalid_{i}.png"
            transform(b, outfile=img_path)
            X_list.append(img_path)
        else:
            X_list.append(transform(b))

        if add_cell_digit_concepts:
            c_cell = cell_digit_concept_vector(b, _pos, _digs)
            c_vec = np.concatenate([c_base, c_cell], axis=0)
        else:
            c_vec = c_base

        C_list.append(c_vec)
        y_list.append(0)
        if pbar: pbar.update(1)

    if pbar: pbar.close()

    X = np.stack(X_list, axis=0)
    C = np.stack(C_list, axis=0).astype(np.int32)
    y = np.asarray(y_list, dtype=np.int32)

    if data_type == "image":
        np.savetxt(ds_path / "concepts.csv", C, delimiter=",")
        np.savetxt(ds_path / "labels.csv", y, delimiter=",")

    # ---- names
    concept_names = (
        [f"row_valid_{i + 1}" for i in range(N)]
        + [f"col_valid_{i + 1}" for i in range(N)]
        + [f"block_valid_{i + 1}" for i in range(N)]
    )

    if add_cell_digit_concepts:
        # Names follow the same order as _cell_digit_concept_vector
        extra_names = [
            f"{cell_concept_prefix}({r+1},{c+1})_is_{d}"
            for (r, c) in _pos
            for d in _digs
        ]
        concept_names.extend(extra_names)

    meta = {
        "classes": [0, 1],
        "concepts": concept_names,
        "data_type": data_type,
        "transform": transform.__name__ if transform else "default",
        "max_corrupt": max_corrupt,
        "seed": seed,
        "n": n,
        "N": N,
        "cell_digit_concepts": {
            "enabled": add_cell_digit_concepts,
            "positions_order_row_major_1based": [(r+1, c+1) for (r, c) in _pos],
            "digits_order": _digs,
            "name_prefix": cell_concept_prefix,
        },
    }

    if target_accuracy is not None:
        meta["label_noise"] = {
            "enabled": epsilon > 0.0,
            "scheme": "symmetric",
            "epsilon": epsilon,
            "target_accuracy": target_accuracy,
        }

    if concept_noise is not None:
        meta["concept_noise"] = {
            "enabled": concept_noise_p > 0.0,
            "scheme": "uniform_flip",
            "p": concept_noise_p,
        }

    if data_type == "image":
        kwargs = {"preprocess": sudoku_image_preprocess}
    else:
        kwargs = {}

    dataset = ConceptDataset(X=X, C=C, y=y, meta=meta, **kwargs)

    if concept_noise_p > 0.0:
        dataset.sample_concept_noise(p=concept_noise_p, rng=seed, enable=True)

    if epsilon > 0.0:
        dataset.sample_label_noise(p=epsilon, rng=seed, enable=True)

    return dataset


def default_transform(board: np.ndarray) -> np.ndarray:
    """Flattens an N×N board to a vector of length N*N."""
    return board.astype(np.float32).reshape(-1)


def onehot_transform(board: np.ndarray) -> np.ndarray:
    """Convert an N×N board (values 1..N or 0 for blanks) to one-hot (N,N,N)."""
    N = board.shape[0]
    x = board.astype(np.int64) - 1
    eye = np.eye(N, dtype=np.float32)
    # For blanks (0→-1), map to all-zeros row by masking negatives
    oh = eye[x.clip(0, N-1)]  # (N,N,N)
    oh[(board <= 0)] = 0.0
    return oh


def histogram_transform(board: np.ndarray) -> np.ndarray:
    """Convert an N×N board to per-unit digit histograms.
    Output shape: (3N, N) for rows, cols, and blocks.
    """
    N = board.shape[0]
    n = int(math.isqrt(N))
    assert n * n == N, "Board size must be a perfect square"
    oh = onehot_transform(board)  # (N,N,N)
    row_h = oh.sum(axis=1)        # (N,N)
    col_h = oh.sum(axis=0)        # (N,N)
    blocks = []
    for br in range(n):
        for bc in range(n):
            blk = oh[br * n : (br + 1) * n, bc * n : (bc + 1) * n, :].sum(axis=(0, 1))
            blocks.append(blk)
    blk_h = np.stack(blocks, axis=0)  # (N,N)
    feats = np.concatenate([row_h, col_h, blk_h], axis=0)  # (3N,N)
    return feats.astype(np.float32)

def image_transform(
    board: np.ndarray,
    *,
    cell_px: int = 40,
    margin_px: int = 3,
    line_px: int = 1,
    bold_px: int = 1,
    font_size: int = 10,
    standardize: bool = True,
    font_path: str | None = None,
    handwriting: bool = False,
    radius: float = 0.5,
    sigma: float = 0.0,
    angle: float = 98,
    outfile: str | None = None,
) -> np.ndarray:
    """Render an NxN Sudoku board to a grayscale image.
    Args:
        board (np.ndarray): NxN array with values in {0, 1..N}. Use 0 for blank
            cells.
        cell_px (int, optional): Pixel size of each cell. Defaults to 16.
        margin_px (int, optional): Outer padding around the grid. Defaults
            to 3.
        line_px (int, optional): Width of thin lines. Defaults to 1.
        bold_px (int, optional): Width of nxn divider lines. Defaults to 1.
        font_size (int, optional): Digit font size. Defaults to 10.
        standardize (bool, optional): If True, standardize pixel values to
            [0, 1]. Defaults to True.
        font_path (str | None, optional): Path to a .ttf font. If None,
            use default font. Defaults to None.
        outfile (str | None, optional): Path to save the image (e.g.,
            "board.png"). If None, do not write to disk. Defaults to None.
        handwriting (bool, optional): Makes the numbers look handwritten if True. 
            Defaults to False.
        radius (float, optional): size of Gaussian aperture. Defaults to 0.
            Only used if using handwriting=True
        sigma (float, optional): Standard deviation of Gaussian operator. Defaults to 12.
            Only used if using handwriting=True
        angle (float, optional): Direction of blur. Defults to 140.
            Only used if using handwriting=True

    Returns:
        np.ndarray: Grayscale image array of the Sudoku board.
        Output dimensions = (1, H, W) where H = W = margin_px * 2 + cell_px * N.
    """
    assert board.ndim == 2 and board.shape[0] == board.shape[1], "board must be square"
    N = board.shape[0]
    n = int(math.isqrt(N)); assert n*n == N, "board size must be n*n"
    W = H = margin_px * 2 + cell_px * N
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default(size=font_size)
        )
    except Exception:
        font = ImageFont.load_default(size=font_size)

    # Helpers
    def cell_rect(r, c):
        x0 = margin_px + c * cell_px
        y0 = margin_px + r * cell_px
        x1 = x0 + cell_px
        y1 = y0 + cell_px
        return x0, y0, x1, y1

    # Grid lines
    # Thin lines
    for i in range(N + 1):
        x = margin_px + i * cell_px
        y = margin_px + i * cell_px
        width_v = bold_px if i % n == 0 else line_px
        width_h = bold_px if i % n == 0 else line_px
        # Vertical
        draw.line([(x, margin_px), (x, H - margin_px)], fill="black", width=width_v)
        # Horizontal
        draw.line([(margin_px, y), (W - margin_px, y)], fill="black", width=width_h)

    # Numbers (centered in each cell)
    for r in range(N):
        for c in range(N):
            val = board[r, c]
            if val is None or int(val) == 0:
                continue
            v = int(val)
            if v <= 0:
                continue
            # Map 1..N to glyphs: 1-9->'1'-'9', 10->'A', 11->'B', ... up to 35->'Z', then 'a'..
            if v <= 9:
                text = str(v)
            else:
                idx = v - 10
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                if idx < len(alphabet):
                    text = alphabet[idx]
                else:
                    text = str(v)
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

    final_img = img
    if handwriting:
        import io
        from wand.image import Image as WandImage
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        with WandImage(blob=buf.getvalue()) as wimg:
            # Sketch modifies in-place
            wimg.sketch(radius=radius, sigma=sigma, angle=angle)
            blob = wimg.make_blob(format="PNG")
        final_img = Image.open(io.BytesIO(blob)).convert("RGB")
        

    if outfile:
        final_img.save(outfile)
        return str(outfile)

    return sudoku_image_preprocess(final_img, standardize=standardize, to_tensor=True)

def sudoku_image_preprocess(
    image: Image.Image,
    standardize: bool = True,
    to_tensor: bool = True,
    vit: bool = True
) -> np.ndarray:
    image = image.convert("L")  # Convert to grayscale
    img_arr = np.array(image)
    """
    Processes an image of a sudoku board/
    Args:
        image: Image to process
        standardize (bool, optional): If standardizing pixel values to within [0,1]. Defaults to True
        to_tensor (bool, optional): If converting to tensor or not. Defaults to True.
        vit (bool, optional): If using the image in a ViT backend model. Defaults to True.
    Returns: 
        An array representing the image.
    """

    if vit:
        # RGB + resize first
        try:
            resample = Image.Resampling.BICUBIC  # Pillow >= 9.1
        except AttributeError:
            resample = Image.BICUBIC
        image = image.convert("RGB").resize((224, 224), resample)

        # HWC -> CHW, scale to [0,1] before mean/std
        arr = np.asarray(image, dtype=np.float32) / 255.0          # [H,W,3] -> [0,1]
        arr = np.transpose(arr, (2, 0, 1))                         # [3,H,W]

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std

        return torch.from_numpy(arr).contiguous() if to_tensor else arr

    # Non-ViT path: grayscale, optional [0,1] standardization, CHW = [1,H,W]
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if standardize:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)  # [1,H,W]
    return torch.from_numpy(arr).contiguous() if to_tensor else arr
