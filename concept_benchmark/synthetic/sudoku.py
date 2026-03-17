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
from typing import Sequence


def _get_default_font(size):
    """Load a default font at the requested size, compatible with Pillow <10 and >=10."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


from concept_benchmark.data import ConceptDataset
from concept_benchmark.paths import data_dir
from concept_benchmark.synthetic.helper.sudoku_helper import (
    generate_invalid_board,
    generate_valid_board,
    get_concepts,
    normalize_positions,
    normalize_digits,
    cell_digit_concept_vector,
)
from concept_benchmark.synthetic.helper.sudoku_handwriting_helper import (
    SimpleHandwrittenGenerator,
    AdvancedHandwrittenGenerator,
    _build_given_mask,
    _synthesize_prior_from_neighbors,
    _render_inline_candidates,
    _draw_wobbly_circle,
)

SUDOKU_DIR = data_dir / "sudoku"


# Future: label noise, concept noise, concept masking toggles
def create_sudoku_dataset(
    *,
    n: int = 3,
    n_samples: int = 1000,
    valid_ratio: float = 0.5,
    max_corrupt: int = 3,
    data_type: str = "image",
    seed: int = 42,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    dataset_name: str | None = None,
    add_cell_digit_concepts: bool = False,
    positions_subset: Sequence[tuple[int, int]] | None = None,
    digits_subset: Sequence[int] | None = None,
    cell_concept_prefix: str = "cell",
    **kwargs,
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
        add_cell_digit_concepts: If True, append per-cell digit concepts
            of the form f"{cell_concept_prefix}({r+1},{c+1})_is_{d}".
        positions_subset: Optional subset of 0-indexed (row, col) pairs.
            None ⇒ all cells.
        digits_subset: Optional subset of digits to include. None ⇒ [1..N].
        cell_concept_prefix: Concept name prefix ("cell" by default).
    """

    # (existing image folder setup unchanged)
    if data_type == "image":
        dataset_name = (
            dataset_name if dataset_name else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        ds_path = SUDOKU_DIR / dataset_name
        ds_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

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

    pbar = (
        tqdm(total=n_valid + n_invalid, desc="Generating Sudoku dataset")
        if tqdm
        else None
    )

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
        if pbar:
            pbar.update(1)

    # ---- invalid boards
    for i in range(n_invalid):
        num_actions = max(1, int(random.randint(1, max_corrupt)))
        inv_seed = random.randint(0, 2**31 - 1)
        b = generate_invalid_board(
            base_board=generate_valid_board(n=n), num_actions=num_actions, seed=inv_seed
        )
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
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

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
            f"{cell_concept_prefix}({r + 1},{c + 1})_is_{d}"
            for (r, c) in _pos
            for d in _digs
        ]
        concept_names.extend(extra_names)

    meta = {
        "classes": [0, 1],
        "concepts": concept_names,
        "data_type": data_type,
        "transform": getattr(
            transform,
            "__name__",
            getattr(transform, "func", transform).__name__ if transform else "default",
        ),
        "max_corrupt": max_corrupt,
        "seed": seed,
        "n": n,
        "N": N,
        "cell_digit_concepts": {
            "enabled": add_cell_digit_concepts,
            "positions_order_row_major_1based": [(r + 1, c + 1) for (r, c) in _pos],
            "digits_order": _digs,
            "name_prefix": cell_concept_prefix,
        },
    }

    if data_type == "image":
        kwargs = {"preprocess": sudoku_image_preprocess}
    else:
        kwargs = {}

    return ConceptDataset(X=X, C=C, y=y, meta=meta, **kwargs)


def default_transform(board: np.ndarray) -> np.ndarray:
    """Flattens an N×N board to a vector of length N*N."""
    return board.astype(np.float32).reshape(-1)


def onehot_transform(board: np.ndarray) -> np.ndarray:
    """Convert an N×N board (values 1..N or 0 for blanks) to one-hot (N,N,N)."""
    N = board.shape[0]
    x = board.astype(np.int64) - 1
    eye = np.eye(N, dtype=np.float32)
    # For blanks (0→-1), map to all-zeros row by masking negatives
    oh = eye[x.clip(0, N - 1)]  # (N,N,N)
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
    row_h = oh.sum(axis=1)  # (N,N)
    col_h = oh.sum(axis=0)  # (N,N)
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
    margin_px: int = 2,
    line_px: int = 1,
    bold_px: int = 3,
    font_size: int = 10,
    standardize: bool = True,
    font_path: str | None = None,
    handwriting: bool = True,
    # Starters control (printed vs user-filled colors)
    given_mask: np.ndarray | None = None,
    starters_ratio: float | None = 0.35,
    starters_count: int | None = None,
    starters_seed: int | None = None,
    given_color: tuple[int, int, int] = (30, 30, 30),  # gray/black (printed)
    fill_color: tuple[int, int, int] = (26, 71, 180),  # blue (user ink)
    # Prior candidates (built from adjacent non-starter pairs/triples)
    prior_options: dict | None = None,
    prior_groups_ratio: float = 0.22,
    prior_groups_seed: int | None = 1337,
    prior_group_sizes: tuple[int, ...] = (2,),
    prior_allow_diagonal: bool = False,
    # (compat only, not used directly)
    outfile: str | None = None,
    # NEW: if True, also return (starters, candidates) alongside the image
    return_meta: bool = False,
) -> np.ndarray | str | tuple:
    """Render an NxN Sudoku board to an RGB image with prior-option bubbles (handwritten).
    Each bubble shows *all* options inline in one row (tight kerning). Starters never get bubbles.
    """
    import math

    N = board.shape[0]
    assert board.ndim == 2 and N == board.shape[1], "board must be square"
    n = int(math.isqrt(N))
    assert n * n == N, "board size must be n*n"
    W = H = margin_px * 2 + cell_px * N

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Base font for non-handwritten/letters
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else _get_default_font(font_size)
        )
    except (OSError, IOError):
        font = _get_default_font(font_size)

    def cell_rect(r, c):
        x0 = margin_px + c * cell_px
        y0 = margin_px + r * cell_px
        x1 = x0 + cell_px
        y1 = y0 + cell_px
        return x0, y0, x1, y1

    # Grid
    for i in range(N + 1):
        x = margin_px + i * cell_px
        y = margin_px + i * cell_px
        wv = bold_px if i % n == 0 else line_px
        wh = bold_px if i % n == 0 else line_px
        draw.line([(x, margin_px), (x, H - margin_px)], fill=(0, 0, 0), width=wv)
        draw.line([(margin_px, y), (W - margin_px, y)], fill=(0, 0, 0), width=wh)

    # ---- build handwritten generator + starters + prior_options ----
    generator = None
    if handwriting:
        from concept_benchmark.paths import pkg_dir

        _fonts_dir = str(pkg_dir / "data" / "fonts")
        try:
            import cv2  # noqa: F401

            generator = AdvancedHandwrittenGenerator(fonts_dir=_fonts_dir)
        except (ImportError, OSError):
            generator = SimpleHandwrittenGenerator(fonts_dir=_fonts_dir)

    # Starters mask (printed vs user-filled)
    # NOTE: this is now *outside* the except, so it always runs.
    starters = _build_given_mask(
        board,
        given_mask=given_mask,
        starters_ratio=starters_ratio,
        starters_count=starters_count,
        starters_seed=starters_seed,
    )

    # Build PRIOR candidates from adjacent non-starter pairs/triples if not provided
    if prior_options is None:
        prior_options = _synthesize_prior_from_neighbors(
            board,
            starters,
            groups_count=None,
            groups_ratio=prior_groups_ratio,
            group_sizes=prior_group_sizes,
            allow_diagonal=prior_allow_diagonal,
            seed=prior_groups_seed,
        )

    # Simple 9x9 numeric encoding of candidates for downstream use:
    #   candidates[r, c] = number of candidate digits for that cell.
    candidates = np.zeros_like(board, dtype=np.int32)
    for (r, c), opts in prior_options.items():
        # ignore junk, keep positive ints
        clean = [
            int(d) for d in opts if isinstance(d, (int, np.integer)) and int(d) > 0
        ]
        candidates[r, c] = len(clean)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # ---- FIRST PASS: draw prior-option bubbles for NON-STARTERS (adaptive + clamped)
    for r in range(N):
        for c in range(N):
            if starters[r, c]:
                continue

            if not prior_options or ((r, c) not in prior_options):
                continue

            # digits only; ignore junk/zeros — and ALWAYS include the cell's main digit
            raw = board[r, c]
            own = int(raw) if raw is not None else 0

            opts = {
                int(d)
                for d in prior_options.get((r, c), [])
                if isinstance(d, (int, np.integer)) and int(d) > 0
            }
            if own > 0:
                opts.add(own)

            opts = sorted(opts)
            if not opts:
                continue

            x0, y0, x1, y1 = cell_rect(r, c)
            cell_w = x1 - x0
            cell_h = y1 - y0

            # target corner and padding
            inset = max(2, int(cell_px * 0.06))
            pad = max(2, int(cell_px * 0.05))

            # Compute the max bubble box we can afford in the corner
            max_w = cell_w - 2 * inset
            max_h = cell_h - 2 * inset
            if max_w <= 6 or max_h <= 6:
                continue  # cell is too small (shouldn't happen)

            # Start with a generous mini size, then *shrink* until it fits
            mini = max(12, int(cell_px * 0.46))
            fitted = None
            tries = 0
            while mini >= 8 and tries < 6:
                seq_img = _render_inline_candidates(
                    generator, opts, fill_color, size=mini, seed_base=(r * N + c) * 997
                )
                bubble_w = seq_img.width + 2 * pad
                bubble_h = seq_img.height + 2 * pad

                if bubble_w <= max_w and bubble_h <= max_h:
                    fitted = (mini, seq_img, bubble_w, bubble_h)
                    break

                # shrink a bit and try again
                mini = int(mini * 0.88)
                tries += 1

            if fitted is None:
                # As a last resort, drop to digits only (no circle) or skip; here we skip
                continue

            mini, seq_img, bubble_w, bubble_h = fitted

            # Place top-left, fully inside the cell
            left = x0 + inset
            top = y0 + inset
            right = left + bubble_w
            bottom = top + bubble_h

            # (safety) clamp within the cell; this preserves size
            if right > x1 - 1:
                shift = right - (x1 - 1)
                left -= shift
                right -= shift
            if bottom > y1 - 1:
                shift = bottom - (y1 - 1)
                top -= shift
                bottom -= shift

            # Draw the fine, wobbly circle outline
            _draw_wobbly_circle(
                draw,
                (left, top, right, bottom),
                color=fill_color,
                width=1,
                seed=(r * N + c) * 137 + len(opts),
            )

            # Center the strip inside the circle
            tx = int(left + (bubble_w - seq_img.width) / 2)
            ty = int(top + (bubble_h - seq_img.height) / 2)
            img.paste(seq_img, (tx, ty), seq_img)

    # ---- SECOND PASS: draw main digits
    for r in range(N):
        for c in range(N):
            raw = board[r, c]
            if raw is None or int(raw) == 0:
                continue
            v = int(raw)
            if v <= 0:
                continue

            x0, y0, x1, y1 = cell_rect(r, c)
            text = (
                str(v)
                if v <= 9
                else alphabet[v - 10]
                if 0 <= (v - 10) < len(alphabet)
                else str(v)
            )
            color = given_color if starters[r, c] else fill_color

            if starters[r, c]:
                # ALWAYS printed font for starters
                tb = draw.textbbox((0, 0), text, font=_get_default_font(font_size))
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                tx = x0 + (cell_px - tw) / 2
                ty = y0 + (cell_px - th) / 2
                draw.text(
                    (tx, ty), text, fill=given_color, font=_get_default_font(font_size)
                )
            else:
                # Non-starters: handwritten if possible, otherwise printed
                if text.isdigit() and (generator is not None):
                    hand_size_main = max(8, int(cell_px * 0.82))
                    rng_seed = (r * N + c) * 131 + v
                    digit_img = generator.generate(
                        digit=int(text),
                        size=hand_size_main,
                        color=fill_color,
                        rng_seed=rng_seed,
                    )
                    arr = np.array(digit_img, dtype=np.uint8)
                    if arr.shape[-1] == 4:
                        arr[:, :, 3] = np.clip(
                            arr[:, :, 3].astype(np.float32) * 1.25, 0, 255
                        ).astype(np.uint8)
                    digit_img = Image.fromarray(arr, mode="RGBA")
                    dw, dh = digit_img.size
                    tx = x0 + (cell_px - dw) // 2
                    ty = y0 + (cell_px - dh) // 2
                    img.paste(digit_img, (int(tx), int(ty)), digit_img)

    # Outer border
    draw.rectangle(
        [margin_px, margin_px, W - margin_px, H - margin_px],
        outline=(0, 0, 0),
        width=bold_px,
    )

    # ----- returns -----
    if outfile:
        img.save(outfile)
        if return_meta:
            return str(outfile), starters, candidates
        return str(outfile)

    # keep RGB; tensorize downstream
    arr = sudoku_image_preprocess(img, standardize=standardize, to_tensor=True)
    if return_meta:
        return arr, starters, candidates
    return arr


def sudoku_image_preprocess(
    image: Image.Image,
    standardize: bool = True,
    to_tensor: bool = True,
    vit: bool = True,
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
        arr = np.asarray(image, dtype=np.float32) / 255.0  # [H,W,3] -> [0,1]
        arr = np.transpose(arr, (2, 0, 1))  # [3,H,W]

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std

        return torch.from_numpy(arr).contiguous() if to_tensor else arr

    # Non-ViT path: grayscale, optional [0,1] standardization, CHW = [1,H,W]
    arr = np.asarray(image, dtype=np.float32)
    if standardize:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)  # [1,H,W]
    return torch.from_numpy(arr).contiguous() if to_tensor else arr
