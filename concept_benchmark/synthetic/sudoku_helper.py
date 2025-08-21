import random
from typing import Optional

import numpy as np

# --------------------------
# Base valid board
# --------------------------
BASE_BOARD = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [9, 1, 2, 3, 4, 5, 6, 7, 8],
    ]
)


# --------------------------
# Transformations to shuffle valid boards
# Used to generate valid Sudoku boards
# --------------------------
def shuffle_rows(board):
    for band in range(0, 9, 3):
        rows = list(range(band, band + 3))
        np.random.shuffle(rows)
        board[band : band + 3] = board[rows]
    return board


def shuffle_columns(board):
    for stack in range(0, 9, 3):
        cols = list(range(stack, stack + 3))
        np.random.shuffle(cols)
        board[:, stack : stack + 3] = board[:, cols]
    return board


def shuffle_row_bands(board):
    bands = [0, 1, 2]
    np.random.shuffle(bands)
    return np.vstack([board[i * 3 : (i + 1) * 3] for i in bands])


def shuffle_col_stacks(board):
    stacks = [0, 1, 2]
    np.random.shuffle(stacks)
    return np.hstack([board[:, i * 3 : (i + 1) * 3] for i in stacks])


def relabel_digits(board):
    digits = list(range(1, 10))
    perm = digits[:]
    random.shuffle(perm)
    mapping = {d: perm[i] for i, d in enumerate(digits)}
    return np.vectorize(lambda x: mapping[x])(board)


def generate_valid_board():
    board = BASE_BOARD.copy()
    board = shuffle_rows(board)
    board = shuffle_columns(board)
    board = shuffle_row_bands(board)
    board = shuffle_col_stacks(board)
    board = relabel_digits(board)
    return board


# --------------------------
# Invalid board generators 
# --------------------------
# Categories:
#   A) Invalid by swapping numbers within rows/columns
#      - Invalid rows only (two rows become invalid, cols/blocks remain valid)
#      - Invalid columns only (two cols become invalid, rows/blocks remain valid)
#   B) Invalid by swapping ENTIRE rows/columns across bands/stacks
#      - Invalid blocks only (rows/cols remain valid)
#   C) Invalid by duplicating numbers in rows/columns
#      - Each action invalidates at most 3 units (row/column/block) due to a single-cell overwrite


def invalid_rows_only_by_within_column_swap(board: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Swap two cells that share the same column and the same 3x3 block-band (i.e., same column, different
    rows within one band). Effect: exactly TWO row violations; columns and blocks remain valid.
    """
    assert board.shape == (9, 9)
    rng = random.Random(seed)
    b = board.copy()
    c = rng.randrange(9)
    br = rng.randrange(3)  # band index 0..2
    rows = [br * 3 + i for i in range(3)]
    r1, r2 = rng.sample(rows, 2)
    b[r1, c], b[r2, c] = b[r2, c], b[r1, c]
    return b


def invalid_cols_only_by_within_row_swap(board: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Swap two cells that share the same row and the same 3x3 stack (i.e., same row, different columns within
    one stack). Effect: exactly TWO column violations; rows and blocks remain valid.
    """
    assert board.shape == (9, 9)
    rng = random.Random(seed)
    b = board.copy()
    r = rng.randrange(9)
    sc = rng.randrange(3)  # stack index 0..2
    cols = [sc * 3 + i for i in range(3)]
    c1, c2 = rng.sample(cols, 2)
    b[r, c1], b[r, c2] = b[r, c2], b[r, c1]
    return b


def invalid_blocks_only_by_row_swap_across_bands(board: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Swap two ENTIRE rows from different bands. Rows/columns remain valid (they are permutations and column
    multisets unchanged), but affected 3x3 blocks become invalid (composition within blocks changes).
    """
    assert board.shape == (9, 9)
    rng = random.Random(seed)
    b = board.copy()
    # choose two rows from different bands
    br1, br2 = rng.sample([0, 1, 2], 2)
    r1 = br1 * 3 + rng.randrange(3)
    r2 = br2 * 3 + rng.randrange(3)
    b[[r1, r2]] = b[[r2, r1]]
    return b


def invalid_blocks_only_by_col_swap_across_stacks(board: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Swap two ENTIRE columns from different stacks. Rows/columns remain valid (still permutations), while
    the composition of blocks is disturbed, invalidating blocks.
    """
    assert board.shape == (9, 9)
    rng = random.Random(seed)
    b = board.copy()
    sc1, sc2 = rng.sample([0, 1, 2], 2)
    c1 = sc1 * 3 + rng.randrange(3)
    c2 = sc2 * 3 + rng.randrange(3)
    b[:, [c1, c2]] = b[:, [c2, c1]]
    return b


def duplicate_in_row(board: np.ndarray, r: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Duplicate one value within row r by copying a value from column c1 into column c2 (c2!=c1).
    Effect of ONE action: invalidates that row, the target column, and the target 3x3 block (<= 3 concepts).
    """
    assert board.shape == (9, 9)
    rng = random.Random(seed)
    b = board.copy()
    c1, c2 = rng.sample(range(9), 2)
    b[r, c2] = b[r, c1]
    return b


def duplicate_in_col(board: np.ndarray, c: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Duplicate one value within column c by copying a value from row r1 into row r2 (r2!=r1).
    Effect of ONE action: invalidates that column, the target row, and the target 3x3 block (<= 3 concepts).
    """
    assert board.shape == (9, 9)
    rng = random.Random(seed)
    b = board.copy()
    r1, r2 = rng.sample(range(9), 2)
    b[r2, c] = b[r1, c]
    return b


# Dispatcher to generate invalid boards according to a chosen mode
INVALID_MODES = {
    # A) swap numbers locally inside units
    "rows_only_swap_within_column": invalid_rows_only_by_within_column_swap,
    "cols_only_swap_within_row": invalid_cols_only_by_within_row_swap,
    # B) swap entire rows/cols across bands/stacks → blocks-only invalid
    "blocks_only_row_swap_across_bands": invalid_blocks_only_by_row_swap_across_bands,
    "blocks_only_col_swap_across_stacks": invalid_blocks_only_by_col_swap_across_stacks, 
}


def generate_invalid_board(
    base_board: Optional[np.ndarray] = None,
    num_actions: int = 1,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Create an invalid board from a valid one using structured corruptions.

    Args:
        base_board: starting valid board; if None, a new valid board is generated.
        num_actions: number of corruption actions to apply (>=1).
        mode: one of INVALID_MODES keys, or one of {"duplicate_row","duplicate_col"} for C),
              or None → choose randomly from all modes.
        seed: optional seed controlling the first action; subsequent actions use derived seeds.

    Notes:
        - For duplicate_* modes, rows/cols are chosen at random for each action.
        - Each duplicate action invalidates at most three concepts (row/col/block) by construction.
    """
    if base_board is None:
        base_board = generate_valid_board()
    rng = random.Random(seed)
    b = base_board.copy()

    for t in range(num_actions):
        cur_seed = rng.randrange(1 << 30)
        if mode is None:
            # include duplicate modes in random choice
            mode_choice = rng.choice(list(INVALID_MODES.keys()) + ["duplicate_row", "duplicate_col"])
        else:
            mode_choice = mode

        if mode_choice in INVALID_MODES:
            b = INVALID_MODES[mode_choice](b, seed=cur_seed)
        elif mode_choice == "duplicate_row":
            r = rng.randrange(9)
            b = duplicate_in_row(b, r=r, seed=cur_seed)
        elif mode_choice == "duplicate_col":
            c = rng.randrange(9)
            b = duplicate_in_col(b, c=c, seed=cur_seed)
        else:
            raise ValueError(f"Unknown invalid mode: {mode_choice}")

    return b


# --------------------------
# Create incomplete boards (mask clues)
# --------------------------

def make_incomplete_board(
    board: np.ndarray,
    num_clues: int | None = 30,
    ensure_one_per_row: bool = False,
    ensure_one_per_col: bool = False,
    ensure_one_per_block: bool = False,
    seed: int | None = None,
    blank_value: int = 0,
) -> np.ndarray:
    """
    Return an incomplete Sudoku board by masking cells from a complete solution.

    Args:
        board: (9,9) numpy array with digits 1..9 (complete/valid solution assumed).
        num_clues: target number of visible cells to keep (defaults to 30). If None,
            a random number in [17, 60] is sampled (17 is a classic minimal clue bound).
        ensure_one_per_row: if True, ensures at least one clue remains in every row.
        ensure_one_per_col: if True, ensures at least one clue remains in every column.
        ensure_one_per_block: if True, ensures at least one clue remains in every 3x3 block.
        seed: optional RNG seed for reproducibility.
        blank_value: value to place in removed cells (0 by default).

    Returns:
        puzzle: (9,9) numpy array with some cells set to `blank_value`.

    Notes:
        - This function does *not* check extendability/uniqueness; it only masks cells.
        - The "ensure_*" options may force keeping more clues than requested if necessary.
    """
    assert board.shape == (9, 9), "board must be 9x9"
    rng = np.random.default_rng(seed)

    # Determine target number of clues
    if num_clues is None:
        num_clues = int(rng.integers(low=17, high=61))  # inclusive low, exclusive high
    num_clues = int(np.clip(num_clues, 0, 81))

    # Build a set of indices we must keep to satisfy constraints
    keep: set[tuple[int, int]] = set()

    if ensure_one_per_row:
        for r in range(9):
            c = int(rng.integers(0, 9))
            keep.add((r, c))

    if ensure_one_per_col:
        for c in range(9):
            r = int(rng.integers(0, 9))
            keep.add((r, c))

    if ensure_one_per_block:
        for br in range(3):
            for bc in range(3):
                r = int(rng.integers(br * 3, br * 3 + 3))
                c = int(rng.integers(bc * 3, bc * 3 + 3))
                keep.add((r, c))

    # Ensure we do not request fewer clues than already forced by constraints
    min_required = len(keep)
    if num_clues < min_required:
        num_clues = min_required

    # Sample the remaining cells to keep uniformly without replacement
    all_idx = [(r, c) for r in range(9) for c in range(9) if (r, c) not in keep]
    remaining_to_keep = num_clues - len(keep)
    if remaining_to_keep > 0:
        chosen = rng.choice(len(all_idx), size=remaining_to_keep, replace=False)
        for i in np.atleast_1d(chosen):
            keep.add(all_idx[int(i)])

    # Construct the puzzle
    puzzle = np.full_like(board, fill_value=blank_value)
    for (r, c) in keep:
        puzzle[r, c] = board[r, c]

    return puzzle


# --------------------------
# Concept extraction
# --------------------------
def get_concepts(board, return_label=False):
    concepts = {}

    row_valid = []
    for i in range(9):
        unique = np.unique(board[i, :])
        row_valid.append(int(len(unique) == 9))
    concepts["row_valid"] = row_valid

    col_valid = []
    for j in range(9):
        unique = np.unique(board[:, j])
        col_valid.append(int(len(unique) == 9))
    concepts["col_valid"] = col_valid

    block_valid = []
    for bi in range(3):
        for bj in range(3):
            block = board[bi * 3 : (bi + 1) * 3, bj * 3 : (bj + 1) * 3].flatten()
            unique = np.unique(block)
            block_valid.append(int(len(unique) == 9))
    concepts["block_valid"] = block_valid

    if return_label:
        board_valid = int(all(row_valid) and all(col_valid) and all(block_valid))
        concepts["board_valid"] = board_valid

    return concepts


# --------------------------
# Partial-board concepts (work with blanks)
# --------------------------

def get_partial_concepts(
    board: np.ndarray,
    blank_value: int = 0,
    return_board_flags: bool = True,
):
    """
    Compute binary concepts for *incomplete* boards.

    Concepts returned (all lists of length 9):
      - row_consistent / col_consistent / block_consistent: 1 if the unit has
        no duplicate non-blank digits (ignores blanks), else 0.
      - row_complete / col_complete / block_complete: 1 if the unit has no blanks
        (i.e., 9 filled cells), else 0.

    If `return_board_flags` is True, also returns:
      - board_consistent: AND of all unit-consistent flags (27 units)
      - board_complete: AND of all unit-complete flags (27 units)

    Notes:
      - This does not check global Sudoku validity for completed units; it
        focuses on duplicate-free (consistency) and filled-ness (completeness).
    """
    assert board.shape == (9, 9), "board must be 9x9"

    def unit_consistent(vals):
        vals = np.asarray(vals)
        mask = vals != blank_value
        nz = vals[mask]
        # no duplicates among non-blanks
        return int(len(nz) == len(np.unique(nz)))

    def unit_complete(vals):
        vals = np.asarray(vals)
        return int(np.count_nonzero(vals != blank_value) == 9)

    # Row concepts
    row_consistent = [unit_consistent(board[r, :]) for r in range(9)]
    row_complete = [unit_complete(board[r, :]) for r in range(9)]

    # Column concepts
    col_consistent = [unit_consistent(board[:, c]) for c in range(9)]
    col_complete = [unit_complete(board[:, c]) for c in range(9)]

    # Block concepts
    block_consistent = []
    block_complete = []
    for br in range(3):
        for bc in range(3):
            blk = board[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3].reshape(-1)
            block_consistent.append(unit_consistent(blk))
            block_complete.append(unit_complete(blk))

    concepts = {
        "row_consistent": row_consistent,
        "col_consistent": col_consistent,
        "block_consistent": block_consistent,
        "row_complete": row_complete,
        "col_complete": col_complete,
        "block_complete": block_complete,
    }

    if return_board_flags:
        all_consistent = (
            all(row_consistent) and all(col_consistent) and all(block_consistent)
        )
        all_complete = (
            all(row_complete) and all(col_complete) and all(block_complete)
        )
        concepts["board_consistent"] = int(all_consistent)
        concepts["board_complete"] = int(all_complete)

    return concepts


# --------------------------
# Concept noise utilities
# --------------------------

def _bernoulli_mask(shape, p, rng):
    """Sample a boolean mask with True ~ Bernoulli(p)."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"probability p must be in [0,1], got {p}")
    return rng.random(shape) < p


def add_noise_to_concepts(
    concepts: dict,
    eta: float = 0.1,
    seed: int | None = None,
    keys: list[str] | None = None,
    include_board_flags: bool = False,
    per_key_eta: dict[str, float] | None = None,
) -> dict:
    """
    Flip concept bits independently with probability eta (η), modeling label noise.

    Args:
        concepts: dict mapping concept-name -> list/array of 0/1 (e.g., from get_concepts or get_partial_concepts)
        eta: global flip probability η for Bernoulli noise (0..1).
        seed: optional RNG seed.
        keys: if provided, only apply noise to these concept keys; otherwise all keys.
        include_board_flags: if False (default), do NOT perturb aggregate board flags
            such as 'board_valid', 'board_consistent', 'board_complete'. Set True to also noise them.
        per_key_eta: optional dict mapping specific keys to their own η; overrides `eta` per key.

    Returns:
        noisy_concepts: a **new** dict with the same keys and noisy bit lists (dtype ints 0/1).
    """
    rng = np.random.default_rng(seed)

    def _should_skip(k: str) -> bool:
        if not include_board_flags and k in {"board_valid", "board_consistent", "board_complete"}:
            return True
        if keys is not None and k not in keys:
            return True
        return False

    out = {}
    for k, v in concepts.items():
        arr = np.asarray(v)
        # pass through non-binary values untouched
        if _should_skip(k) or arr.ndim == 0:
            out[k] = v
            continue
        if arr.dtype != np.bool_ and not np.isin(arr, [0, 1]).all():
            # Not a binary vector -> leave as-is
            out[k] = v
            continue
        p = eta
        if per_key_eta is not None and k in per_key_eta:
            p = per_key_eta[k]
        mask = _bernoulli_mask(arr.shape, p, rng)
        noisy = np.logical_xor(arr.astype(bool), mask).astype(np.int32)
        out[k] = noisy.tolist() if isinstance(v, list) else noisy
    return out


def add_noise_to_concept_vector(vec: np.ndarray, eta: float = 0.1, seed: int | None = None) -> np.ndarray:
    """
    Flip bits of a 0/1 numpy vector with probability η (independent Bernoulli noise).
    Returns a new array with same shape and dtype int32.
    """
    if vec.ndim == 0:
        return vec
    rng = np.random.default_rng(seed)
    mask = _bernoulli_mask(vec.shape, eta, rng)
    return np.logical_xor(vec.astype(bool), mask).astype(np.int32)
