from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

# --------------------------
# Parametric base board & valid board generation (supports N = n*n)
# --------------------------


def _assert_board_size(board: np.ndarray) -> tuple[int, int]:
    assert board.ndim == 2 and board.shape[0] == board.shape[1], "board must be square"
    N = board.shape[0]
    n = int(np.sqrt(N))
    assert n * n == N, f"board size {N} is not a perfect square (needed for n×n blocks)"
    return N, n


def build_base_board(n: int) -> np.ndarray:
    """Construct a canonical valid Sudoku of size N=n*n using the standard pattern.

    pattern(r,c) = (n*(r % n) + r//n + c) % N
    digits are 1..N
    """
    N = n * n
    board = np.fromfunction(
        lambda r, c: (n * (r % n) + r // n + c) % N + 1, (N, N), dtype=int
    )
    return board.astype(int)


def shuffle_rows(board: np.ndarray, n: int | None = None) -> np.ndarray:
    N, n0 = _assert_board_size(board)
    n = n0 if n is None else n
    for band in range(0, N, n):
        rows = list(range(band, band + n))
        np.random.shuffle(rows)
        board[band : band + n] = board[rows]
    return board


def shuffle_columns(board: np.ndarray, n: int | None = None) -> np.ndarray:
    N, n0 = _assert_board_size(board)
    n = n0 if n is None else n
    for stack in range(0, N, n):
        cols = list(range(stack, stack + n))
        np.random.shuffle(cols)
        board[:, stack : stack + n] = board[:, cols]
    return board


def shuffle_row_bands(board: np.ndarray, n: int | None = None) -> np.ndarray:
    _, n0 = _assert_board_size(board)
    n = n0 if n is None else n
    bands = list(range(n))
    np.random.shuffle(bands)
    return np.vstack([board[i * n : (i + 1) * n] for i in bands])


def shuffle_col_stacks(board: np.ndarray, n: int | None = None) -> np.ndarray:
    _, n0 = _assert_board_size(board)
    n = n0 if n is None else n
    stacks = list(range(n))
    np.random.shuffle(stacks)
    return np.hstack([board[:, i * n : (i + 1) * n] for i in stacks])


def relabel_digits(board: np.ndarray, N: int | None = None) -> np.ndarray:
    N0, _ = _assert_board_size(board)
    N = N0 if N is None else N
    digits = list(range(1, N + 1))
    perm = digits[:]
    random.shuffle(perm)
    mapping = {d: perm[i] for i, d in enumerate(digits)}
    vfunc = np.vectorize(lambda x: mapping[int(x)])
    return vfunc(board).astype(int)


def generate_valid_board(n: int = 3) -> np.ndarray:
    """Generate a valid N=n*n Sudoku board using isotopy-safe shuffles and relabeling."""
    N = n * n
    board = build_base_board(n)
    board = shuffle_rows(board, n)
    board = shuffle_columns(board, n)
    board = shuffle_row_bands(board, n)
    board = shuffle_col_stacks(board, n)
    board = relabel_digits(board, N)
    return board


# Backward-compatibility constant for 9x9 (n=3)
BASE_BOARD = build_base_board(3)


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


def invalid_rows_only_by_within_column_swap(
    board: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """
    Swap two cells that share the same column and the same n x n block-band (i.e., same column, different
    rows within one band). Effect: exactly TWO row violations; columns and blocks remain valid.
    """
    N, n = _assert_board_size(board)
    rng = random.Random(seed)
    b = board.copy()
    c = rng.randrange(N)
    br = rng.randrange(n)  # band index 0..n-1
    rows = [br * n + i for i in range(n)]
    r1, r2 = rng.sample(rows, 2)
    b[r1, c], b[r2, c] = b[r2, c], b[r1, c]
    return b


def invalid_cols_only_by_within_row_swap(
    board: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """
    Swap two cells that share the same row and the same n x n stack (i.e., same row, different columns within
    one stack). Effect: exactly TWO column violations; rows and blocks remain valid.
    """
    N, n = _assert_board_size(board)
    rng = random.Random(seed)
    b = board.copy()
    r = rng.randrange(N)
    sc = rng.randrange(n)  # stack index 0..n-1
    cols = [sc * n + i for i in range(n)]
    c1, c2 = rng.sample(cols, 2)
    b[r, c1], b[r, c2] = b[r, c2], b[r, c1]
    return b


def invalid_blocks_only_by_row_swap_across_bands(
    board: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """
    Swap two ENTIRE rows from different bands. Rows/columns remain valid (they are permutations and column
    multisets unchanged), but affected n x n blocks become invalid (composition within blocks changes).
    """
    N, n = _assert_board_size(board)
    rng = random.Random(seed)
    b = board.copy()
    br1, br2 = rng.sample(list(range(n)), 2)
    r1 = br1 * n + rng.randrange(n)
    r2 = br2 * n + rng.randrange(n)
    b[[r1, r2]] = b[[r2, r1]]
    return b


def invalid_blocks_only_by_col_swap_across_stacks(
    board: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """
    Swap two ENTIRE columns from different stacks. Rows/columns remain valid (still permutations), while
    the composition of blocks is disturbed, invalidating blocks.
    """
    N, n = _assert_board_size(board)
    rng = random.Random(seed)
    b = board.copy()
    sc1, sc2 = rng.sample(list(range(n)), 2)
    c1 = sc1 * n + rng.randrange(n)
    c2 = sc2 * n + rng.randrange(n)
    b[:, [c1, c2]] = b[:, [c2, c1]]
    return b


def duplicate_in_row(board: np.ndarray, r: int, seed: int | None = None) -> np.ndarray:
    """
    Duplicate one value within row r by copying a value from column c1 into column c2 (c2!=c1).
    Effect of ONE action: invalidates that row, the target column, and the target block (<= 3 concepts).
    """
    N, n = _assert_board_size(board)
    rng = random.Random(seed)
    b = board.copy()
    c1, c2 = rng.sample(range(N), 2)
    b[r % N, c2] = b[r % N, c1]
    return b


def duplicate_in_col(board: np.ndarray, c: int, seed: int | None = None) -> np.ndarray:
    """
    Duplicate one value within column c by copying a value from row r1 into row r2 (r2!=r1).
    Effect of ONE action: invalidates that column, the target row, and the target block (<= 3 concepts).
    """
    N, n = _assert_board_size(board)
    rng = random.Random(seed)
    b = board.copy()
    r1, r2 = rng.sample(range(N), 2)
    b[r2, c % N] = b[r1, c % N]
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
    base_board: np.ndarray | None = None,
    num_actions: int = 1,
    mode: str | None = None,
    seed: int | None = None,
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
    N, n = _assert_board_size(b)

    for t in range(num_actions):
        cur_seed = rng.randrange(1 << 30)
        if mode is None:
            # include duplicate modes in random choice
            mode_choice = rng.choice(
                list(INVALID_MODES.keys()) + ["duplicate_row", "duplicate_col"]
            )
        else:
            mode_choice = mode

        if mode_choice in INVALID_MODES:
            b = INVALID_MODES[mode_choice](b, seed=cur_seed)
        elif mode_choice == "duplicate_row":
            r = rng.randrange(N)
            b = duplicate_in_row(b, r=r, seed=cur_seed)
        elif mode_choice == "duplicate_col":
            c = rng.randrange(N)
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
        board: (N,N) numpy array with digits 1..N (complete/valid solution assumed).
        num_clues: target number of visible cells to keep (defaults to 30). If None,
            a random number in [n*n+1, int(0.75*N*N)] is sampled.
        ensure_one_per_row: if True, ensures at least one clue remains in every row.
        ensure_one_per_col: if True, ensures at least one clue remains in every column.
        ensure_one_per_block: if True, ensures at least one clue remains in every n x n block.
        seed: optional RNG seed for reproducibility.
        blank_value: value to place in removed cells (0 by default).

    Returns:
        puzzle: (N,N) numpy array with some cells set to `blank_value`.

    Notes:
        - This function does *not* check extendability/uniqueness; it only masks cells.
        - The "ensure_*" options may force keeping more clues than requested if necessary.
    """
    N, n = _assert_board_size(board)
    rng = np.random.default_rng(seed)

    if num_clues is None:
        low = n * n + 1  # just above a typical minimal-feasible region
        high = max(low + 1, int(0.75 * N * N))
        num_clues = int(rng.integers(low=low, high=high))
    num_clues = int(np.clip(num_clues, 0, N * N))

    keep: set[tuple[int, int]] = set()

    if ensure_one_per_row:
        for r in range(N):
            c = int(rng.integers(0, N))
            keep.add((r, c))

    if ensure_one_per_col:
        for c in range(N):
            r = int(rng.integers(0, N))
            keep.add((r, c))

    if ensure_one_per_block:
        for br in range(n):
            for bc in range(n):
                r = int(rng.integers(br * n, br * n + n))
                c = int(rng.integers(bc * n, bc * n + n))
                keep.add((r, c))

    min_required = len(keep)
    if num_clues < min_required:
        num_clues = min_required

    all_idx = [(r, c) for r in range(N) for c in range(N) if (r, c) not in keep]
    remaining_to_keep = num_clues - len(keep)
    if remaining_to_keep > 0 and len(all_idx) > 0:
        chosen = rng.choice(
            len(all_idx), size=min(remaining_to_keep, len(all_idx)), replace=False
        )
        for i in np.atleast_1d(chosen):
            keep.add(all_idx[int(i)])

    puzzle = np.full_like(board, fill_value=blank_value)
    for r, c in keep:
        puzzle[r, c] = board[r, c]

    return puzzle


# --------------------------
# Concept extraction
# --------------------------
def get_concepts(board, return_label=False):
    concepts = {}
    N, n = _assert_board_size(board)

    row_valid = []
    for i in range(N):
        unique = np.unique(board[i, :])
        row_valid.append(int(len(unique) == N))
    concepts["row_valid"] = row_valid

    col_valid = []
    for j in range(N):
        unique = np.unique(board[:, j])
        col_valid.append(int(len(unique) == N))
    concepts["col_valid"] = col_valid

    block_valid = []
    for bi in range(n):
        for bj in range(n):
            block = board[bi * n : (bi + 1) * n, bj * n : (bj + 1) * n].reshape(-1)
            unique = np.unique(block)
            block_valid.append(int(len(unique) == N))
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

    Concepts returned (all lists of length N):
      - row_consistent / col_consistent / block_consistent: 1 if the unit has
        no duplicate non-blank digits (ignores blanks), else 0.
      - row_complete / col_complete / block_complete: 1 if the unit has no blanks
        (i.e., N filled cells), else 0.

    If `return_board_flags` is True, also returns:
      - board_consistent: AND of all unit-consistent flags (3N units)
      - board_complete: AND of all unit-complete flags (3N units)

    Notes:
      - This does not check global Sudoku validity for completed units; it
        focuses on duplicate-free (consistency) and filled-ness (completeness).
    """
    N, n = _assert_board_size(board)

    def unit_consistent(vals):
        vals = np.asarray(vals)
        mask = vals != blank_value
        nz = vals[mask]
        return int(len(nz) == len(np.unique(nz)))

    def unit_complete(vals):
        vals = np.asarray(vals)
        return int(np.count_nonzero(vals != blank_value) == N)

    row_consistent = [unit_consistent(board[r, :]) for r in range(N)]
    row_complete = [unit_complete(board[r, :]) for r in range(N)]

    col_consistent = [unit_consistent(board[:, c]) for c in range(N)]
    col_complete = [unit_complete(board[:, c]) for c in range(N)]

    block_consistent = []
    block_complete = []
    for br in range(n):
        for bc in range(n):
            blk = board[br * n : (br + 1) * n, bc * n : (bc + 1) * n].reshape(-1)
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
        all_complete = all(row_complete) and all(col_complete) and all(block_complete)
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
        if not include_board_flags and k in {
            "board_valid",
            "board_consistent",
            "board_complete",
        }:
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


def add_noise_to_concept_vector(
    vec: np.ndarray, eta: float = 0.1, seed: int | None = None
) -> np.ndarray:
    """
    Flip bits of a 0/1 numpy vector with probability η (independent Bernoulli noise).
    Returns a new array with same shape and dtype int32.
    """
    if vec.ndim == 0:
        return vec
    rng = np.random.default_rng(seed)
    mask = _bernoulli_mask(vec.shape, eta, rng)
    return np.logical_xor(vec.astype(bool), mask).astype(np.int32)


# --------------------------
# Cell and value position concept helpers
# --------------------------


def normalize_positions(
    N: int, positions_subset: Sequence[tuple[int, int]] | None
) -> list[tuple[int, int]]:
    """
    Validate and canonicalize a list of cell positions.

    Args:
        N (int): Board dimension (e.g., 9 for a 9×9 board). Must be >= 1.
        positions_subset (Sequence[tuple[int, int]] | None): Optional sequence
            of 0-indexed (row, col) pairs to include. If ``None``, all cells
            ``[(0,0), (0,1), …, (N-1,N-1)]`` are used.

    Returns:
        list[tuple[int, int]]: A row-major-sorted list of valid (row, col) pairs,
        each satisfying ``0 <= row < N`` and ``0 <= col < N``.

    Raises:
        ValueError: If any provided (row, col) lies outside ``[0, N)`` for either
        coordinate.
    """
    if positions_subset is None:
        return [(r, c) for r in range(N) for c in range(N)]
    out = []
    for r, c in positions_subset:
        if not (0 <= r < N and 0 <= c < N):
            raise ValueError(f"Cell {(r, c)} out of bounds for N={N}.")
        out.append((r, c))
    # stable order: row-major
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def normalize_digits(N: int, digits_subset: Sequence[int] | None) -> list[int]:
    """
    Validate and canonicalize a list of digits.

    Args:
        N (int): Board dimension (e.g., 9 for a 9×9 board). Valid digits are
            integers in ``[1, N]``.
        digits_subset (Sequence[int] | None): Optional sequence of digits to
            include. If ``None``, defaults to ``[1, 2, …, N]``.

    Returns:
        list[int]: A sorted (ascending) list of valid digits in ``[1, N]``.

    Raises:
        ValueError: If any provided digit is not an integer in ``[1, N]``.
    """
    if digits_subset is None:
        return list(range(1, N + 1))
    out = []
    for d in digits_subset:
        if not (1 <= int(d) <= N):
            raise ValueError(f"Digit {d} out of range 1..{N}.")
        out.append(int(d))
    out.sort()
    return out


def cell_digit_concept_vector(
    board: np.ndarray, positions: Sequence[tuple[int, int]], digits: Sequence[int]
) -> np.ndarray:
    """
    Build a binary vector encoding per-cell digit indicators.

    For each ``(row, col)`` in ``positions`` (row-major order), and for each
    digit ``d`` in ``digits`` (ascending), emits a 1 iff ``board[row, col] == d``,
    else 0. The final vector concatenates these indicators in the same nested
    order.

    Args:
        board (np.ndarray): An ``N×N`` array of cell values. Expected to contain
            integers in ``{0, 1, …, N}`` where 0 (or non-positive / None) means
            “blank”. Only exact equality to digits in ``digits`` yields 1s.
        positions (Sequence[tuple[int, int]]): 0-indexed cell coordinates
            ``(row, col)``. Typically produced by ``_normalize_positions`` for
            validation and row-major ordering.
        digits (Sequence[int]): Digits to test for each position. Typically
            produced by ``_normalize_digits`` to ensure values lie in ``[1, N]``.

    Returns:
        np.ndarray: A 1D array of dtype ``np.int32`` and length
        ``len(positions) * len(digits)`` with entries in ``{0, 1}``.
    """
    vals = []
    for r, c in positions:
        v = int(board[r, c]) if board[r, c] is not None else 0
        for d in digits:
            vals.append(1 if v == d else 0)
    return np.asarray(vals, dtype=np.int32)
