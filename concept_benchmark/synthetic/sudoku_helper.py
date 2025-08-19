import numpy as np
import random

# --------------------------
# Base valid board
# --------------------------
BASE_BOARD = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 6, 7, 8, 9, 1, 2, 3],
    [7, 8, 9, 1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7, 8, 9, 1],
    [5, 6, 7, 8, 9, 1, 2, 3, 4],
    [8, 9, 1, 2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8, 9, 1, 2],
    [6, 7, 8, 9, 1, 2, 3, 4, 5],
    [9, 1, 2, 3, 4, 5, 6, 7, 8]
])

# --------------------------
# Transformations to shuffle valid boards
# Used to generate valid Sudoku boards
# --------------------------
def shuffle_rows(board):
    for band in range(0, 9, 3):
        rows = list(range(band, band+3))
        np.random.shuffle(rows)
        board[band:band+3] = board[rows]
    return board

def shuffle_columns(board):
    for stack in range(0, 9, 3):
        cols = list(range(stack, stack+3))
        np.random.shuffle(cols)
        board[:, stack:stack+3] = board[:, cols]
    return board

def shuffle_row_bands(board):
    bands = [0, 1, 2]
    np.random.shuffle(bands)
    return np.vstack([board[i*3:(i+1)*3] for i in bands])

def shuffle_col_stacks(board):
    stacks = [0, 1, 2]
    np.random.shuffle(stacks)
    return np.hstack([board[:, i*3:(i+1)*3] for i in stacks])

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
# Corrupting valid boards
# --------------------------
def make_row_invalid(board, r):
    b = board.copy()
    c1, c2 = random.sample(range(9), 2)
    b[r, c2] = b[r, c1]
    return b

def make_col_invalid(board, c):
    b = board.copy()
    r1, r2 = random.sample(range(9), 2)
    b[r2, c] = b[r1, c]
    return b

def make_block_invalid(board, br, bc):
    b = board.copy()
    cells = [(i, j) for i in range(br*3, (br+1)*3) for j in range(bc*3, (bc+1)*3)]
    (r1, c1), (r2, c2) = random.sample(cells, 2)
    b[r2, c2] = b[r1, c1]
    return b

def make_invalid(board):
    choice = random.choice(["row", "col", "block"])
    if choice == "row":
        return make_row_invalid(board, random.randrange(9))
    if choice == "col":
        return make_col_invalid(board, random.randrange(9))
    br, bc = random.randrange(3), random.randrange(3)
    return make_block_invalid(board, br, bc)

def generate_invalid_board(base_board=None, num_changes=1):
    if base_board is None:
        base_board = generate_valid_board()
    for _ in range(num_changes):
        base_board = make_invalid(base_board)
    return base_board

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
            block = board[bi*3:(bi+1)*3, bj*3:(bj+1)*3].flatten()
            unique = np.unique(block)
            block_valid.append(int(len(unique) == 9))
    concepts["block_valid"] = block_valid

    if return_label:
        board_valid = int(all(row_valid) and all(col_valid) and all(block_valid))
        concepts["board_valid"] = board_valid

    return concepts