
import pytest
import numpy as np
from concept_benchmark.synthetic.helper.sudoku_helper import (
    solve_sudoku_with_random_objective,
    generate_valid_board_scip,
    Model
)

@pytest.mark.skipif(Model is None, reason="SCIP (pyscipopt) not installed")
def test_solve_sudoku_with_random_objective_validity():
    """Test that the solver produces a valid sudoku board."""
    n = 3
    N = n * n
    board = solve_sudoku_with_random_objective(n=n)
    
    assert board.shape == (N, N)
    
    # Check rows
    for r in range(N):
        assert len(np.unique(board[r, :])) == N
        
    # Check cols
    for c in range(N):
        assert len(np.unique(board[:, c])) == N
        
    # Check blocks
    for br in range(n):
        for bc in range(n):
            block = board[br*n:(br+1)*n, bc*n:(bc+1)*n].reshape(-1)
            assert len(np.unique(block)) == N

@pytest.mark.skipif(Model is None, reason="SCIP (pyscipopt) not installed")
def test_solve_sudoku_respects_size():
    """Test that the solver respects the requested size."""
    n = 2
    N = n * n
    board = solve_sudoku_with_random_objective(n=n)
    assert board.shape == (N, N)
    assert np.max(board) <= N
    assert np.min(board) >= 1

@pytest.mark.skipif(Model is None, reason="SCIP (pyscipopt) not installed")
def test_solve_sudoku_randomness():
    """Test that different seeds produce different boards."""
    n = 3
    board1 = solve_sudoku_with_random_objective(n=n, seed=42)
    board2 = solve_sudoku_with_random_objective(n=n, seed=43)
    
    # It is extremely unlikely that two random 9x9 sudoku boards are identical
    assert not np.array_equal(board1, board2)

@pytest.mark.skipif(Model is None, reason="SCIP (pyscipopt) not installed")
def test_solve_sudoku_reproducibility():
    """Test that the same seed produces the same board."""
    n = 3
    board1 = solve_sudoku_with_random_objective(n=n, seed=123)
    board2 = solve_sudoku_with_random_objective(n=n, seed=123)
    
    np.testing.assert_array_equal(board1, board2)

@pytest.mark.skipif(Model is None, reason="SCIP (pyscipopt) not installed")
def test_generate_valid_board_scip_integration():
    """Test the higher-level generator which adds relabeling."""
    n = 3
    N = n * n
    board = generate_valid_board_scip(n=n, seed=555)
    
    assert board.shape == (N, N)
    # Check validity again roughly (helpers check structure)
    for r in range(N):
        assert len(np.unique(board[r, :])) == N
        
    # Check reproducibility of the wrapper
    board2 = generate_valid_board_scip(n=n, seed=555)
    np.testing.assert_array_equal(board, board2)
    
    # Check diversity with different seed
    board3 = generate_valid_board_scip(n=n, seed=556)
    assert not np.array_equal(board, board3)
