"""Tests for CLI argument parsing and verbosity flags."""
from __future__ import annotations

import pytest
from concept_benchmark.cli import main


def test_cli_help(capsys):
    """--help exits cleanly and prints usage."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "cbm-benchmark" in out


def test_cli_verbose_flag():
    """--verbose is accepted without error (parsing only)."""
    with pytest.raises(SystemExit):
        # Will fail because no subcommand, but flag parsing should work
        main(["--verbose"])


def test_cli_quiet_flag():
    """--quiet is accepted without error (parsing only)."""
    with pytest.raises(SystemExit):
        main(["--quiet"])


def test_cli_verbose_quiet_mutual_exclusion(capsys):
    """--verbose and --quiet cannot be used together."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--verbose", "--quiet", "robot"])
    assert exc_info.value.code != 0


def test_cli_robot_subcommand_help(capsys):
    """robot --help exits cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        main(["robot", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--seed" in out
    assert "--subconcept" in out


def test_cli_sudoku_subcommand_help(capsys):
    """sudoku --help exits cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        main(["sudoku", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--seed" in out


def test_cli_dry_run_robot(capsys):
    """--dry-run prints config without executing."""
    main(["--dry-run", "robot", "--seed", "42"])
    out = capsys.readouterr().out
    assert "benchmark: robot" in out
    assert "seed:      42" in out
    assert "stages:" in out


def test_cli_dry_run_sudoku(capsys):
    """--dry-run works for sudoku subcommand."""
    main(["--dry-run", "sudoku"])
    out = capsys.readouterr().out
    assert "benchmark: sudoku" in out
