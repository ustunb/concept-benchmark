"""Sudoku demo pipeline — thin wrapper around concept_benchmark.benchmarks.sudoku.

Preserves the original CLI interface for backward compatibility.
"""
from argparse import ArgumentParser
import sys

from concept_benchmark.benchmarks.sudoku import run
from concept_benchmark.config import SudokuBenchmarkConfig


def main():
    defaults = SudokuBenchmarkConfig.default()

    p = ArgumentParser(description="Run the sudoku experimental pipeline.")
    p.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "ocr", "cs", "dnn", "intervene"],
    )
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--ignore-errors", action="store_true", help="continue on errors")
    args = p.parse_args()

    config = SudokuBenchmarkConfig(seed=args.seed)
    try:
        run(config, stages=args.stages)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        if not args.ignore_errors:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
