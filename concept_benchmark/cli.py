"""CLI entry point for running concept benchmarks.

Usage:
    cbm-benchmark robot --seed 1014 --stages setup cbm dnn intervene
    cbm-benchmark robot --config my_config.yaml
    cbm-benchmark sudoku --seed 171 --stages setup ocr cs dnn intervene
"""
from __future__ import annotations

import argparse
import sys


def _robot_cmd(args: argparse.Namespace) -> None:
    from concept_benchmark.benchmarks.robot import run
    from concept_benchmark.config import RobotBenchmarkConfig

    if args.config:
        config = RobotBenchmarkConfig.from_yaml(args.config)
    else:
        config = RobotBenchmarkConfig(seed=args.seed)
        if args.subconcept:
            config = RobotBenchmarkConfig.default_subconcept()
            config.seed = args.seed

    run(config, stages=args.stages, missing=args.missing)


def _sudoku_cmd(args: argparse.Namespace) -> None:
    from concept_benchmark.benchmarks.sudoku import run
    from concept_benchmark.config import SudokuBenchmarkConfig

    if args.config:
        config = SudokuBenchmarkConfig.from_yaml(args.config)
    else:
        config = SudokuBenchmarkConfig(seed=args.seed)

    run(config, stages=args.stages)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cbm-benchmark",
        description="Run concept benchmark experiments.",
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    # Robot subcommand
    robot_p = subparsers.add_parser("robot", help="Run the robot classification benchmark.")
    robot_p.add_argument("--seed", type=int, default=1014)
    robot_p.add_argument(
        "--stages", nargs="+", default=["setup", "cbm", "dnn", "intervene", "align", "collect"],
    )
    robot_p.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    robot_p.add_argument("--subconcept", action="store_true")
    robot_p.add_argument("--missing", action="store_true", default=True,
                         help="Run MCAR/MNAR missingness variants.")
    robot_p.add_argument("--no-missing", dest="missing", action="store_false")
    robot_p.set_defaults(func=_robot_cmd)

    # Sudoku subcommand
    sudoku_p = subparsers.add_parser("sudoku", help="Run the sudoku validation benchmark.")
    sudoku_p.add_argument("--seed", type=int, default=171)
    sudoku_p.add_argument(
        "--stages", nargs="+", default=["setup", "ocr", "cs", "dnn", "intervene", "selective", "align", "collect"],
    )
    sudoku_p.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    sudoku_p.set_defaults(func=_sudoku_cmd)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
