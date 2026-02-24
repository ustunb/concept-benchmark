"""CLI entry point for running concept benchmarks.

Usage:
    cbm-benchmark robot --seed 1014 --stages setup cbm dnn intervene
    cbm-benchmark robot --config my_config.yaml
    cbm-benchmark sudoku --seed 171 --stages setup ocr cs dnn intervene
    cbm-benchmark robot-text --seed 1337 --stages setup cbm dnn intervene
"""
from __future__ import annotations

import argparse
import logging
import sys


_ROBOT_STAGES = {"setup", "cbm", "dnn", "intervene", "align", "collect"}
_SUDOKU_STAGES = {"setup", "ocr", "cs", "dnn", "intervene", "selective", "align", "collect"}
_ROBOT_TEXT_STAGES = {"setup", "cbm", "dnn", "lfcbm", "intervene", "align", "collect"}


def _validate_stages(stages: list[str], valid: set[str], benchmark: str) -> None:
    unknown = set(stages) - valid
    if unknown:
        raise ValueError(
            f"unknown stages for {benchmark}: {sorted(unknown)}. "
            f"Valid: {sorted(valid)}"
        )


def _robot_cmd(args: argparse.Namespace) -> None:
    from concept_benchmark.benchmarks.robot import run
    from concept_benchmark.config import RobotBenchmarkConfig

    _validate_stages(args.stages, _ROBOT_STAGES, "robot")

    if args.config:
        config = RobotBenchmarkConfig.from_yaml(args.config)
    elif args.subconcept:
        config = RobotBenchmarkConfig.default_subconcept()
        config.seed = args.seed
    else:
        config = RobotBenchmarkConfig(seed=args.seed)

    if args.regimes:
        config.intervention_regimes = args.regimes
    if args.strategy:
        config.intervention_strategy = args.strategy
    if getattr(args, "llm_api_key", None):
        config.llm_api_key = args.llm_api_key
    if getattr(args, "force_retrain", False):
        config.force_retrain = True

    if args.concept_missing is not None:
        config.concept_missing = args.concept_missing
        config.concept_missing_mech = args.concept_missing_mech or "mcar"
    elif args.concept_missing_mech is not None:
        config.concept_missing_mech = args.concept_missing_mech

    run(config, stages=args.stages)


def _sudoku_cmd(args: argparse.Namespace) -> None:
    from concept_benchmark.benchmarks.sudoku import run
    from concept_benchmark.config import SudokuBenchmarkConfig

    _validate_stages(args.stages, _SUDOKU_STAGES, "sudoku")
    if args.config:
        config = SudokuBenchmarkConfig.from_yaml(args.config)
    else:
        config = SudokuBenchmarkConfig(seed=args.seed)

    run(config, stages=args.stages)


def _robot_text_cmd(args: argparse.Namespace) -> None:
    from concept_benchmark.benchmarks.robot_text import run
    from concept_benchmark.config import RobotTextBenchmarkConfig

    _validate_stages(args.stages, _ROBOT_TEXT_STAGES, "robot-text")
    if args.config:
        config = RobotTextBenchmarkConfig.from_yaml(args.config)
    else:
        config = RobotTextBenchmarkConfig(seed=args.seed)
    if args.lfcbm:
        config.lfcbm_enable = True
        if "lfcbm" not in args.stages:
            args.stages.append("lfcbm")
    if args.regimes:
        config.intervention_regimes = args.regimes
    if args.strategy:
        config.intervention_strategy = args.strategy

    run(config, stages=args.stages)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cbm-benchmark",
        description="Run concept benchmark experiments.",
    )
    # Global verbosity flags
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show debug-level output.",
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true",
        help="Only show warnings and errors.",
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
    robot_p.add_argument("--concept-missing", type=float, default=None,
                         help="Fraction of concept labels to mask (e.g. 0.2).")
    robot_p.add_argument("--concept-missing-mech", type=str, default=None,
                         choices=["none", "mcar", "mnar"],
                         help="Missingness mechanism (default: mcar if --concept-missing is set).")
    robot_p.add_argument("--regimes", nargs="+", default=None,
                         help="Intervention regimes (e.g. baseline expert subjective machine).")
    robot_p.add_argument("--strategy", type=str, default=None,
                         choices=["kflip", "exact_k"],
                         help="Intervention strategy: kflip (up-to-k) or exact_k.")
    robot_p.add_argument("--llm-api-key", type=str, default=None,
                         help="API key for LLM provider (alternative to env var).")
    robot_p.add_argument("--force-retrain", action="store_true",
                         help="Force retrain LFCBM/subjective models even if cached.")
    robot_p.set_defaults(func=_robot_cmd)

    # Sudoku subcommand
    sudoku_p = subparsers.add_parser("sudoku", help="Run the sudoku validation benchmark.")
    sudoku_p.add_argument("--seed", type=int, default=171)
    sudoku_p.add_argument(
        "--stages", nargs="+", default=["setup", "ocr", "cs", "dnn", "intervene", "selective", "align", "collect"],
    )
    sudoku_p.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    sudoku_p.set_defaults(func=_sudoku_cmd)

    # Robot-text subcommand
    robot_text_p = subparsers.add_parser("robot-text", help="Run the robot text classification benchmark.")
    robot_text_p.add_argument("--seed", type=int, default=1337)
    robot_text_p.add_argument(
        "--stages", nargs="+", default=["setup", "cbm", "dnn", "intervene", "align", "collect"],
    )
    robot_text_p.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    robot_text_p.add_argument("--lfcbm", action="store_true", help="Also run LFCBM variant.")
    robot_text_p.add_argument("--regimes", nargs="+", default=None,
                              help="Intervention regimes (e.g. baseline expert subjective machine).")
    robot_text_p.add_argument("--strategy", type=str, default=None,
                              choices=["kflip", "exact_k"],
                              help="Intervention strategy: kflip (up-to-k) or exact_k.")
    robot_text_p.set_defaults(func=_robot_text_cmd)

    # Global --dry-run flag
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and stages without executing.",
    )

    args = parser.parse_args(argv)

    # Configure logging based on verbosity flags
    from concept_benchmark._logging import setup_logging
    if args.verbose:
        setup_logging(level=logging.DEBUG, verbose_format=True)
    elif args.quiet:
        setup_logging(level=logging.WARNING)
    else:
        setup_logging(level=logging.INFO)

    if args.dry_run:
        print(f"benchmark: {args.benchmark}")
        print(f"seed:      {getattr(args, 'seed', 'N/A')}")
        print(f"stages:    {getattr(args, 'stages', 'N/A')}")
        if hasattr(args, "subconcept"):
            print(f"subconcept: {args.subconcept}")
        if hasattr(args, "regimes") and args.regimes:
            print(f"regimes:   {args.regimes}")
        if hasattr(args, "strategy") and args.strategy:
            print(f"strategy:  {args.strategy}")
        return

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
