"""Robot demo pipeline — thin wrapper around concept_benchmark.benchmarks.robot.

Preserves the original CLI interface for backward compatibility.
"""
from argparse import ArgumentParser
import sys

from concept_benchmark.benchmarks.robot import run
from concept_benchmark.config import RobotBenchmarkConfig


def main():
    defaults = RobotBenchmarkConfig.default_ideal()

    p = ArgumentParser(description="Run the robot experimental pipeline.")
    p.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "cbm", "dnn", "intervene"],
    )
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--draw", action="store_true", help="draw robots at setup")
    p.add_argument("--missing", action="store_true", help="use concept missingness")
    p.add_argument("--ignore-errors", action="store_true", help="continue on errors")
    args = p.parse_args()
    args.missing = True

    config = RobotBenchmarkConfig(seed=args.seed)
    try:
        run(config, stages=args.stages, missing=args.missing)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        if not args.ignore_errors:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
