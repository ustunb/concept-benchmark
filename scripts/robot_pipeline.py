"""Robot classification benchmark pipeline.

Provides functions to run each stage of the robot benchmark programmatically.

Usage:
    python scripts/robot_pipeline.py --seed 1014
    python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes
    python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --regimes baseline expert
    python scripts/robot_pipeline.py --config my_config.yaml
"""
from __future__ import annotations

from concept_benchmark.utils import parse_budgets
from concept_benchmark.config import RobotBenchmarkConfig

# Re-export public API so existing imports like
#   from scripts.robot_pipeline import train_cbm, run, ...
# continue to work.
from scripts.robot.training import (  # noqa: F401
    FEOnProbs,
    setup_dataset,
    train_cbm,
    train_cbm_subjective,
    train_dnn,
    train_lfcbm,
)
from scripts.robot.regimes import (  # noqa: F401
    InterventionSettings,
    _run_llm_regime,
    _run_regime,
    _test_interventions,
)
from scripts.robot.stages import (  # noqa: F401
    align,
    run,
    run_interventions,
)
from scripts.robot.collect import (  # noqa: F401
    collect_results,
)


# ── CLI entry point ──────────────────────────────────────────────────

ROBOT_STAGES = ("setup", "cbm", "dnn", "intervene", "align", "collect")


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the robot classification benchmark pipeline.",
    )
    parser.add_argument("--seed", type=int, default=1014)
    parser.add_argument(
        "--stages", nargs="+", default=list(ROBOT_STAGES),
        help=f"Pipeline stages to run (default: all). Valid: {' -> '.join(ROBOT_STAGES)}",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--concept-preset", choices=["ground_truth", "foot_subtypes"], default="ground_truth")
    parser.add_argument("--missing-fraction", type=float, default=None,
                        help="Fraction of concept labels to mask (e.g. 0.2).")
    parser.add_argument("--missing-mechanism", type=str, default=None,
                        choices=["mcar", "mnar"])
    parser.add_argument("--budgets", nargs="+", default=None,
                        help="Intervention budgets (e.g. 1 3 5 max).")
    parser.add_argument("--regimes", nargs="+", default=None,
                        help="Intervention regimes (e.g. baseline expert subjective machine).")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["up_to_k", "exactly_k"])
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--force-retrain", action="store_true", dest="force_retrain")
    parser.add_argument("--force-setup", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    unknown = set(args.stages) - set(ROBOT_STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}. Valid: {list(ROBOT_STAGES)}")

    if args.config:
        config = RobotBenchmarkConfig.from_yaml(args.config)
    elif args.concept_preset == "foot_subtypes":
        config = RobotBenchmarkConfig.default_subconcept()
        config.seed = args.seed
    else:
        config = RobotBenchmarkConfig(seed=args.seed)

    if args.budgets:
        config.intervention_budgets = parse_budgets(args.budgets)
    if args.regimes:
        config.intervention_regimes = args.regimes
    if args.strategy:
        config.intervention_strategy = args.strategy
    if args.llm_api_key:
        config.llm_api_key = args.llm_api_key
    if args.force_retrain:
        config.force_retrain = True
    missing_fraction = args.missing_fraction or 0.0
    missing_mechanism = args.missing_mechanism or "mcar"

    run(config, stages=args.stages, force_setup=args.force_setup,
        missing_fraction=missing_fraction, missing_mechanism=missing_mechanism)


if __name__ == "__main__":
    main()
