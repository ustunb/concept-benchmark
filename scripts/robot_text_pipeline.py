"""Robot text classification benchmark pipeline.

Provides functions to run each stage of the robot text benchmark
programmatically, following the same pattern as robot_pipeline.py and
sudoku_pipeline.py.

Usage:
    python scripts/robot_text_pipeline.py --seed 1337
    python scripts/robot_text_pipeline.py --regimes baseline expert
    python scripts/robot_text_pipeline.py --config my_config.yaml
"""
from __future__ import annotations

from concept_benchmark.utils import parse_budgets
from concept_benchmark.config import RobotBenchmarkConfig

# Re-export public API so existing imports like
#   from scripts.robot_text_pipeline import train_cbm, run, ...
# continue to work.
from scripts.robot_text.training import (  # noqa: F401
    _internal_output_mode,
    _TextDS,
    _train_dnn_text,
    _fit_platt,
    setup_dataset,
    train_cbm,
    train_cbm_subjective,
    train_dnn,
    train_lfcbm,
)
from scripts.robot_text.regimes import (  # noqa: F401
    _run_text_regime,
    _ensure_intervention_imports,
)
from scripts.robot_text.stages import (  # noqa: F401
    run_interventions,
    align,
    run,
)
from scripts.robot_text.collect import (  # noqa: F401
    collect_results,
)


# ── CLI entry point ──────────────────────────────────────────────────

ROBOT_TEXT_STAGES = ("setup", "cbm", "dnn", "lfcbm", "intervene", "align", "collect")


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the robot text classification benchmark pipeline.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(ROBOT_TEXT_STAGES),
        help=f"Pipeline stages to run (default: all). Valid: {' -> '.join(ROBOT_TEXT_STAGES)}",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help="Intervention budgets (e.g. 1 2 5 max).",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="Intervention regimes (e.g. baseline expert subjective).",
    )
    parser.add_argument(
        "--strategy", type=str, default=None, choices=["up_to_k", "exactly_k"]
    )
    parser.add_argument(
        "--concept-preset",
        type=str,
        default=None,
        choices=["ground_truth", "foot_subtypes"],
        help="Concept granularity preset (default: ground_truth).",
    )
    parser.add_argument(
        "--missing-fraction",
        type=float,
        default=None,
        help="Fraction of concept labels to mask in training set.",
    )
    parser.add_argument(
        "--missing-mechanism",
        type=str,
        default=None,
        choices=["mcar", "mnar"],
        help="Missingness mechanism (default: mcar).",
    )
    parser.add_argument("--lfcbm", action="store_true", help="Also run LFCBM variant.")
    parser.add_argument("--force-setup", action="store_true")
    return parser.parse_args(argv)



def main(argv=None):
    args = _parse_args(argv)

    unknown = set(args.stages) - set(ROBOT_TEXT_STAGES)
    if unknown:
        raise ValueError(
            f"unknown stages: {sorted(unknown)}. Valid: {list(ROBOT_TEXT_STAGES)}"
        )

    if args.config:
        config = RobotBenchmarkConfig.from_yaml(args.config)
    else:
        init_kwargs = {"data_type": "text", "seed": args.seed}
        if args.concept_preset:
            init_kwargs["concept_preset"] = args.concept_preset
        config = RobotBenchmarkConfig(**init_kwargs)

    if args.budgets:
        config.intervention_budgets = parse_budgets(args.budgets)
    if args.regimes:
        config.intervention_regimes = args.regimes
    if args.strategy:
        config.intervention_strategy = args.strategy
    if args.lfcbm:
        config.use_label_free_concepts = True
        if "lfcbm" not in args.stages:
            args.stages.append("lfcbm")

    run(config, stages=args.stages, force_setup=args.force_setup)


if __name__ == "__main__":
    main()
