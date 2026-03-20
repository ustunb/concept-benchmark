"""Sudoku validation benchmark pipeline.

Provides functions to run each stage of the sudoku benchmark programmatically.

Usage:
    python scripts/sudoku_pipeline.py --seed 171
    python scripts/sudoku_pipeline.py --seed 171 --stages cs dnn selective
    python scripts/sudoku_pipeline.py --config my_config.yaml
"""
from __future__ import annotations

from concept_benchmark.utils import parse_budgets
from concept_benchmark.config import SudokuBenchmarkConfig

# Re-export public API so existing imports like
#   from scripts.sudoku_pipeline import train_cs, run, ...
# continue to work.
from scripts.sudoku.training import (  # noqa: F401
    setup_dataset,
    train_ocr,
    train_cs,
    train_dnn,
)
from scripts.sudoku.selective import (  # noqa: F401
    _selective_accuracy_threshold,
    _decision_threshold_sweep,
    _selective_metrics,
    _cs_val_probs,
    _dnn_val_probs,
    compute_selective_results,
)
from scripts.sudoku.stages import (  # noqa: F401
    run_interventions,
    align,
    run,
)
from scripts.sudoku.collect import (  # noqa: F401
    _dataset_label,
    collect_results,
)


# ── CLI entry point ──────────────────────────────────────────────────

SUDOKU_STAGES = ("setup", "ocr", "cs", "dnn", "intervene", "selective", "align", "collect")


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the sudoku validation benchmark pipeline.",
    )
    parser.add_argument("--seed", type=int, default=171)
    parser.add_argument(
        "--stages", nargs="+", default=list(SUDOKU_STAGES),
        help=f"Pipeline stages to run (default: all). Valid: {' -> '.join(SUDOKU_STAGES)}",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--budgets", nargs="+", default=None,
                        help="Intervention budgets (e.g. 1 3 5 max).")
    parser.add_argument("--data-type", type=str, default=None,
                        choices=["tabular", "image"])
    hw_group = parser.add_mutually_exclusive_group()
    hw_group.add_argument("--handwriting", action="store_true", default=None)
    hw_group.add_argument("--no-handwriting", action="store_true")
    parser.add_argument("--force-setup", action="store_true")
    return parser.parse_args(argv)




def main(argv=None):
    args = _parse_args(argv)

    unknown = set(args.stages) - set(SUDOKU_STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}. Valid: {list(SUDOKU_STAGES)}")

    if args.config:
        config = SudokuBenchmarkConfig.from_yaml(args.config)
    else:
        config = SudokuBenchmarkConfig(seed=args.seed)

    if args.budgets:
        config.intervention_budgets = parse_budgets(args.budgets)
    if args.no_handwriting:
        config.font_style = "printed"
    elif args.handwriting:
        config.font_style = "handwritten"
    if args.data_type is not None:
        config.data_type = args.data_type

    run(config, stages=args.stages, force_setup=args.force_setup)


if __name__ == "__main__":
    main()
