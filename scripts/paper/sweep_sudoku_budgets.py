"""Score a trained sudoku CBM at every intervention budget, on the validation and the test split (the budget sweep).

Loads the recognizer-inferred dataset and the CBM that `scripts/sudoku_pipeline.py` wrote for a seed and resolution,
runs `automation_table` at the budgets 1, 2, 3, 5, 8, 13 and max on both splits, adds the net work automated (coverage
minus the checks as a share of all concepts) and writes one CSV per split under the names `plot_sudoku_ksweep.py` reads:

    python scripts/paper/sweep_sudoku_budgets.py --seed 171 --cell-px 50 --tau 0.99 --out results/paper/sudoku/ksweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401  (puts the repository root on the import path)
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.ext.fileutils import load
from experiments.evaluate import automation_table

BUDGETS = (1, 2, 3, 5, 8, 13, "max")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cell-px", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    config = SudokuBenchmarkConfig(seed=args.seed)
    config.cell_px = args.cell_px
    data = load(
        config.get_dataset_path(data_type="image") / "ocr_inferred_full_dataset.pkl"
    )
    data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=args.seed)
    model = load(config.get_model_path("cs", data_type="tabular"))
    model._random_state = args.seed

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in (("validation", data.validation), ("test", data.test)):
        table = automation_table(
            model,
            data.validation,
            split,
            budgets=BUDGETS,
            target_accuracy=args.tau,
            seed=args.seed,
            concept_groups=config.block_size**2,
        )
        table["net_work"] = table["coverage_after"] - table["total_concept_checks"] / (
            split.n * split.n_concepts
        )
        out = (
            args.out
            / f"sudoku__arch-cbm__res-{args.cell_px}px__tau-{args.tau:.2f}__{name}__seed-{args.seed}__ksweep.csv"
        )
        table.to_csv(out, index=False)
        print(name, out)


if __name__ == "__main__":
    main()
