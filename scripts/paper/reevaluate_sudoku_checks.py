"""Re-score the stored sudoku models with the current concept-check policy, without retraining.

For one architecture, resolution and seed, loads the trained model from `--models` and the recognizer-inferred dataset
from `--data`, splits it as `scripts/sudoku_pipeline.py` does, and runs `automation_table` at every selective-accuracy
target and cost model. Writes the cells under the names the paper scripts read (`--out`), and with `--sweep` also the
budget sweep (budgets 1, 2, 3, 5, 8, 13, max on the validation and the test split, the CBM only):

    python scripts/paper/reevaluate_sudoku_checks.py --arch cbm --cell-px 50 --seed 171 \
        --models results/paper/sudoku/models --data results/paper/sudoku/data/sudoku --out results/paper/sudoku/cells_gain
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import _common  # noqa: F401  (puts the repository root on the import path)
from concept_benchmark.ext.fileutils import load
from experiments.evaluate import automation_table

TAUS = ("0.90", "0.925", "0.95", "0.975", "0.99")
_draw = random.Random(0)
COSTS = {  # cost of a row, column and block check; the same models as make_sudoku_sensitivity_tables.py
    "equal": (1.0, 1.0, 1.0),
    "2xblock": (1.0, 1.0, 2.0),
    "random": tuple(round(_draw.uniform(0.5, 1.5), 2) for _ in range(3)),
}
COST_TAU = "0.99"
MODEL_KEY = {"cbm": "cs", "cem": "cem", "probcbm": "probcbm", "ecbm": "ecbm"}
SWEEP_BUDGETS = (1, 2, 3, 5, 8, 13, "max")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=sorted(MODEL_KEY), required=True)
    ap.add_argument("--cell-px", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taus", nargs="+", default=list(TAUS))
    ap.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="Also write the budget sweep here (CBM, tau 0.99).",
    )
    ap.add_argument(
        "--costs",
        nargs="+",
        choices=sorted(COSTS),
        default=sorted(COSTS),
        help=f"Cost models to score at tau {COST_TAU}.",
    )
    args = ap.parse_args()

    data = load(
        args.data
        / f"sudoku_image_n3_ns1000_mc9_px{args.cell_px}_seed{args.seed}"
        / "ocr_inferred_full_dataset.pkl"
    )
    data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=args.seed)
    model = load(
        args.models
        / f"sudoku_{MODEL_KEY[args.arch]}_tabular_n3_mc9_px{args.cell_px}_seed{args.seed}.model"
    )
    model._random_state = args.seed
    groups = int(round(data.n_concepts / 3))

    def score(split, tau, budgets=(1, 3, "max"), cost="equal"):
        kwargs = {}
        if (
            cost != "equal"
        ):  # keeps the script runnable on code without cost-aware checks (the control)
            kwargs["concept_costs"] = np.repeat(COSTS[cost], groups)
        model._random_state = args.seed  # the same Monte Carlo draws for every target
        return automation_table(
            model,
            data.validation,
            split,
            budgets,
            target_accuracy=float(tau),
            seed=args.seed,
            concept_groups=groups,
            calibrate=True,
            **kwargs,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"sudoku__arch-{args.arch}__res-{args.cell_px}px"
    for tau in args.taus:
        costs = args.costs if tau == COST_TAU else ["equal"]
        for cost in costs:
            table = score(data.test, tau, cost=cost)
            tag = "" if cost == "equal" else f"__cost-{cost}"
            out = (
                args.out
                / f"{stem}__tau-{tau}{tag}__threshold-per-budget__seed-{args.seed}__interventions.csv"
            )
            table.to_csv(out, index=False)
            print(out, flush=True)
    if args.sweep is not None and args.arch == "cbm":
        args.sweep.mkdir(parents=True, exist_ok=True)
        for name, split in (("validation", data.validation), ("test", data.test)):
            table = score(split, "0.99", budgets=SWEEP_BUDGETS)
            table["net_work"] = table["coverage_after"] - table[
                "total_concept_checks"
            ] / (split.n * split.n_concepts)
            out = args.sweep / f"{stem}__tau-0.99__{name}__seed-{args.seed}__ksweep.csv"
            table.to_csv(out, index=False)
            print(out, flush=True)


if __name__ == "__main__":
    main()
