"""Generate the sudoku automation figures from the refit grid.

Writes two PDFs: one panel per selective-accuracy target, and a single panel at
the headline target.

Usage:
    python scripts/paper/generate_automation_figures.py
    python scripts/paper/generate_automation_figures.py --out-dir ../concept-benchmark-paper/figures
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments.plots import (  # noqa: E402
    AutomationSeries,
    plot_coverage_and_work,
    use_paper_style,
)

N_TEST_BOARDS = 200
N_CONCEPTS = 27
BUDGETS = ["0", "1", "3", "27"]
BUDGET_LABELS = [0, 1, 3, "max"]
TARGETS = {"90": 0.90, "925": 0.925, "95": 0.95, "975": 0.975, "99": 0.99}


def standard_error(values: list[float]) -> float:
    return st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0


def read_series(data_dir: Path, token: str, res: int, family: str) -> AutomationSeries | None:
    per_seed: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    pattern = f"sudoku__arch-{family}__res-{res}px__tau-0.{token}__threshold-per-budget__seed-*__interventions.csv"
    for path in sorted(glob.glob(str(data_dir / pattern))):
        seed = Path(path).name.split("__seed-")[1].split("__")[0]
        with open(path) as handle:
            for row in csv.DictReader(handle):
                try:
                    coverage = float(row["coverage_after"]) * 100
                except (ValueError, TypeError, KeyError):
                    continue
                checks = float(row["total_concept_checks"] or 0)
                net = coverage - 100 * checks / (N_TEST_BOARDS * N_CONCEPTS)
                per_seed[seed][row["budget"]] = (coverage, net)

    if not per_seed:
        return None
    coverage, coverage_err, net, net_err = [], [], [], []
    for budget in BUDGETS:
        points = [per_seed[s][budget] for s in per_seed if budget in per_seed[s]]
        if not points:
            return None
        coverage.append(st.mean(p[0] for p in points))
        coverage_err.append(standard_error([p[0] for p in points]))
        net.append(st.mean(p[1] for p in points))
        net_err.append(standard_error([p[1] for p in points]))
    return AutomationSeries(BUDGET_LABELS, coverage, net, coverage_err, net_err)


def read_baseline(selective_dir: Path, target: float, res: int) -> float | None:
    values: list[float] = []
    for path in sorted(glob.glob(str(selective_dir / f"sudoku__*__res-{res}px__threshold-per-budget__seed-*__selective-all-tau.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("model") != "dnn" or not row.get("selective_cov"):
                    continue
                if abs(float(row["target_accuracy"]) - target) < 1e-9:
                    values.append(float(row["selective_cov"]) * 100)
    return st.mean(values) if values else None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "results/paper/sudoku/cells")
    ap.add_argument("--selective-dir", type=Path, default=root / "results/paper/sudoku/selective")
    ap.add_argument("--out-dir", type=Path, default=root / "results/paper/figures")
    ap.add_argument("--res", type=int, default=18)
    ap.add_argument("--family", default="cbm")
    ap.add_argument("--headline", type=float, default=0.95)
    args = ap.parse_args()

    use_paper_style()
    series, baselines = {}, {}
    for token, target in TARGETS.items():
        found = read_series(args.data_dir, token, args.res, args.family)
        if found is None:
            print(f"  no data for tau={target}, skipping")
            continue
        series[target] = found
        baseline = read_baseline(args.selective_dir, target, args.res)
        if baseline is not None:
            baselines[target] = baseline
        print(
            f"tau={target:<6} coverage {found.coverage[0]:6.2f} -> {found.coverage[-1]:6.2f}"
            f"   net {found.net_work[0]:6.2f} -> {found.net_work[-1]:6.2f}"
            f"   dnn {baselines.get(target, float('nan')):.2f}"
        )

    if not series:
        raise SystemExit(f"no CELL2_iv_* data under {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep = args.out_dir / "automation_targets.pdf"
    single = args.out_dir / "automation_single.pdf"

    fig, _ = plot_coverage_and_work(series, baselines, panel_width=2.25)
    fig.savefig(sweep, bbox_inches="tight")

    headline = {args.headline: series[args.headline]}
    headline_baseline = {k: v for k, v in baselines.items() if k == args.headline}
    fig, _ = plot_coverage_and_work(
        headline, headline_baseline, panel_width=3.8, panel_height=2.6
    )
    fig.savefig(single, bbox_inches="tight")
    print(f"\nwrote {sweep}\nwrote {single}")


if __name__ == "__main__":
    main()
