"""Write the sudoku automation table (net work automated and coverage by resolution x architecture x budget).

Reads the installed cells (results/paper/sudoku/cells, selective-accuracy target tau = 0.95, per-budget thresholds)
and writes `tables/sudoku_px_sweep.tex`: mean ± SE over seeds at k = 0, 1, 3, max. Net work automated is the coverage
minus the share of concepts checked (checks / (test boards x 27 concepts)). The highest value of each metric per
resolution is bold, once per row at the earliest budget. The DNN row is its coverage at the same target (the DNN takes no
interventions, so it has one value per metric).

    python scripts/paper/make_sudoku_table.py --out ../concept-benchmark-paper/tables/sudoku_px_sweep.tex
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from _common import (
    PAPER,
    add_results_root,
    latex_cell,
    mean_se,
    read_budget_rows,
    use_results_root,
)

ARCHS = [
    ("cbm", "\\CBM{}"),
    ("cem", "\\CEM{}"),
    ("probcbm", "\\ProbCBM{}"),
    ("ecbm", "\\ECBM{}"),
]
RESOLUTIONS = (50, 10)
N_TEST_BOARDS, N_CONCEPTS = 200, 27
TAU = "0.95"  # the main text; the appendix sweeps the target


def read_cells(
    res: int, arch: str, tau: str = TAU
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-seed net work automated and coverage (percent) at k = 0, 1, 3, max."""
    net, coverage = [], []
    for f in sorted(
        PAPER.sudoku_cells.glob(
            f"sudoku__arch-{arch}__res-{res}px__tau-{tau}__threshold-per-budget__seed-*__interventions.csv"
        )
    ):
        rows = read_budget_rows(f)
        if len(rows) != 4 or any(not r.get("coverage_after") for r in rows):
            raise SystemExit(f"{f.name}: expected 4 budgets with selective columns")
        cov = [100 * float(r["coverage_after"]) for r in rows]
        coverage.append(cov)
        net.append(
            [
                c
                - 100 * float(r["total_concept_checks"]) / (N_TEST_BOARDS * N_CONCEPTS)
                for c, r in zip(cov, rows)
            ]
        )
    return net, coverage


def dnn_coverage(res: int, tau: str = TAU) -> list[float]:
    """Per-seed coverage of the DNN at the selective-accuracy target (percent)."""
    values = []
    for f in sorted(
        PAPER.sudoku_selective.glob(
            f"sudoku__*__res-{res}px__threshold-per-budget__seed-*__selective-all-tau.csv"
        )
    ):
        values += [
            100 * float(r["selective_cov"])
            for r in csv.DictReader(f.open())
            if r["model"] == "dnn"
            and r["selective_cov"]
            and abs(float(r["target_accuracy"]) - float(tau)) < 1e-9
        ]
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--tau", default=TAU, help="Selective-accuracy target of the cells to tabulate."
    )
    add_results_root(ap)
    args = ap.parse_args()
    use_results_root(args)
    lines = [
        r"\renewcommand{\arraystretch}{1.05}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll rrrr|rrrr@{}}",
        r"\toprule",
        r"& & \multicolumn{4}{c|}{$\wrk$} & \multicolumn{4}{c}{\textsf{Coverage}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"\textbf{Resolution} & \textbf{Model} & $k{=}0$ & $k{=}1$ & $k{=}3$ & $k{=}\text{max}$ & $k{=}0$ & $k{=}1$ & $k{=}3$ & $k{=}\text{max}$ \\",
    ]
    for res in RESOLUTIONS:
        data = {arch: read_cells(res, arch, args.tau) for arch, _ in ARCHS}
        seeds = {len(net) for net, _ in data.values()}
        if len(seeds) != 1:
            raise SystemExit(
                f"{res}px: architectures have different numbers of seeds {seeds}"
            )
        n = seeds.pop()
        # as in the robot table: bold the highest mean (as printed) of each block, once per row at the earliest budget
        means = {
            a: [
                [round(st.mean(r[k] for r in data[a][m]), 1) for k in range(4)]
                for m in (0, 1)
            ]
            for a, _ in ARCHS
        }
        best = [max(v for a, _ in ARCHS for v in means[a][m]) for m in (0, 1)]
        dnn = dnn_coverage(res, args.tau)
        if len(dnn) != n:
            raise SystemExit(
                f"{res}px: {len(dnn)} DNN seeds, {n} for the architectures"
            )
        lines.append(r"\midrule")
        lines.append(rf"\multirow{{5}}{{*}}{{\textds{{{res}\,px}}}}")
        dnn_macro = "\\DNN{}"
        lines.append(
            f" & {dnn_macro:<11}& "
            + " & ".join([latex_cell(dnn), "--", "--", "--"] * 2)
            + r" \\"
        )
        print(
            f"{res}px dnn      n={n}  coverage {mean_se(dnn)[0]:5.1f}±{mean_se(dnn)[1]:4.1f}"
        )
        for arch, macro in ARCHS:
            net, coverage = data[arch]
            row = []
            for m, runs in enumerate((net, coverage)):
                first = (
                    means[arch][m].index(best[m]) if best[m] in means[arch][m] else None
                )
                row += [
                    latex_cell([r[k] for r in runs], is_best=k == first)
                    for k in range(4)
                ]
            lines.append(f" & {macro:<11}& " + " & ".join(row) + r" \\")
            print(
                f"{res}px {arch:8s} n={n}  net "
                + " ".join(
                    f"{mean_se([r[k] for r in net])[0]:5.1f}±{mean_se([r[k] for r in net])[1]:4.1f}"
                    for k in range(4)
                )
                + "  | coverage "
                + " ".join(
                    f"{mean_se([r[k] for r in coverage])[0]:5.1f}±{mean_se([r[k] for r in coverage])[1]:4.1f}"
                    for k in range(4)
                )
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
