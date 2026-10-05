"""Write the two sensitivity tables of the sudoku automation results and print the paired tests behind their text.

Reads the installed cells (results/paper/sudoku/cells, per-budget thresholds) and writes

- `--tau-out`: net work automated and coverage at the selective-accuracy targets tau = 0.90, 0.95, 0.99;
- `--cost-out`: the same at tau = 0.95 under three costs of a concept check: every check costs the same (`equal`), a
  block check costs twice a row or column check (`2x block`), and each kind of check has its own cost drawn once from
  U(0.5, 1.5) (`random`).

Cells are mean ± SE over seeds at k = 0, 1, 3, max. Net work automated is the coverage minus the cost of the checks as a
share of all concepts (cost / (test boards x 27 concepts)). For every setting and architecture, the script prints the
paired t-tests of the two comparisons made in the text: 18 px against 50 px without interventions, and k = max against
k = 0 at 18 px.

    python scripts/paper/make_sudoku_sensitivity_tables.py \
        --tau-out ../concept-benchmark-paper/tables/sudoku_tau_sweep.tex \
        --cost-out ../concept-benchmark-paper/tables/sudoku_cost_models.tex
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from scipy import stats

from _common import SUDOKU_CELLS, latex_cell, read_budget_rows
from make_sudoku_table import ARCHS, N_CONCEPTS, N_TEST_BOARDS, RESOLUTIONS

TAUS = ("0.90", "0.95", "0.99")
COST_TAU = "0.95"
_draw = random.Random(0)
# cost of a row, column and block check
COST_MODELS = [
    ("equal", (1.0, 1.0, 1.0)),
    ("2$\\times$ block", (1.0, 1.0, 2.0)),
    ("random", tuple(round(_draw.uniform(0.5, 1.5), 2) for _ in range(3))),
]


def read_cells(res: int, arch: str, tau: str, costs: tuple[float, float, float]) -> dict[int, tuple[list[float], list[float]]]:
    """seed -> (net work automated, coverage) in percent at k = 0, 1, 3, max."""
    runs = {}
    pattern = f"sudoku__arch-{arch}__res-{res}px__tau-{tau}__threshold-per-budget__seed-*__interventions.csv"
    for f in sorted(SUDOKU_CELLS.glob(pattern)):
        rows = read_budget_rows(f)
        if len(rows) != 4 or any(not r.get("coverage_after") for r in rows):
            raise SystemExit(f"{f.name}: expected 4 budgets with selective columns")
        coverage = [100 * float(r["coverage_after"]) for r in rows]
        cost = [sum(c * float(r[k]) for c, k in zip(costs, ("row_checks", "col_checks", "block_checks"))) for r in rows]
        net = [cov - 100 * c / (N_TEST_BOARDS * N_CONCEPTS) for cov, c in zip(coverage, cost)]
        runs[int(re.search(r"seed-(\d+)", f.name).group(1))] = (net, coverage)
    return runs


def paired(a: list[float], b: list[float]) -> str:
    if all(abs(x - y) < 1e-9 for x, y in zip(a, b)):
        return "identical"
    t, p = stats.ttest_rel(a, b)
    return f"diff {sum(a) / len(a) - sum(b) / len(b):+6.1f}  t={t:6.2f}  p={p:.3f}"


def table(settings: list[tuple[str, str, tuple[float, float, float]]], first_header: str) -> list[str]:
    """One block of rows per setting (label, tau, costs); prints the tests of each setting."""
    lines = [
        r"\begin{tabular}{@{}lll rrrr|rrrr@{}}",
        r"\toprule",
        r"& & & \multicolumn{4}{c|}{$\wrk$} & \multicolumn{4}{c}{\textsf{Coverage}} \\",
        r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}",
        rf"\textbf{{{first_header}}} & \textbf{{Resolution}} & \textbf{{Model}} & "
        + " & ".join([r"$k{=}0$ & $k{=}1$ & $k{=}3$ & $k{=}\text{max}$"] * 2) + r" \\",
    ]
    for label, tau, costs in settings:
        lines.append(r"\midrule")
        runs = {(res, arch): read_cells(res, arch, tau, costs) for res in RESOLUTIONS for arch, _ in ARCHS}
        for i, res in enumerate(RESOLUTIONS):
            for j, (arch, macro) in enumerate(ARCHS):
                seeds = sorted(runs[res, arch])
                net, coverage = ([runs[res, arch][s][m] for s in seeds] for m in (0, 1))
                cells = [latex_cell([r[k] for r in metric]) for metric in (net, coverage) for k in range(4)]
                head = (rf"\multirow{{8}}{{*}}{{{label}}}" if i == j == 0 else "") + " & "
                head += rf"\multirow{{4}}{{*}}{{\textds{{{res}\,px}}}}" if j == 0 else ""
                lines.append(f"{head} & {macro:<11}& " + " & ".join(cells) + r" \\")
                if len(seeds) != 10:
                    print(f"  note: {label} {res}px {arch}: {len(seeds)} seeds")
        print(f"== {first_header} {label}")
        for arch, _ in ARCHS:
            shared = sorted(set(runs[18, arch]) & set(runs[50, arch]))
            low, high = runs[18, arch], runs[50, arch]
            print(f"  {arch:8s} 18px vs 50px, net at k=0 (n={len(shared)}): {paired([low[s][0][0] for s in shared], [high[s][0][0] for s in shared])}")
            seeds = sorted(low)
            print(f"  {arch:8s} 18px, net at k=max vs k=0:        {paired([low[s][0][3] for s in seeds], [low[s][0][0] for s in seeds])}")
            print(f"  {arch:8s} 18px, coverage at k=max vs k=0:   {paired([low[s][1][3] for s in seeds], [low[s][1][0] for s in seeds])}")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-out", type=Path, required=True)
    ap.add_argument("--cost-out", type=Path, required=True)
    args = ap.parse_args()
    args.tau_out.write_text("\n".join(table([(f"${tau}$", tau, COST_MODELS[0][1]) for tau in TAUS], r"$\tau$")) + "\n")
    print(f"wrote {args.tau_out}")
    print("cost of a row, column and block check:", {name: costs for name, costs in COST_MODELS})
    args.cost_out.write_text("\n".join(table([(name, COST_TAU, costs) for name, costs in COST_MODELS], "Cost")) + "\n")
    print(f"wrote {args.cost_out}")


if __name__ == "__main__":
    main()
