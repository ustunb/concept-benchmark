"""Write the table of deferred sudoku boards that remain deferred with every concept verified.

Reads the per-board files of `measure_intervention_ceiling.py` (results/paper/sudoku/ceiling) and reports, per
architecture, the share of deferred test boards that remain in the abstention band with every concept set to its true
value (per resolution), and the label probability the front-end model assigns to invalid boards with two, three, or
four violated constraints (over both resolutions). Cells are mean ± SE over the runs that reach a threshold:

    python scripts/paper/make_sudoku_ceiling_table.py --out ../concept-benchmark-paper/tables/sudoku_ceiling.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import PAPER, add_results_root, latex_cell, mean_se, use_results_root
from make_sudoku_table import ARCHS, RESOLUTIONS

VIOLATIONS = (2, 3, 4)


def read_boards() -> pd.DataFrame:
    files = sorted(PAPER.sudoku_ceiling.glob("sudoku__*__ceiling.csv"))
    if not files:
        raise SystemExit(f"no ceiling files in {PAPER.sudoku_ceiling}")
    boards = pd.concat(pd.read_csv(f) for f in files)
    return boards[boards["threshold"].notna()]


def probability_cell(values: list[float]) -> str:
    mean, se = mean_se(values)
    return f"{mean:.2f}{{\\scriptsize$\\pm${se:.2f}}}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    add_results_root(ap)
    args = ap.parse_args()
    use_results_root(args)

    boards = read_boards()
    lines = [
        r"\begin{tabular}{@{}l rr rrr@{}}",
        r"& \multicolumn{2}{c}{\textbf{Remain deferred}} & \multicolumn{3}{c}{\textbf{Probability of a valid board}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-6}",
        r"\textbf{Model} & "
        + " & ".join(rf"\textds{{{res}\,px}}" for res in RESOLUTIONS)
        + " & "
        + " & ".join(f"{v} violated" for v in VIOLATIONS)
        + r" \\",
        r"\midrule",
    ]
    for arch, macro in ARCHS:
        rows = boards[boards["arch"] == arch]
        cells = []
        for res in RESOLUTIONS:
            shares = []
            for _, run in rows[rows["px"] == res].groupby("seed"):
                deferred = run["is_deferred"].astype(bool)
                if deferred.any():
                    stuck = deferred & run["is_deferred_true"].astype(bool)
                    shares.append(100 * stuck.sum() / deferred.sum())
            cells.append(latex_cell(shares))
            print(f"{arch} {res}px: {len(shares)} runs with deferred boards")
        invalid = rows[rows["y"] == 0]
        for v in VIOLATIONS:
            per_run = (
                invalid[invalid["n_violations"] == v]
                .groupby(["px", "seed"])["prob_true"]
                .mean()
            )
            cells.append(probability_cell(list(per_run)))
        lines.append(f"{macro:<11}& " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
