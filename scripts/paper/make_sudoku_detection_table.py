"""Write the detection-quality table of the sudoku automation results.

Reads `results/paper/sudoku/detection.csv` (one row per resolution and seed) and writes, per resolution, the digit
recognizer's cell accuracy on the test boards, the share of test boards with at least one misread digit, the CBM's
concept accuracy on the test boards and the share of test boards with at least one wrong concept. Cells are mean ± SE
over seeds, in percent.

    python scripts/paper/make_sudoku_detection_table.py --out ../concept-benchmark-paper/tables/sudoku_detection.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _common import PAPER, add_results_root, use_results_root

COLUMNS = (
    ("cell_acc_test", "Cell accuracy"),
    ("boards_misread_test", "Boards with a misread digit"),
    ("concept_acc_test", "Concept accuracy"),
    ("boards_concept_error_test", "Boards with a wrong concept"),
)


def cell(values: list[float]) -> str:
    v = np.asarray(values) * 100
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return rf"{v.mean():.1f}\%{{\scriptsize$\pm${se:.1f}}}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    add_results_root(ap)
    args = ap.parse_args()
    use_results_root(args)
    with PAPER.sudoku_detection.open() as f:
        rows = list(csv.DictReader(f))
    lines = [
        r"\begin{tabular}{@{}l rr|rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{\textbf{Digit recognizer}} & \multicolumn{2}{c}{\textbf{\CBM{} concept detectors}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"\textbf{Resolution} & "
        + " & ".join(f"\\textsf{{{name}}}" for _, name in COLUMNS)
        + r" \\",
        r"\midrule",
    ]
    for res in (50, 18):
        seeds = [r for r in rows if int(r["px"]) == res]
        cells = [cell([float(r[key]) for r in seeds]) for key, _ in COLUMNS]
        lines.append(rf"\textds{{{res}\,px}} & " + " & ".join(cells) + r" \\")
        print(
            res,
            "px",
            f"n={len(seeds)}",
            " | ".join(f"{name}: {c}" for (_, name), c in zip(COLUMNS, cells)),
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.write_text("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
