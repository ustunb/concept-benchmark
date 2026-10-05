"""Paths and helpers shared by the paper scripts.

Importing this module also puts the repository root and `scripts/` on the import path, so a script run as
`python scripts/paper/<name>.py` can import `concept_benchmark`, `experiments`, `robot_pipeline` and `sudoku_pipeline`.
"""

from __future__ import annotations

import csv
import statistics as st
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER_RESULTS = REPO / "results/paper"
BALANCED_RULE = PAPER_RESULTS / "robot/balanced_rule"
SPARSE_RULE = PAPER_RESULTS / "robot/grid"
INTERVENTION_RECORDS = BALANCED_RULE / "intervention_records"
ROBOT_DATASETS = PAPER_RESULTS / "robot/datasets"
SUDOKU_CELLS = PAPER_RESULTS / "sudoku/cells"
BALANCED_TAG = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"

for _path in (str(REPO), str(REPO / "scripts")):
    if _path not in sys.path:
        sys.path.append(_path)


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error (sample standard deviation / sqrt(n))."""
    return st.mean(values), st.stdev(values) / len(values) ** 0.5


def latex_cell(values: Sequence[float], is_best: bool = False) -> str:
    """Table cell `mean% ± SE` over runs, in bold if `is_best`."""
    mean, se = mean_se(values)
    if is_best:
        return f"\\textbf{{{mean:.1f}\\%}}{{\\scriptsize$\\boldsymbol{{\\pm}}$\\textbf{{{se:.1f}}}}}"
    return f"{mean:.1f}\\%{{\\scriptsize$\\pm${se:.1f}}}"


def filename_fields(path: Path) -> dict[str, str]:
    """Fields of a result file name: `robot__rule-sparse__seed-1014__results.csv` -> {"rule": "sparse", "seed": "1014"}."""
    return dict(kv.split("-", 1) for kv in path.stem.split("__")[1:-1])


def read_budget_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a result CSV with one row per intervention budget, ordered by budget."""
    with path.open() as fh:
        return sorted(csv.DictReader(fh), key=lambda r: int(r["budget"]))


def compile_tex(tex_path: Path, passes: int = 1) -> Path:
    """Compile a standalone .tex file with pdflatex, remove its .aux and .log files and return the PDF path."""
    for _ in range(passes):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                                cwd=tex_path.parent, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(result.stdout[-2500:])
    for ext in (".aux", ".log"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return tex_path.with_suffix(".pdf")
