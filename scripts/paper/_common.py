"""Paths and helpers shared by the paper scripts.

Importing this module also puts the repository root and `scripts/` on the import path, so a script run as
`python scripts/paper/<name>.py` can import `concept_benchmark`, `experiments`, `robot_pipeline` and `sudoku_pipeline`.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BALANCED_TAG = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"
RESULTS_ROOT_VARIABLE = "CONCEPT_BENCHMARK_PAPER_RESULTS"
RESULTS_DOWNLOAD = "https://github.com/ustunb/concept-benchmark/releases"


class PaperResults:
    """Folders of the paper's results under one root (see `scripts/paper/README.md` for the layout)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def balanced_rule(self) -> Path:
        return self.root / "robot/balanced_rule"

    @property
    def sparse_rule(self) -> Path:
        return self.root / "robot/grid"

    @property
    def intervention_records(self) -> Path:
        return self.balanced_rule / "intervention_records"

    @property
    def detector_outputs(self) -> Path:
        return self.balanced_rule / "detector_outputs"

    @property
    def alignment(self) -> Path:
        return self.root / "robot/alignment"

    @property
    def robot_datasets(self) -> Path:
        return self.root / "robot/datasets"

    @property
    def llm_caches(self) -> Path:
        return self.root / "robot/llm_caches"

    @property
    def sudoku_cells(self) -> Path:
        return self.root / "sudoku/cells"

    @property
    def sudoku_selective(self) -> Path:
        return self.root / "sudoku/selective"

    @property
    def sudoku_confidence(self) -> Path:
        return self.root / "sudoku/confidence"

    @property
    def real_datasets(self) -> Path:
        return self.root / "real_datasets"

    @property
    def models_balanced(self) -> Path:
        return self.root / "models/robot/balanced"

    @property
    def models_sparse(self) -> Path:
        return self.root / "models/robot/grid"

    @property
    def images_224px(self) -> Path:
        return self.root / "images/robot_224px"


# The scripts read their inputs from here; `use_results_root` points it at the folder given on the command line.
PAPER = PaperResults(os.environ.get(RESULTS_ROOT_VARIABLE, REPO / "results/paper"))


def add_results_root(parser: argparse.ArgumentParser) -> None:
    """Add `--results-root`, the folder that holds the paper's results."""
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PAPER.root,
        help=f"Folder with the paper's results (default: ${RESULTS_ROOT_VARIABLE} or results/paper; "
        f"download from {RESULTS_DOWNLOAD}, or build one from your own runs with collect_pipeline_runs.py).",
    )


def use_results_root(args: argparse.Namespace) -> None:
    """Read the paper's results from `args.results_root`; exit with the download link if the folder is missing."""
    if not args.results_root.is_dir():
        sys.exit(
            f"No results folder at {args.results_root}. Download the paper's results from {RESULTS_DOWNLOAD} "
            "and pass the unpacked folder as --results-root, or build one with collect_pipeline_runs.py."
        )
    PAPER.root = args.results_root


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
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(result.stdout[-2500:])
    for ext in (".aux", ".log"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return tex_path.with_suffix(".pdf")
