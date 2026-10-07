"""Measure how well the digits and the constraints of the sudoku boards are detected, per seed and resolution.

For every seed and resolution, compares the recognizer-inferred boards with the true boards (cell accuracy and the
share of boards with a misread digit, per split) and the CBM's concept predictions with the true concepts on the
test boards (concept accuracy, share of boards with a wrong concept, false and missed violation rates). Appends one
row per seed and resolution to the CSV that `make_sudoku_detection_table.py` reads:

    python scripts/paper/measure_sudoku_detection.py --seeds 171 172 --cell-px 50 18 --out results/paper/sudoku/detection.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import _common  # noqa: F401  (puts the repository root on the import path)
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.ext.fileutils import load


def split_indices(split) -> np.ndarray:
    indices = np.asarray(split.indices)
    return np.where(indices)[0] if indices.dtype == bool else indices


def measure(seed: int, cell_px: int) -> dict[str, float]:
    config = SudokuBenchmarkConfig(seed=seed)
    config.cell_px = cell_px
    inferred = load(
        config.get_dataset_path(data_type="image") / "ocr_inferred_full_dataset.pkl"
    )
    true = load(config.get_dataset_path(data_type="tabular") / "sudoku_dataset.pkl")
    n_cells = config.block_size**4
    X_inferred = np.asarray(inferred.X).reshape(-1, n_cells).astype(int)
    X_true = np.asarray(true.X).reshape(-1, n_cells).astype(int)
    inferred.sample(test_size=0.2, val_size=0.2, stratify=inferred.y, seed=seed)
    model = load(config.get_model_path("cs", data_type="tabular"))
    predicted = model.concept_detector.predict_proba(inferred.test) >= 0.5
    actual = np.asarray(inferred.test.C) >= 0.5

    row: dict[str, float] = {"px": cell_px, "seed": seed}
    for name, split in (
        ("train", inferred.train),
        ("validation", inferred.validation),
        ("test", inferred.test),
    ):
        rows = split_indices(split)
        misread = X_inferred[rows] != X_true[rows]
        row[f"cell_acc_{name}"] = float(1 - misread.mean())
        row[f"boards_misread_{name}"] = float(misread.any(axis=1).mean())
    row["concept_acc_test"] = float((predicted == actual).mean())
    row["boards_concept_error_test"] = float((predicted != actual).any(axis=1).mean())
    row["false_violation_rate_test"] = float((~predicted & actual).sum() / actual.sum())
    row["missed_violation_rate_test"] = float(
        (predicted & ~actual).sum() / max((~actual).sum(), 1)
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--cell-px", type=int, nargs="+", default=[50, 18])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [measure(seed, px) for px in args.cell_px for seed in args.seeds]
    for row in rows:
        print(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    print("wrote", len(rows), "rows to", args.out)


if __name__ == "__main__":
    main()
