#!/usr/bin/env python3
"""
Sandbox script to compare OCR-inferred test dataset vs. tabular dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from concept_benchmark.ext.fileutils import load as load_object
from concept_benchmark.paths import data_dir


def flatten_X(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 81:
        return x
    if x.ndim == 3 and x.shape[1:] == (9, 9):
        return x.reshape(x.shape[0], 81)
    return x.reshape(x.shape[0], -1)


def compute_cell_and_board_acc(pred_X: np.ndarray, gt_X: np.ndarray):
    pred_flat = flatten_X(pred_X)
    gt_flat = flatten_X(gt_X)
    if pred_flat.shape != gt_flat.shape:
        raise ValueError(f"Shape mismatch: pred={pred_flat.shape} gt={gt_flat.shape}")
    cell_correct = (pred_flat == gt_flat).sum()
    total_cells = pred_flat.size
    board_correct = (pred_flat == gt_flat).all(axis=1).sum()
    total_boards = pred_flat.shape[0]
    return (
        cell_correct / max(total_cells, 1),
        board_correct / max(total_boards, 1),
        int(cell_correct),
        int(total_cells),
        int(board_correct),
        int(total_boards),
    )


def main():
    ap = argparse.ArgumentParser(description="Compare OCR-inferred test dataset vs tabular dataset.")
    ap.add_argument(
        "--inferred",
        default=str(data_dir / "sudoku" / "demo_ocr_m_21" / "ocr_inferred_full_dataset.pkl"),
        help="OCR-inferred ConceptDataset (.pkl).",
    )
    ap.add_argument(
        "--tabular",
        default=str(data_dir / "sudoku" / "demo_ocr_m_21_tabular" / "sudoku_dataset.pkl"),
        help="Tabular ConceptDataset (.pkl).",
    )
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--fold-id", type=str, default="K05N01")
    ap.add_argument("--fold-num-validation", type=int, default=4)
    ap.add_argument("--fold-num-test", type=int, default=5)
    ap.add_argument("--show-samples", type=int, default=3)
    args = ap.parse_args()

    inferred_ds = load_object(Path(args.inferred))
    tabular_ds = load_object(Path(args.tabular))

    inferred_ds.generate_cvindices(
        strata=inferred_ds.y,
        total_folds_for_cv=[5],
        seed=args.split_seed,
    )
    inferred_ds.split(
        fold_id=args.fold_id,
        fold_num_validation=args.fold_num_validation,
        fold_num_test=args.fold_num_test,
    )

    tabular_ds.generate_cvindices(
        strata=tabular_ds.y,
        total_folds_for_cv=[5],
        seed=args.split_seed,
    )
    tabular_ds.split(
        fold_id=args.fold_id,
        fold_num_validation=args.fold_num_validation,
        fold_num_test=args.fold_num_test,
    )

    print("=== Dataset Shapes ===")
    print(f"inferred X: {np.asarray(inferred_ds.X).shape}")
    print(f"inferred C: {np.asarray(inferred_ds.C).shape}")
    print(f"inferred y: {np.asarray(inferred_ds.y).shape}")
    print(f"inferred test X: {np.asarray(inferred_ds.test.X).shape}")
    print(f"inferred test C: {np.asarray(inferred_ds.test.C).shape}")
    print(f"inferred test y: {np.asarray(inferred_ds.test.y).shape}")
    print(f"tabular test X: {np.asarray(tabular_ds.test.X).shape}")
    print(f"tabular test C: {np.asarray(tabular_ds.test.C).shape}")
    print(f"tabular test y: {np.asarray(tabular_ds.test.y).shape}")

    try:
        cell_acc, board_acc, cell_ok, cell_total, board_ok, board_total = compute_cell_and_board_acc(
            inferred_ds.test.X, tabular_ds.test.X
        )
        print("\n=== OCR vs Tabular (test split) ===")
        print(f"cell acc:  {cell_acc:.4f} ({cell_ok}/{cell_total})")
        print(f"board acc: {board_acc:.4f} ({board_ok}/{board_total})")
    except ValueError as exc:
        print(f"\n[WARN] Could not compute OCR-vs-tabular accuracy: {exc}")

    try:
        pred_flat = flatten_X(inferred_ds.test.X)
        gt_flat = flatten_X(tabular_ds.test.X)
        if pred_flat.shape != gt_flat.shape:
            raise ValueError(f"Shape mismatch: pred={pred_flat.shape} gt={gt_flat.shape}")
        cell_correct = (pred_flat == gt_flat).all(axis=1)
        wrong_indices = np.where(~cell_correct)[0]
        print(f"wrong board indices (test split): {wrong_indices.tolist()}")
        test_y = np.asarray(tabular_ds.test.y).reshape(-1)
        for label in np.unique(test_y):
            mask = test_y == label
            total = int(mask.sum())
            correct = int(cell_correct[mask].sum())
            acc = correct / max(total, 1)
            print(f"board acc (y={int(label)}): {acc:.4f} ({correct}/{total})")
    except ValueError as exc:
        print(f"[WARN] Could not compute accuracy by label: {exc}")

    n_show = max(0, int(args.show_samples))
    if n_show:
        print("\n=== Sample Boards ===")
        for i in range(min(n_show, len(inferred_ds.test.y))):
            pred_board = np.asarray(inferred_ds.test.X[i]).reshape(9, 9)
            gt_board = np.asarray(tabular_ds.test.X[i]).reshape(9, 9)
            print(f"\nSample {i}")
            print("pred:")
            print(pred_board)
            print("gt:")
            print(gt_board)


if __name__ == "__main__":
    main()
