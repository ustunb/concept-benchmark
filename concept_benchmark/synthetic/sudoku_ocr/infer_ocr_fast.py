#!/usr/bin/env python3
"""
Fast inference for the Sudoku OCR demo (argmax per cell, no solver).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

# repo path shim (safe if already installed)
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from concept_benchmark.synthetic.sudoku_ocr.ocr_utils import (
    DATA_SUDOKU,
    TinyResNet,
    crop_cell,
    cell_preprocess_28x28,
    load_sidecars,
)


@torch.no_grad()
def predict_board_argmax(
    model: torch.nn.Module,
    img_path: Path,
    *,
    device: str = "cpu",
    cell_px: int = 50,
    margin_px: int = 2,
):
    import cv2

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(str(img_path))

    cells = []
    for r in range(9):
        for c in range(9):
            cell = crop_cell(bgr, r, c, cell_px=cell_px, margin_px=margin_px)
            x28 = cell_preprocess_28x28(cell)
            cells.append(x28)
    cells = np.stack(cells, axis=0)
    xb = torch.from_numpy(cells).unsqueeze(1).to(device)
    logits = model(xb)
    return logits.argmax(1).cpu().numpy().reshape(9, 9)


def main():
    ap = argparse.ArgumentParser(description="Run fast argmax OCR on a Sudoku demo dataset.")
    default_dataset = DATA_SUDOKU / "demo_ocr_m_21"
    ap.add_argument("--dataset-dir", default=str(default_dataset))
    ap.add_argument(
        "--jsonl",
        default=str(default_dataset / "ocr_preprocessing" / "ocr_preprocessing.jsonl"),
        help="Sidecar JSONL with img + board labels.",
    )
    ap.add_argument("--model", default=str(default_dataset / "ocr_best_model.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cell-px", type=int, default=50, help="Must match the render settings.")
    ap.add_argument("--margin-px", type=int, default=2, help="Must match the render settings.")
    ap.add_argument("--preds-out", default=str(default_dataset / "ocr_predictions_fast_demo.jsonl"))

    args = ap.parse_args()

    device = args.device
    model = TinyResNet(num_classes=10).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset_dir = Path(args.dataset_dir)
    records = load_sidecars(Path(args.jsonl))

    preds_path = Path(args.preds_out)
    preds_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "cells": 0,
        "correct_cells": 0,
        "boards": 0,
        "correct_boards": 0,
    }

    t0 = time.time()
    with preds_path.open("w") as fw:
        for rec in tqdm(records, desc="inference"):
            img_path = dataset_dir / rec["img"]
            pred_board = predict_board_argmax(
                model,
                img_path,
                device=device,
                cell_px=args.cell_px,
                margin_px=args.margin_px,
            )

            out_rec = {
                "img": rec["img"],
                "pred_board": pred_board.tolist(),
            }

            # optional accuracy if labels present
            if "board" in rec:
                gt_board = np.array(rec["board"], dtype=int)
                cell_correct = (pred_board == gt_board).sum()
                totals["cells"] += gt_board.size
                totals["correct_cells"] += int(cell_correct)
                totals["boards"] += 1
                totals["correct_boards"] += int(cell_correct == gt_board.size)
                out_rec["gt_board"] = rec["board"]
                out_rec["cell_correct"] = int(cell_correct)
                out_rec["board_correct"] = bool(cell_correct == gt_board.size)

            fw.write(json.dumps(out_rec) + "\n")
    t1 = time.time()

    print(f"Wrote predictions to {preds_path}")
    if totals["cells"]:
        cell_acc = totals["correct_cells"] / totals["cells"]
        board_acc = totals["correct_boards"] / totals["boards"] if totals["boards"] else 0.0
        print(f"Cell acc:  {cell_acc:.4f} ({totals['correct_cells']}/{totals['cells']})")
        print(f"Board acc: {board_acc:.4f} ({totals['correct_boards']}/{totals['boards']})")
    print(f"Elapsed: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
