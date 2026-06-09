#!/usr/bin/env python3
"""
Train TinyResNet on the Sudoku OCR demo dataset (fast pipeline only).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from concept_benchmark.data import ConceptDataset
from concept_benchmark.ext.fileutils import load as load_object, save as save_object

from concept_benchmark.synthetic.sudoku.ocr.ocr_utils import (
    SudokuCellDataset,
    TinyResNet,
    compute_class_weights,
    load_sidecars,
    predict_board_argmax,
)
from concept_benchmark.utils import set_deterministic_seed
from concept_benchmark.config import SudokuBenchmarkConfig


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sudoku_ocr_demo")
    logger.setLevel(logging.INFO)

    have_file = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == str(log_path)
        for h in logger.handlers
    )
    if not have_file:
        fh = logging.FileHandler(str(log_path), mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    logger.info(f"[LOG] logging to {log_path}")
    return logger


def train_resnet_tiny(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    class_weights: torch.Tensor | None,
    logger: logging.Logger,
) -> dict:
    model = TinyResNet(num_classes=10).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_val_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"resnet_tiny train {ep}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            preds = out.argmax(1)
            acc = (preds == yb).float().mean().item()
            pbar.set_postfix(loss=float(loss.item()), acc=acc)

        val_loss, val_acc, _ = eval_model(model, val_loader, device)
        logger.info(f"[VAL] resnet_tiny ep={ep} loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return {
        "best_val_acc": float(best_val_acc),
        "best_state": best_state,
    }


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    total_correct = 0
    n_classes = 10
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        loss = crit(out, yb)

        total_loss += float(loss.item()) * xb.size(0)
        total += xb.size(0)

        preds = out.argmax(1)
        total_correct += (preds == yb).sum().item()

        for cls in range(n_classes):
            mask = yb == cls
            total_per_class[cls] += mask.sum().item()
            correct_per_class[cls] += ((preds == yb) & mask).sum().item()

    avg_loss = total_loss / total
    avg_acc = total_correct / total
    per_class_acc = {}
    for cls in range(n_classes):
        if total_per_class[cls] == 0:
            per_class_acc[cls] = None
        else:
            per_class_acc[cls] = correct_per_class[cls] / total_per_class[cls]
    return avg_loss, avg_acc, per_class_acc


def build_ocr_concept_dataset(
    model: nn.Module,
    *,
    dataset_dir: Path,
    jsonl_path: Path,
    tabular_dataset_path: Path,
    cell_px: int,
    margin_px: int,
    device: str,
    out_path: Path,
    logger: logging.Logger,
) -> None:
    records = load_sidecars(jsonl_path)
    tab_ds = load_object(tabular_dataset_path)

    if len(records) != tab_ds.n:
        logger.warning(
            "[WARN] jsonl records (%d) != tabular dataset (%d); indices may misalign.",
            len(records),
            tab_ds.n,
        )

    model.eval()
    pred_X_full = np.zeros((len(records), 81), dtype=np.int64)
    img_paths = []
    for i, rec in enumerate(tqdm(records, desc="ocr inference (boards)")):
        img_path = dataset_dir / rec["img"]
        img_paths.append(str(img_path))
        pred_board = predict_board_argmax(
            model,
            img_path,
            device=device,
            cell_px=cell_px,
            margin_px=margin_px,
        )
        pred_X_full[i] = pred_board.reshape(-1)

    meta = dict(tab_ds.meta)
    meta["data_type"] = meta.get("data_type", "tabular")
    meta["transform"] = "ocr_inferred"
    meta["dataset_name"] = f"{meta.get('dataset_name', 'sudoku')}_ocr_inferred_full"
    meta["img_paths"] = img_paths

    ocr_ds = ConceptDataset(
        inputs=pred_X_full,
        C=tab_ds.C,
        y=tab_ds.y,
        meta=meta,
        input_type="tabular",
        classes=tuple(tab_ds.classes),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_object(ocr_ds, out_path, overwrite=True)
    logger.info("[INFO] saved OCR-inferred ConceptDataset to %s", out_path)


def main():
    defaults = SudokuBenchmarkConfig.default()
    ap = argparse.ArgumentParser(
        description="Train TinyResNet on Sudoku OCR sidecar (demo)."
    )
    ap.add_argument("--n", type=int, default=defaults.block_size)
    ap.add_argument("--n-samples", type=int, default=defaults.n_boards)
    ap.add_argument("--max-corrupt", type=int, default=defaults.max_cell_swaps)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--val-fraction", type=float, default=0.12, help="Validation split fraction."
    )
    ap.add_argument(
        "--min-val", type=int, default=600, help="Minimum validation examples."
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--cell-px", type=int, default=50, help="Must match the render settings."
    )
    ap.add_argument(
        "--margin-px", type=int, default=2, help="Must match the render settings."
    )
    ap.add_argument(
        "--no-debug-dumps", action="store_true", help="Disable writing sample crops."
    )
    ap.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use existing model.",
    )

    args, _ = ap.parse_known_args()
    set_deterministic_seed(args.seed)
    cfg = SudokuBenchmarkConfig(
        block_size=args.n,
        n_boards=args.n_samples,
        max_cell_swaps=args.max_corrupt,
        seed=args.seed,
        cell_px=args.cell_px,
    )

    dataset_dir = cfg.get_dataset_path(data_type="image")
    tab_dataset_dir = cfg.get_dataset_path(data_type="tabular")
    jsonl_path = dataset_dir / "ocr_preprocessing" / "ocr_preprocessing.jsonl"
    model_out = dataset_dir / "ocr_best_model.pt"
    meta_out = dataset_dir / "ocr_best_model.json"
    log_file = dataset_dir / "ocr_train.log"
    tabular_dataset_path = tab_dataset_dir / "sudoku_dataset.pkl"
    ocr_dataset_out = dataset_dir / "ocr_inferred_full_dataset.pkl"

    logger = setup_logging(log_file)

    logger.info(f"[INFO] loading dataset from {dataset_dir} using {jsonl_path}")
    sudoku_ds = SudokuCellDataset(
        dataset_dir,
        jsonl_path,
        cell_px=args.cell_px,
        margin_px=args.margin_px,
        dump_debug=not args.no_debug_dumps,
    )

    # split
    n_total = len(sudoku_ds)
    n_val = max(args.min_val, int(args.val_fraction * n_total))
    n_val = min(n_val, n_total // 2) if n_total > 1 else 0
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        sudoku_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"[INFO] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = args.device
    class_weights = compute_class_weights(train_ds, device=device)
    logger.info(
        "[INFO] class weights: %s", class_weights.cpu().numpy().round(4).tolist()
    )

    if args.skip_training:
        if not model_out.exists():
            raise FileNotFoundError(f"model not found: {model_out}")
        logger.info(
            "[INFO] skip training enabled; using existing model at %s", model_out
        )
    else:
        logger.info("\n==============================")
        logger.info("[INFO] training model: resnet_tiny (fast pipeline)")
        logger.info("==============================")
        t0 = time.time()
        res = train_resnet_tiny(
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            class_weights=class_weights,
            logger=logger,
        )
        t1 = time.time()

        torch.save(res["best_state"], model_out)
        with meta_out.open("w") as f:
            json.dump(
                {"name": "resnet_tiny", "val_acc": float(res["best_val_acc"])},
                f,
                indent=2,
            )

        logger.info("\n================ BEST MODEL ================")
        logger.info("name: resnet_tiny")
        logger.info(f"val acc: {res['best_val_acc']:.4f}")
        logger.info(f"weights: {model_out}")
        logger.info(f"meta:    {meta_out}")
        logger.info(f"elapsed: {t1 - t0:.2f}s")

    logger.info("\n================ OCR -> CONCEPT DATASET ================")
    model = TinyResNet(num_classes=10).to(device)
    state = torch.load(model_out, map_location=device)
    model.load_state_dict(state)
    build_ocr_concept_dataset(
        model,
        dataset_dir=dataset_dir,
        jsonl_path=jsonl_path,
        tabular_dataset_path=tabular_dataset_path,
        cell_px=args.cell_px,
        margin_px=args.margin_px,
        device=device,
        out_path=ocr_dataset_out,
        logger=logger,
    )
    logger.info("[INFO] done.")


if __name__ == "__main__":
    main()
