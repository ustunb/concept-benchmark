#!/usr/bin/env python3
"""
Generate an image Sudoku dataset plus OCR sidecar JSONL
for the TinyResNet OCR pipeline. Also save a
separate tabular ConceptDataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2

from concept_benchmark.data import ConceptDataset
from concept_benchmark.ext.fileutils import save as save_object
from concept_benchmark.ext.fileutils import load as load_object
from concept_benchmark.synthetic.sudoku import (
    create_sudoku_dataset,
    default_transform,
    image_transform,
)
from concept_benchmark.synthetic.sudoku_ocr.ocr_utils import DATA_SUDOKU
from concept_benchmark.config import SudokuBenchmarkConfig


DIGITS_DIR = DATA_SUDOKU / "digits"


# ---------------- image transform wrapper ----------------
def make_image_transform(args):
    """
    Wrap image_transform so that:
      - It still renders and saves the image.
      - It ALSO collects per-board 9x9 board/starters/candidates
        for later use when building the OCR JSONL.
    """
    boards_list = []
    starters_list = []
    candidates_list = []

    def _wrapped(board, *, outfile=None):
        # image_transform must support return_meta=True and return:
        #   (img_or_path, starters, candidates_meta)
        img_or_path, starters, _candidates = image_transform(
            board,
            cell_px=args.cell_px,
            margin_px=args.margin_px,
            line_px=args.line_px,
            bold_px=args.bold_px,
            font_size=args.font_size,
            standardize=(not args.no_standardize),
            font_path=args.font_path,
            handwriting=args.handwriting,
            outfile=outfile,
            return_meta=True,
        )

        # ensure numpy arrays
        board_arr = np.array(board, copy=True)
        starters_arr = np.array(starters, copy=True).astype(int)

        # candidates = inverse of starters (1 for non-starter cell, 0 for starter)
        candidates_arr = (1 - starters_arr).astype(int)

        boards_list.append(board_arr)
        starters_list.append(starters_arr)
        candidates_list.append(candidates_arr)

        return img_or_path

    _wrapped.boards = boards_list
    _wrapped.starters = starters_list
    _wrapped.candidates = candidates_list
    _wrapped.__name__ = "image_transform"
    return _wrapped


# ---------------- helpers ----------------
def infer_meta_like(ds, transform_name: str):
    """Fallback summary dict if ds.meta is missing."""
    C = ds.C
    N = n = None
    if isinstance(C, np.ndarray) and C.ndim == 2 and C.shape[1] % 3 == 0:
        # Heuristic: C.shape[1] = 3 * N^2 -> N^2 = C.shape[1] // 3
        N_cells = C.shape[1] // 3
        rt = int(np.sqrt(N_cells))
        if rt * rt == N_cells:
            n = rt
            N = rt
    return {
        "data_type": "image",
        "N": N,
        "n": n,
        "transform": transform_name,
    }


def save_dataset_pkl(ds, meta_full: dict, save_dir: Path):
    """
    Save the *entire* ConceptDataset instance as a .pkl file and write
    both meta.json and blue_blob.json (for downstream preprocessing)
    into save_dir.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out = save_dir / "sudoku_dataset.pkl"
    save_object(ds, out, overwrite=True)
    print(f"Saved ConceptDataset pickle to {out}")

    meta_path = save_dir / "meta.json"
    meta_path.write_text(json.dumps(meta_full, indent=2))
    print(f"Saved meta to {meta_path}")

    blue_blob_path = save_dir / "blue_blob.json"
    blue_blob_path.write_text(json.dumps(meta_full, indent=2))
    print(f"Saved blue_blob metadata to {blue_blob_path}")


def save_image_side_artifacts(image_dir: Path, meta_full: dict, args):
    """
    For image datasets, also save:
      - blue_blob.json
      - preprocessing.json
    under the dataset directory.
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    # 1) blue_blob.json (mirror of meta_full)
    bb_path = image_dir / "blue_blob.json"
    bb_path.write_text(json.dumps(meta_full, indent=2))
    print(f"Saved image blue_blob metadata to {bb_path}")

    # 2) preprocessing.json – the knobs the preprocessing step probably needs
    preprocessing = {
        "cell_px": args.cell_px,
        "margin_px": args.margin_px,
        "line_px": args.line_px,
        "bold_px": args.bold_px,
        "font_size": args.font_size,
        "standardize": not args.no_standardize,
        "font_path": args.font_path,
        "handwriting": args.handwriting,
        "radius": args.radius,
        "sigma": args.sigma,
        "angle": args.angle,
        "n": meta_full.get("n"),
        "N": meta_full.get("N"),
        "data_type": meta_full.get("data_type"),
        "transform": meta_full.get("transform"),
        "dataset_name": meta_full.get("dataset_name"),
    }
    prep_path = image_dir / "preprocessing.json"
    prep_path.write_text(json.dumps(preprocessing, indent=2))
    print(f"Saved preprocessing config to {prep_path}")


def build_ocr_preprocessing(
    ds,
    meta_full: dict,
    args,
    dataset_dir: Path,
    image_transform_wrapper=None,
):
    """
    Build the ocr_preprocessing folder with a JSONL file:

      <dataset_dir>/ocr_preprocessing/ocr_preprocessing.jsonl

    Each line looks like:
      {
        "img": "<actual filename>.png",
        "starters": [...9x9...],
        "candidates": [...9x9...],
        "board": [...9x9...]
      }
    """
    if image_transform_wrapper is None:
        print("WARNING: build_ocr_preprocessing called without image_transform_wrapper; cannot build OCR JSON.")
        return

    boards_list = getattr(image_transform_wrapper, "boards", None)
    starters_list = getattr(image_transform_wrapper, "starters", None)

    if boards_list is None or starters_list is None:
        print("WARNING: image_transform_wrapper is missing boards/starters; cannot build OCR JSON.")
        return

    boards = np.asarray(boards_list)
    starters = np.asarray(starters_list).astype(int)
    y = np.asarray(ds.y)

    if boards.ndim != 3:
        print(f"WARNING: boards array has unexpected shape {boards.shape}; cannot build OCR JSON.")
        return

    num_samples, N1, N2 = boards.shape
    if N1 != N2:
        print(f"WARNING: boards are not square: shape={boards.shape}; cannot build OCR JSON.")
        return

    if num_samples != len(y):
        print(
            f"WARNING: boards count {num_samples} != len(y) {len(y)}; "
            "mismatch between collected metadata and dataset."
        )
        return

    N = N1
    if N != 9:
        print(f"WARNING: expected 9x9 boards, got {N}x{N}; OCR JSON will not be generated.")
        return

    candidates = (1 - starters).astype(int)
    X_paths = np.asarray(ds.X)

    ocr_dir = dataset_dir / "ocr_preprocessing"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    out_path = ocr_dir / "ocr_preprocessing.jsonl"

    with out_path.open("w") as f:
        for i in range(num_samples):
            img_path = Path(str(X_paths[i]))
            img_name = img_path.name

            row = {
                "img": img_name,
                "starters": starters[i].tolist(),     # 9x9
                "candidates": candidates[i].tolist(), # 9x9 = inverse of starters
                "board": boards[i].astype(int).tolist(),  # 9x9
            }
            f.write(json.dumps(row) + "\n")

    print(f"Saved OCR preprocessing JSONL to {out_path}")


def load_boards_from_jsonl(jsonl_path: Path) -> np.ndarray:
    boards = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "board" not in row:
                raise ValueError(f"Missing 'board' key in {jsonl_path}")
            boards.append(np.array(row["board"], dtype=np.int32))
    if not boards:
        raise ValueError(f"No boards found in {jsonl_path}")
    return np.stack(boards, axis=0)


def ensure_digits_dir(args):
    """
    Ensure that DATA_SUDOKU/digits exists and contains example digits for
    the OCR model to train on.

    Layout:
      data/sudoku/digits/0/*.png
      ...
      data/sudoku/digits/9/*.png
    """
    if DIGITS_DIR.exists() and any(DIGITS_DIR.iterdir()):
        print(f"Digits directory already exists at {DIGITS_DIR}, not regenerating.")
        return

    if cv2 is None:
        print(
            "WARNING: cv2 is not available, cannot generate example digit images "
            f"for {DIGITS_DIR}. Install opencv-python if you need fresh digits."
        )
        return

    print(f"Creating example digits for OCR under {DIGITS_DIR}...")
    DIGITS_DIR.mkdir(parents=True, exist_ok=True)

    digits_per_class = getattr(args, "digits_per_class", 64)
    h = w = args.cell_px

    for d in range(10):
        label_dir = DIGITS_DIR / str(d)
        label_dir.mkdir(parents=True, exist_ok=True)

        for i in range(digits_per_class):
            img = np.ones((h, w, 3), dtype=np.uint8) * 255

            text = str(d)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x = (w - tw) // 2
            y = (h + th) // 2

            cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            out_path = label_dir / f"{d}_{i:04d}.png"
            cv2.imwrite(str(out_path), img)

    print("Finished generating example digits.")


# ---------------- main ----------------
def main():
    defaults = SudokuBenchmarkConfig.default()
    ap = argparse.ArgumentParser(description="Generate a Sudoku OCR dataset (images + sidecar).")
    ap.add_argument("--n", type=int, default=defaults.n)
    ap.add_argument("--n-samples", type=int, default=defaults.n_samples)
    ap.add_argument("--valid-ratio", type=float, default=defaults.valid_ratio)
    ap.add_argument("--max-corrupt", type=int, default=defaults.max_corrupt)
    ap.add_argument("--seed", type=int, default=defaults.seed)

    ap.add_argument("--progress", action="store_true", help="Show a progress bar during generation.")

    # image knobs (aligned with run_sudoku defaults)
    ap.add_argument("--cell-px", type=int, default=50)
    ap.add_argument("--margin-px", type=int, default=2)
    ap.add_argument("--line-px", type=int, default=2)
    ap.add_argument("--bold-px", type=int, default=5)
    ap.add_argument("--font-size", type=int, default=25)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--font-path", type=str, default=None)
    ap.add_argument("--handwriting", type=bool, default=True)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--angle", type=float, default=98)

    ap.add_argument("--digits-per-class", type=int, default=64)
    ap.add_argument("--skip-digits", action="store_true", help="Skip creating data/sudoku/digits samples.")
    ap.add_argument("--skip-image", action="store_true", help="Skip image dataset generation.")

    args = ap.parse_args()

    cfg = SudokuBenchmarkConfig(
        n=args.n, n_samples=args.n_samples,
        max_corrupt=args.max_corrupt, seed=args.seed,
    )
    dataset_dir = cfg.get_dataset_path(data_type="image")
    dataset_name = dataset_dir.name
    image_dataset_path = dataset_dir / "sudoku_dataset.pkl"
    ocr_jsonl = dataset_dir / "ocr_preprocessing" / "ocr_preprocessing.jsonl"

    if args.skip_image:
        ds = load_object(image_dataset_path)
        transform = None
        meta = getattr(ds, "meta", {})
        transform_name = meta.get("transform", "image_transform") if isinstance(meta, dict) else "image_transform"
    else:
        # wrap image_transform so we can capture boards/starters
        transform = make_image_transform(args)
        transform_name = "image_transform"

        ds = create_sudoku_dataset(
            n=args.n,
            n_samples=args.n_samples,
            valid_ratio=args.valid_ratio,
            max_corrupt=args.max_corrupt,
            data_type="image",
            seed=args.seed,
            transform=transform,
            dataset_name=dataset_name,
        )

    X, C, y = ds.X, ds.C, ds.y

    meta = getattr(ds, "meta", None)
    if meta is None:
        meta = getattr(getattr(ds, "_full", None), "meta", None)
    if meta is None:
        meta = infer_meta_like(ds, transform_name=transform_name)

    meta_full = dict(meta)
    meta_full.setdefault("data_type", "image")
    meta_full.setdefault("transform", transform_name)

    if "N" not in meta_full or "n" not in meta_full:
        inferred = infer_meta_like(ds, transform_name=transform_name)
        if inferred.get("N") is not None:
            meta_full.setdefault("N", inferred["N"])
        if inferred.get("n") is not None:
            meta_full.setdefault("n", inferred["n"])

    meta_full["dataset_name"] = dataset_name

    def shape_of(x):
        try:
            return x.shape
        except Exception:
            try:
                return tuple(np.array(x, dtype=object).shape)
            except Exception:
                return ("<unknown>",)

    print("=== Dataset Summary ===")
    print(f"data_type: {meta_full.get('data_type')}")
    print(f"N (board size): {meta_full.get('N')}  |  n (block size): {meta_full.get('n')}")
    print(f"samples: {len(y)}  |  valid: {int((y == 1).sum())}  |  invalid: {int((y == 0).sum())}")
    print(f"X shape: {shape_of(X)}")
    print(f"C shape: {C.shape}")
    print(f"y shape: {y.shape}")
    print(f"transform: {meta_full.get('transform')}")
    print(f"Images & CSVs saved under {dataset_dir}.")

    if not args.skip_image:
        save_dataset_pkl(ds, meta_full, dataset_dir)

        image_dir = dataset_dir
        save_image_side_artifacts(image_dir, meta_full, args)

        if not args.skip_digits:
            ensure_digits_dir(args)

        build_ocr_preprocessing(
            ds,
            meta_full,
            args,
            dataset_dir,
            image_transform_wrapper=transform,
        )

    # Tabular ConceptDataset aligned to the image dataset (always saved)
    tab_dataset_dir = cfg.get_dataset_path(data_type="tabular")
    tab_ds_name = tab_dataset_dir.name
    if transform is not None:
        boards = np.asarray(getattr(transform, "boards", []))
    else:
        boards = load_boards_from_jsonl(ocr_jsonl)
    X_tab = np.stack([default_transform(b) for b in boards], axis=0)
    tab_meta_full = dict(meta_full)
    tab_meta_full["data_type"] = "tabular"
    tab_meta_full["transform"] = "default_transform"
    tab_meta_full["dataset_name"] = tab_ds_name
    tab_ds = ConceptDataset(X=X_tab, C=ds.C, y=ds.y, meta=tab_meta_full)

    tab_save_dir = tab_dataset_dir
    save_dataset_pkl(tab_ds, tab_meta_full, tab_save_dir)

    print("=== Tabular Dataset Summary ===")
    print(f"data_type: {tab_meta_full.get('data_type')}")
    print(f"N (board size): {tab_meta_full.get('N')}  |  n (block size): {tab_meta_full.get('n')}")
    print(f"samples: {len(tab_ds.y)}  |  valid: {int((tab_ds.y == 1).sum())}  |  invalid: {int((tab_ds.y == 0).sum())}")
    print(f"X shape: {getattr(tab_ds.X, 'shape', '<unknown>')}")
    print(f"C shape: {getattr(tab_ds.C, 'shape', '<unknown>')}")
    print(f"y shape: {getattr(tab_ds.y, 'shape', '<unknown>')}")
    print(f"transform: {tab_meta_full.get('transform')}")
    print(f"Saved tabular ConceptDataset to {tab_save_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
