"""
make_sudoku_dataset.py

Run the create_sudoku_dataset() helper to generate synthetic Sudoku data.

Examples
--------
# 9x9, 10k tabular rows, 50/50 valid:invalid
python make_sudoku_dataset.py --n 3 --n-samples 10000 --data-type tabular --save-dir out/tabular_npz

# 9x9, one-hot features (N x N x N), save as .pkl
python make_sudoku_dataset.py --data-type tabular --transform onehot --save-dir out/onehot_pkl

# 9x9, per-unit histograms (3N x N), save as .pkl
python make_sudoku_dataset.py --data-type tabular --transform histogram --save-dir out/hist_pkl

# 9x9 image dataset; images + CSVs land in <data_dir>/sudoku/<ds_name>
# and the ConceptDataset is saved as a .pkl via fileutils.save
python make_sudoku_dataset.py --data-type image --dataset_name demo_imgs --n-samples 2000 --valid-ratio 0.4 --cell-px 24 --font-size 14 --save-dir out/image_pkl

# generate both image + tabular (separate random draws), each saved as .pkl
python make_sudoku_dataset.py --data-type image tabular --dataset_name demo --save-dir out/both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

import numpy as np

# progress bar
try:
    from tqdm import tqdm  # noqa: F401
except Exception:  # pragma: no cover
    tqdm = None

# --- repo path shim (safe if already installed) ---
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from concept_benchmark.paths import data_dir
from concept_benchmark.synthetic.sudoku import (
    create_sudoku_dataset,
    default_transform,
    onehot_transform,
    histogram_transform,
    image_transform,
)
from concept_benchmark.ext.fileutils import save as save_object

# where image datasets live by default: MUST match create_sudoku_dataset
DATA_SUDOKU = data_dir / "sudoku"
DIGITS_DIR = DATA_SUDOKU / "digits"

# we’ll use cv2 to render digit examples
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


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
def infer_meta_like(ds, data_type: str, transform_name: str):
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
        "data_type": data_type,
        "N": N,
        "n": n,
        "transform": transform_name,
    }


def save_dataset_pkl(ds, meta_full: dict, save_dir: Path):
    """
    Save the *entire* ConceptDataset instance as a .pkl file and write
    both meta.json and blue_blob.json (for downstream preprocessing)
    into save_dir.

    This is what run_sudoku expects: when you load the pkl, you get a
    ConceptDataset back, not a wrapper dict.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    out = save_dir / "sudoku_dataset.pkl"
    save_object(ds, out)
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
    under data/sudoku/<dataset_name>.
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


def ensure_digits_dir(args):
    """
    Ensure that DATA_SUDOKU/digits exists and contains example digits for
    the OCR model to train on.

    Layout:
      data/sudoku/digits/0/*.png
      data/sudoku/digits/1/*.png
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


def build_ocr_preprocessing(
    ds,
    meta_full: dict,
    args,
    ds_name: str | None,
    image_transform_wrapper=None,
):
    """
    Build the ocr_preprocessing folder with a JSONL file:

      data/sudoku/<dataset_name>/ocr_preprocessing/ocr_preprocessing.jsonl

    Each line looks like:
      {
        "img": "<actual filename>.png",
        "starters": [...9x9...],
        "candidates": [...9x9...],
        "board": [...9x9...]
      }

    We NO LONGER use ds.C. Instead we rely on the metadata collected by
    the wrapped image transform created via make_image_transform(args).

    - boards:    9x9 int grids (same as used to render images)
    - starters:  9x9 mask (1 = starter / given digit, 0 = non-starter)
    - candidates: 9x9 mask = 1 - starters (inverse of starters)
    """
    if image_transform_wrapper is None:
        print("WARNING: build_ocr_preprocessing called without image_transform_wrapper; cannot build OCR JSON.")
        return

    # Extract collected per-board metadata from the wrapper
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

    # candidates = inverse of starters (1 for non-starter cells, 0 for starters)
    candidates = (1 - starters).astype(int)

    # Use the actual filenames from ds.X instead of synthesizing names
    X_paths = np.asarray(ds.X)

    if ds_name is None:
        ds_name = meta_full.get("dataset_name", "unnamed")

    ocr_dir = DATA_SUDOKU / ds_name / "ocr_preprocessing"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    out_path = ocr_dir / "ocr_preprocessing.jsonl"

    with out_path.open("w") as f:
        for i in range(num_samples):
            img_path = Path(str(X_paths[i]))
            img_name = img_path.name  # e.g. "valid_0.png" or "invalid_123.png"

            row = {
                "img": img_name,
                "starters": starters[i].tolist(),     # 9x9
                "candidates": candidates[i].tolist(), # 9x9 = inverse of starters
                "board": boards[i].astype(int).tolist(),  # 9x9
            }
            f.write(json.dumps(row) + "\n")

    print(f"Saved OCR preprocessing JSONL to {out_path}")


def choose_tabular_transform(args):
    """
    Decide which transform function to use for tabular data.

    - auto/default -> default_transform
    - onehot      -> onehot_transform
    - histogram   -> histogram_transform
    - image       -> disallowed for tabular
    """
    if args.transform in ("auto", "default"):
        return default_transform, "default_transform"
    elif args.transform == "onehot":
        return onehot_transform, "onehot_transform"
    elif args.transform == "histogram":
        return histogram_transform, "histogram_transform"
    elif args.transform == "image":
        raise ValueError(
            "Invalid configuration: --data-type tabular with --transform image. "
            "There is no reason to transform tabular data to image data."
        )
    else:  # should not happen due to argparse choices
        return default_transform, "default_transform"


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Generate a Sudoku ConceptDataset.")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--valid-ratio", type=float, default=0.5)
    ap.add_argument("--max-corrupt", type=int, default=21)

    # allow multiple data types, e.g. --data-type image tabular
    ap.add_argument(
        "--data-type",
        choices=["tabular", "image"],
        nargs="+",
        default=["image", "tabular"],
        help="One or more data types to generate: tabular, image.",
    )

    ap.add_argument(
        "--transform",
        choices=["auto", "default", "onehot", "histogram", "image"],
        default="auto",
        help=(
            "Transform for tabular data. "
            "For image data, image_transform is used automatically. "
            "With multiple --data-type values, tabular uses this, "
            "image always uses image_transform."
        ),
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_name", type=str, default="multimodal_m_21")
    ap.add_argument("--save-dir", type=Path, default="data/sudoku/multimodal_m_21")
    ap.add_argument("--save-format", choices=["npz", "pt"], default="npz")  # unused, kept for compat
    ap.add_argument("--progress", action="store_true", help="Show a progress bar during generation.")

    # image knobs
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

    # how many examples per digit class to generate in data/sudoku/digits
    ap.add_argument("--digits-per-class", type=int, default=64)

    args = ap.parse_args()

    # Normalize and dedupe data types while preserving order
    data_types = list(dict.fromkeys(args.data_type))
    multiple = len(data_types) > 1

    digits_done = False  # only generate digits once

    for data_type in data_types:
        print("=" * 60)
        print(f"Generating data_type={data_type!r}...")

        # Use suffixed dataset_name in multi-type case to avoid collisions
        if multiple and args.dataset_name is not None:
            ds_name = f"{args.dataset_name}_{data_type}"
        else:
            ds_name = args.dataset_name

        # Choose transform
        if data_type == "image":
            # wrap image_transform so we can capture boards/starters
            transform = make_image_transform(args)
            transform_name = "image_transform"
        else:
            transform, transform_name = choose_tabular_transform(args)

        # ---- actually generate the ConceptDataset (this calls image rendering) ----
        ds = create_sudoku_dataset(
            n=args.n,
            n_samples=args.n_samples,
            valid_ratio=args.valid_ratio,
            max_corrupt=args.max_corrupt,
            data_type=data_type,
            seed=args.seed,
            transform=transform,
            dataset_name=ds_name,
        )

        X, C, y = ds.X, ds.C, ds.y

        # Prefer the ConceptDataset's own meta, fall back if needed
        meta = getattr(ds, "meta", None)
        if meta is None:
            meta = getattr(getattr(ds, "_full", None), "meta", None)
        if meta is None:
            meta = infer_meta_like(ds, data_type=data_type, transform_name=transform_name)

        meta_full = dict(meta)
        meta_full.setdefault("data_type", data_type)
        meta_full.setdefault("transform", transform_name)

        # If N/n are missing, try to infer and fill them
        if "N" not in meta_full or "n" not in meta_full:
            inferred = infer_meta_like(ds, data_type=data_type, transform_name=transform_name)
            if inferred.get("N") is not None:
                meta_full.setdefault("N", inferred["N"])
            if inferred.get("n") is not None:
                meta_full.setdefault("n", inferred["n"])

        if ds_name is not None:
            meta_full["dataset_name"] = ds_name

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
        if data_type == "image":
            print(f"Images & CSVs saved under {DATA_SUDOKU / (ds_name or '')}.")

        # Save ConceptDataset as .pkl (+ meta/blue_blob) in save_dir if requested
        if args.save_dir is not None:
            save_dir = args.save_dir / data_type if multiple else args.save_dir
            save_dataset_pkl(ds, meta_full, save_dir)

        # For image datasets, drop side artifacts next to the images directory
        if data_type == "image" and ds_name is not None:
            image_dir = DATA_SUDOKU / ds_name
            save_image_side_artifacts(image_dir, meta_full, args)

            if not digits_done:
                ensure_digits_dir(args)
                digits_done = True

            # Build the OCR preprocessing JSONL using the SAME wrapped image transform
            build_ocr_preprocessing(
                ds,
                meta_full,
                args,
                ds_name,
                image_transform_wrapper=transform,
            )

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
