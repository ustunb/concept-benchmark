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

# 9x9 image dataset; images + CSVs land in <repo_root>/data/sudoku/<ds_name>
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
sys.path.append(os.getcwd())

import numpy as np
import torch

# progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# --- repo path shim (safe if already installed) ---
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# where image datasets live by default
DATA_SUDOKU = repo_root / "data" / "sudoku"
DIGITS_DIR = DATA_SUDOKU / "digits"

from concept_benchmark.synthetic.sudoku import (
    create_sudoku_dataset,
    default_transform,
    onehot_transform,
    histogram_transform,
    image_transform,
)

from concept_benchmark.ext.fileutils import save as save_object

# we’ll use cv2 to render digit examples
try:
    import cv2
except Exception:
    cv2 = None


# ---------------- image transform wrapper ----------------
def make_image_transform(args):
    def _wrapped(board, *, outfile=None):
        return image_transform(
            board,
            cell_px=args.cell_px,
            margin_px=args.margin_px,
            line_px=args.line_px,
            font_size=args.font_size,
            standardize=(not args.no_standardize),
            font_path=args.font_path,
            handwriting=args.handwriting,
            outfile=outfile,
        )
    _wrapped.__name__ = "image_transform"
    return _wrapped


# ---------------- helpers ----------------
def infer_meta_like(ds, data_type: str, transform_name: str):
    """Build a summary dict without touching ds.meta."""
    C = ds.C
    N = n = None
    if isinstance(C, np.ndarray) and C.ndim == 2 and C.shape[1] % 3 == 0:
        # C.shape[1] = 3 * N^2 -> N^2 = C.shape[1] // 3
        N_cells = C.shape[1] // 3
        rt = int(np.sqrt(N_cells))
        n = rt if rt * rt == N_cells else None
        N = rt if n is not None else None
    return {
        "data_type": data_type,
        "N": N,
        "n": n,
        "transform": transform_name,
    }


def save_dataset_pkl(ds, meta_full, save_dir: Path):
    """
    Save the dataset using fileutils.save as a .pkl file and write
    both meta.json and blue_blob.json (for downstream preprocessing)
    into save_dir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    obj = {
        "X": ds.X,
        "C": ds.C,
        "y": ds.y,
        "meta_like": meta_full,
    }
    out = save_dir / "sudoku_dataset.pkl"
    # NOTE: if your fileutils.save has signature save(path, obj),
    # flip the arguments here.
    save_object(obj, out)
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

    # You can tune this if you want more/less examples per class
    digits_per_class = getattr(args, "digits_per_class", 64)
    h = w = args.cell_px

    for d in range(10):
        label_dir = DIGITS_DIR / str(d)
        label_dir.mkdir(parents=True, exist_ok=True)

        for i in range(digits_per_class):
            img = np.ones((h, w, 3), dtype=np.uint8) * 255

            text = str(d)
            # Rough centering heuristic
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


def build_ocr_preprocessing(ds, meta_full: dict, args, ds_name: str | None):
    """
    Build the ocr_preprocessing folder with a JSONL file:

      data/sudoku/<dataset_name>/ocr_preprocessing/ocr_preprocessing.jsonl

    Each line looks like:
      {"img": "valid_1.png", "starters": [...9x9...],
       "candidates": [...9x9...], "board": [...9x9...]}

    We assume:
      - ds.C has shape (num_samples, 3 * N^2)
      - channels are [board, starters, candidates] per cell.
    """
    C = np.asarray(ds.C)
    y = np.asarray(ds.y)

    if C.ndim != 2 or C.shape[1] % 3 != 0:
        print("WARNING: C does not have shape (num_samples, 3 * N^2); "
              "cannot build ocr_preprocessing.")
        return

    num_samples, dim = C.shape
    N_cells = dim // 3
    rt = int(np.sqrt(N_cells))
    if rt * rt != N_cells:
        print("WARNING: C second dimension not consistent with square board; "
              "cannot build ocr_preprocessing.")
        return

    N = rt

    # reshape to (num_samples, N_cells, 3)
    flat = C.reshape(num_samples, N_cells, 3)

    boards = flat[:, :, 0].reshape(num_samples, N, N).astype(int)
    starters = flat[:, :, 1].reshape(num_samples, N, N).astype(int)
    candidates = flat[:, :, 2].reshape(num_samples, N, N).astype(int)

    # where to write
    if ds_name is None:
        ds_name = meta_full.get("dataset_name", "unnamed")

    ocr_dir = DATA_SUDOKU / ds_name / "ocr_preprocessing"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    out_path = ocr_dir / "ocr_preprocessing.jsonl"

    with out_path.open("w") as f:
        for i in range(num_samples):
            label_str = "valid" if int(y[i]) == 1 else "invalid"
            # 1-based index to match your example "valid_1.png"
            img_name = f"{label_str}_{i+1}.png"

            row = {
                "img": img_name,
                "starters": starters[i].tolist(),
                "candidates": candidates[i].tolist(),
                "board": boards[i].tolist(),
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
    else:
        # shouldn't happen due to argparse choices
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
    ap.add_argument("--save-dir", type=Path, default=None)
    # kept for compatibility, but unused for pkl saving
    ap.add_argument("--save-format", choices=["npz", "pt"], default="npz")
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

        if data_type == "image":
            # image always uses image_transform
            transform = make_image_transform(args)
            transform_name = "image_transform"
        else:
            # tabular uses CLI-specified transform
            transform, transform_name = choose_tabular_transform(args)

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

        # -------- summary WITHOUT touching ds.meta ----------
        X, C, y = ds.X, ds.C, ds.y
        meta = getattr(getattr(ds, "_full", None), "meta", None)
        if meta is None:
            meta = infer_meta_like(ds, data_type=data_type, transform_name=transform_name)

        def shape_of(x):
            try:
                return x.shape
            except Exception:
                try:
                    return tuple(np.array(x, dtype=object).shape)
                except Exception:
                    return ("<unknown>",)

        print("=== Dataset Summary ===")
        print(f"data_type: {meta.get('data_type')}")
        print(f"N (board size): {meta.get('N')}  |  n (block size): {meta.get('n')}")
        print(f"samples: {len(y)}  |  valid: {int((y == 1).sum())}  |  invalid: {int((y == 0).sum())}")
        print(f"X shape: {shape_of(X)}")
        print(f"C shape: {C.shape}")
        print(f"y shape: {y.shape}")
        print(f"transform: {meta.get('transform')}")
        if data_type == "image":
            print(f"Images & CSVs saved under {DATA_SUDOKU / (ds_name or '')}.")

        # Build full meta (including dataset_name) once
        meta_like = {
            k: meta.get(k)
            for k in ("data_type", "N", "n", "transform")
            if k in meta
        }
        meta_full = dict(meta_like)
        if ds_name is not None:
            meta_full["dataset_name"] = ds_name

        # save ConceptDataset as .pkl and write blue_blob.json into save_dir (if provided)
        if args.save_dir is not None:
            # If generating multiple types, save each under its own subdir
            save_dir = args.save_dir / data_type if multiple else args.save_dir
            save_dataset_pkl(ds, meta_full, save_dir)

        # For image datasets, drop side artifacts next to the images
        if data_type == "image" and ds_name is not None:
            image_dir = DATA_SUDOKU / ds_name
            save_image_side_artifacts(image_dir, meta_full, args)

            # Also ensure the global digits/ folder exists with example digits
            if not digits_done:
                ensure_digits_dir(args)
                digits_done = True

            # And build the OCR preprocessing JSONL
            build_ocr_preprocessing(ds, meta_full, args, ds_name)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
