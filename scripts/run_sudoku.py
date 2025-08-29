"""
make_sudoku_dataset.py

Run the create_sudoku_dataset() helper to generate synthetic Sudoku data.

Examples
--------
# 9x9, 10k tabular rows, 50/50 valid:invalid
python make_sudoku_dataset.py --n 3 --n-samples 10000 --data-type tabular --save-dir out/tabular_npz

# 9x9, one-hot features (N x N x N), save as .npz
python make_sudoku_dataset.py --transform onehot --save-dir out/onehot_npz

# 9x9, per-unit histograms (3N x N), save as .pt
python make_sudoku_dataset.py --transform histogram --save-dir out/hist_pt --save-format pt

# 9x9 image dataset; images + CSVs land in <data_dir>/sudoku/<ds_name>
python make_sudoku_dataset.py --data-type image --ds-name demo_imgs --n-samples 2000 --valid-ratio 0.4 --cell-px 24 --font-size 14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os
sys.path.append(os.getcwd())

import argparse, json, sys
from pathlib import Path
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

from concept_benchmark.synthetic.sudoku import (
    create_sudoku_dataset,
    default_transform,
    onehot_transform,
    histogram_transform,
    image_transform,
    sudoku_image_preprocess,
)

# ---------------- image transform wrapper ----------------
def make_image_transform(args):
    def _wrapped(board, *, outfile=None):
        return image_transform(
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
        )
    _wrapped.__name__ = "image_transform"
    return _wrapped


# ---------------- helpers ----------------
def infer_meta_like(ds, args):
    """Build a summary dict without touching ds.meta."""
    C = ds.C
    N = n = None
    if isinstance(C, np.ndarray) and C.ndim == 2 and C.shape[1] % 3 == 0:
        N = C.shape[1] // 3
        rt = int(np.sqrt(N))
        n = rt if rt * rt == N else None
    return {
        "data_type": args.data_type,
        "N": N,
        "n": n,
        "transform": (args.transform if args.transform != "auto"
                      else ("image_transform" if args.data_type == "image" else "default_transform")),
    }

def save_non_image(ds, meta_like, save_dir: Path, save_format: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "meta.json").write_text(json.dumps(meta_like, indent=2))
    if save_format == "npz":
        out = save_dir / "sudoku_dataset.npz"
        np.savez_compressed(out, X=np.asarray(ds.X), C=ds.C, y=ds.y)
        print(f"Saved arrays to {out}")
    else:
        out = save_dir / "sudoku_dataset.pt"
        torch.save({"X": ds.X, "C": ds.C, "y": ds.y, "meta_like": meta_like}, out)
        print(f"Saved tensors to {out}")
    print(f"Saved meta to {save_dir / 'meta.json'}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Generate a Sudoku ConceptDataset.")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--valid-ratio", type=float, default=0.5)
    ap.add_argument("--max-corrupt", type=int, default=3)
    ap.add_argument("--data-type", choices=["tabular", "image"], default="image")
    ap.add_argument("--transform", choices=["auto", "default", "onehot", "histogram", "image"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ds_name", type=str, default=None)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--save-format", choices=["npz", "pt"], default="npz")
    ap.add_argument("--progress", action="store_true", help="Show a progress bar during generation.")
    # image knobs
    ap.add_argument("--cell-px", type=int, default=16)
    ap.add_argument("--margin-px", type=int, default=3)
    ap.add_argument("--line-px", type=int, default=1)
    ap.add_argument("--bold-px", type=int, default=1)
    ap.add_argument("--font-size", type=int, default=10)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--font-path", type=str, default=None)
    ap.add_argument("--handwriting", type=bool, default=False)
    ap.add_argument("--radius", type=float, default=0)
    ap.add_argument("--sigma", type=float, default=12)
    ap.add_argument("--angle", type=float, default=140)
    args = ap.parse_args()

    # choose transform
    if args.data_type == "image":
        transform = make_image_transform(args)
    else:
        if args.transform in ("auto", "default"):
            transform = default_transform
        elif args.transform == "onehot":
            transform = onehot_transform
        elif args.transform == "histogram":
            transform = histogram_transform
        elif args.transform == "image":
            args.data_type = "image"
            transform = make_image_transform(args)
        else:
            transform = default_transform

    ds = create_sudoku_dataset(
        n=args.n,
        n_samples=args.n_samples,
        valid_ratio=args.valid_ratio,
        max_corrupt=args.max_corrupt,
        data_type=args.data_type,
        seed=args.seed,
        transform=transform,
        ds_name=args.ds_name,
    )

    # -------- summary WITHOUT touching ds.meta ----------
    X, C, y = ds.X, ds.C, ds.y
    # try the internal sample's meta; otherwise infer
    meta = getattr(getattr(ds, "_full", None), "meta", None)
    if meta is None:
        meta = infer_meta_like(ds, args)

    def shape_of(x):
        try: return x.shape
        except Exception:
            try: return tuple(np.array(x, dtype=object).shape)
            except Exception: return ("<unknown>",)

    print("=== Dataset Summary ===")
    print(f"data_type: {meta.get('data_type')}")
    print(f"N (board size): {meta.get('N')}  |  n (block size): {meta.get('n')}")
    print(f"samples: {len(y)}  |  valid: {int((y==1).sum())}  |  invalid: {int((y==0).sum())}")
    print(f"X shape: {shape_of(X)}")
    print(f"C shape: {C.shape}")
    print(f"y shape: {y.shape}")
    print(f"transform: {meta.get('transform')}")
    if args.data_type == "image":
        print(f"Images & CSVs saved under data_dir/sudoku/{args.ds_name}.")

    # save (non-image)
    if args.data_type != "image" and args.save_dir is not None:
        meta_like = {
            k: meta.get(k)
            for k in ("data_type", "N", "n", "transform")
            if k in meta
        }
        save_non_image(ds, meta_like, args.save_dir, args.save_format)

if __name__ == "__main__":
    main()
