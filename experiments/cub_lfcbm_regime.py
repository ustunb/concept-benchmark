"""Label-free (CLIP-concept) CBM interventions on REAL data (CUB-200 subset).

Nobody in the literature documents intervention backfire specifically for
label-free / CLIP-annotated CBMs -- this is what our benchmark shows. Derm7pt
could not test it (CLIP cannot read dermoscopy concepts -> uninformative). CUB is
LF-CBM's native setting: CLIP reads bird attributes well -> informative concept
layer -> leakage -> the regime where interventions can backfire.

We train a LabelFreeCBM (experiments/lfcbm.py) on a CUB class subset with the 312
attribute names as CLIP concepts, then intervene toward the ground-truth CUB
attributes. We compare:
  hard   : revealed concept -> 0/1, whole vector hardened (the pipeline's encoding)
  calib  : revealed concept -> Koh-2020 5th/95th-percentile in-distribution value
Backfire that SURVIVES calib is a genuine leakage effect (not an OOD artifact).

Run:  ./venv/bin/python experiments/cub_lfcbm_regime.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig  # noqa: E402

CUB = Path(os.path.expanduser("~/Downloads/CUB_200_2011/CUB_200_2011"))
IMG_ROOT = CUB / "images"
N_CLASSES = 25          # subset for speed (smaller/simpler than full 200)
BUDGETS = [0, 1, 2, 3, 5, 10]
INCACC_KS = [1, 2, 5]
SEED = 0
CACHE = Path(__file__).resolve().parent / f"_cub_lfcbm_cache_{N_CLASSES}"


def _read_cols(path, n=None):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            rows.append(parts[:n] if n else parts)
    return rows


def load_cub_subset():
    imgs = {int(i): p for i, p in _read_cols(CUB / "images.txt", 2)}
    cls = {int(i): int(c) for i, c in _read_cols(CUB / "image_class_labels.txt", 2)}
    split = {int(i): int(s) for i, s in _read_cols(CUB / "train_test_split.txt", 2)}
    # per-image attribute matrix: file is ordered image-major, 312 attrs each
    raw = np.loadtxt(CUB / "attributes" / "image_attribute_labels.txt",
                     usecols=(0, 1, 2), dtype=int, ndmin=2)
    n_img = raw[:, 0].max()
    n_attr = raw[:, 1].max()
    A = np.zeros((n_img + 1, n_attr), dtype=np.int8)
    A[raw[:, 0], raw[:, 1] - 1] = raw[:, 2]
    # attribute names -> readable CLIP prompts
    attr_file = CUB / "attributes.txt"
    if not attr_file.exists():
        attr_file = CUB.parent / "attributes.txt"     # ships one level up in some CUB copies
    names = [n for _, n in _read_cols(attr_file, 2)]
    texts = ["a photo of a bird with " +
             n.replace("has_", "").replace("::", " ").replace("_", " ") for n in names]

    keep_classes = set(range(1, N_CLASSES + 1))
    ids = [i for i in imgs if cls[i] in keep_classes]
    paths = [str(IMG_ROOT / imgs[i]) for i in ids]
    y = np.array([cls[i] - 1 for i in ids])
    C = A[[i for i in ids]]                      # (n, 312) ground-truth attrs
    is_train = np.array([split[i] for i in ids], dtype=bool)
    return paths, y, C, is_train, texts


def acc(pp, y):
    return float((pp.argmax(1) == y).mean())


def main():
    paths, y, C_true, is_train, texts = load_cub_subset()
    tr_idx = np.where(is_train)[0]
    te_idx = np.where(~is_train)[0]
    # hold out 20% of train as validation for LFCBM
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(tr_idx)
    cut = int(0.8 * len(perm)); fit_idx, va_idx = perm[:cut], perm[cut:]
    maj = np.bincount(y[te_idx]).max() / len(te_idx)
    print(f"[data] classes={N_CLASSES} imgs={len(paths)} train={len(fit_idx)} "
          f"val={len(va_idx)} test={len(te_idx)} | majority={maj*100:.1f}%")

    cset = LFConceptSet(keys=[f"a{i}" for i in range(len(texts))], texts=texts)
    cfg = LFTrainingConfig(device=_dev(), seed=SEED, cache_dir=CACHE)
    lf = LabelFreeCBM(cfg)
    stats = lf.fit(train_X=[paths[i] for i in fit_idx], train_y=y[fit_idx],
                   valid_X=[paths[i] for i in va_idx], valid_y=y[va_idx], concept_set=cset)
    keep = np.asarray(lf.keep_mask, dtype=bool)
    print(f"[lfcbm] kept {int(keep.sum())}/{len(texts)} concepts")

    Ck_full = C_true[:, keep].astype(float)      # GT attrs aligned to kept concepts
    P_tr = lf.concept_proba([paths[i] for i in np.r_[fit_idx, va_idx]])
    pct5 = np.percentile(P_tr, 5, axis=0)
    pct95 = np.percentile(P_tr, 95, axis=0)

    P_te = lf.concept_proba([paths[i] for i in te_idx])
    yt = y[te_idx]
    Ck = Ck_full[te_idx]
    acc_det = acc(lf.predict_from_probs(P_te), yt)

    base = (P_te >= 0.5).astype(float)
    bp = lf.predict_from_probs(base); bcls = bp.argmax(1)
    n, m = base.shape; rows = np.arange(n)
    score = np.zeros((n, m))
    for j in range(m):
        fl = base.copy(); fl[:, j] = 1 - fl[:, j]
        score[:, j] = np.abs(lf.predict_from_probs(fl)[rows, bcls] - bp[rows, bcls])
    order = np.argsort(-score, axis=1)
    Ck_int = Ck.astype(int)
    calib = pct95[None, :] * Ck + pct5[None, :] * (1 - Ck)   # 95th if attr=1 else 5th

    print(f"\n=== CUB-{N_CLASSES} LFCBM (CLIP label-free) interventions toward GT attrs ===")
    print(f"{'method':10} " + " ".join(f"acc@{k}" for k in BUDGETS) + "   incacc(1,2,5)")
    out = {}
    for label, mode in (("hard", "hard"), ("calib", "calib")):
        accs = {0: acc_det}
        for k in [b for b in BUDGETS if b > 0]:
            rev = np.zeros((n, m), bool); rev[rows[:, None], order[:, :min(k, m)]] = True
            if mode == "hard":
                Cfin = np.where(rev, Ck, base)
            else:
                Cfin = np.where(rev, calib, P_te)
            accs[k] = acc(lf.predict_from_probs(Cfin), yt)
        inc = float(np.mean([accs[k] - acc_det for k in INCACC_KS]))
        out[label] = inc
        print(f"{label:10} " + " ".join(f"{accs[k]*100:5.1f}" for k in BUDGETS) + f"    {inc*100:+.1f}")

    print("\n=== VERDICT ===")
    print(f"hard-encoding incacc  = {out['hard']*100:+.1f}%")
    print(f"calibrated    incacc  = {out['calib']*100:+.1f}%")
    if out['calib'] < -2:
        print("Backfire SURVIVES calibration on real CUB -> genuine label-free/CLIP-CBM "
              "leakage backfire, in-distribution. This is the effect no prior work reports for LF-CBMs.")
    elif out['hard'] < -2:
        print("Backfire only under hard encoding -> OOD-value artifact here (calibrated ~0).")
    else:
        print("No backfire on CUB subset -> interventions help/neutral; try more classes.")


def _dev():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
