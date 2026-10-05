"""CUB-200 real-data check: interventions under STANDARD (GT-supervised) concepts
vs LABEL-FREE (LF-CBM, Oikarinen 2023) concepts.

Both arms use the SAME-CAPACITY label head (an MLP of identical size) and the same
CLIP backbone, so neither is handicapped -- each is the best in its own category.
They differ only where they must:
  standard   : INDEPENDENT CBM -- MLP head trained on the GROUND-TRUTH concepts
               (no image leakage); concept detector predicts the concepts.
               Interventions set concepts to their true 0/1 value (in-distribution
               for a binary-trained head).
  label-free : LF-CBM -- MLP head trained on the CLIP-projected concept scores
               (unsupervised, inherently sequential). Interventions use Koh-2020
               5th/95th-percentile in-distribution values (as we report for robots).

Run:  ./venv/bin/python experiments/cub_standard_vs_clip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig, _CLIPEncoder  # noqa: E402
from experiments.cub_lfcbm_regime import load_cub_subset, CACHE, N_CLASSES, _dev  # noqa: E402

SEED = 0
KCOLS = [0, 1, 10, "max"]
HEAD = dict(hidden_layer_sizes=(128,), max_iter=800, random_state=SEED)
FEAT_CACHE = Path(__file__).resolve().parent / f"_cub_featstd_{N_CLASSES}.npz"


def acc(pp, y):
    return float((pp.argmax(1) == y).mean())


def resp_order(pred, base, reveal):
    bp = pred(base); bcls = bp.argmax(1); n, m = base.shape; rows = np.arange(n)
    score = np.zeros((n, m))
    for j in range(m):
        t = base.copy(); t[:, j] = reveal[:, j]
        score[:, j] = np.abs(pred(t)[rows, bcls] - bp[rows, bcls])
    return np.argsort(-score, axis=1)


def curve(pred, base, reveal, order, y):
    n, m = base.shape; rows = np.arange(n)
    out = {0: acc(pred(base), y)}
    for k in [c for c in KCOLS if c != 0]:
        kk = m if k == "max" else min(int(k), m)
        rev = np.zeros((n, m), bool); rev[rows[:, None], order[:, :kk]] = True
        out[k] = acc(pred(np.where(rev, reveal, base)), y)
    return out


def main():
    paths, y, C_true, is_train, texts = load_cub_subset()
    rng = np.random.default_rng(SEED)
    tr_all = np.where(is_train)[0]; te = np.where(~is_train)[0]
    perm = rng.permutation(tr_all); cut = int(0.8 * len(perm))
    fit, va = perm[:cut], perm[cut:]; trv = np.r_[fit, va]
    maj = np.bincount(y[te]).max() / len(te)
    print(f"[data] CUB classes={N_CLASSES} imgs={len(paths)} train={len(trv)} test={len(te)} majority={maj*100:.1f}%")

    if FEAT_CACHE.exists():
        F = np.load(FEAT_CACHE)["F"]
    else:
        F = _CLIPEncoder("ViT-B-32", "laion2b_s34b_b79k", _dev()).encode_images(paths, batch_size=64)
        np.savez(FEAT_CACHE, F=F)

    # ---------- STANDARD: independent CBM, MLP head on GROUND-TRUTH concepts ----------
    Chat = np.zeros_like(C_true)                       # detector: predict concepts from CLIP feats
    for j in range(C_true.shape[1]):
        col = C_true[trv, j]
        Chat[:, j] = col[0] if col.min() == col.max() else \
            LogisticRegression(max_iter=1000, random_state=SEED).fit(F[trv], col).predict(F)
    head_s = MLPClassifier(**HEAD).fit(C_true[trv].astype(float), y[trv])   # independent (on GT)
    pred_s = head_s.predict_proba
    base_s = Chat[te].astype(float)
    reveal_s = C_true[te].astype(float)                # true 0/1
    std = curve(pred_s, base_s, reveal_s, resp_order(pred_s, base_s, reveal_s), y[te])

    # ---------- LABEL-FREE: LF-CBM, MLP head on projected concept scores ----------
    cset = LFConceptSet(keys=[f"a{i}" for i in range(len(texts))], texts=texts)
    lf = LabelFreeCBM(LFTrainingConfig(device=_dev(), seed=SEED, cache_dir=CACHE))
    lf.fit(train_X=[paths[i] for i in fit], train_y=y[fit],
           valid_X=[paths[i] for i in va], valid_y=y[va], concept_set=cset)
    keep = np.asarray(lf.keep_mask, dtype=bool)
    S = lf.concept_proba(paths)                        # (N, M_kept) continuous
    Cgt = C_true[:, keep]
    head_c = MLPClassifier(**HEAD).fit(S[trv], y[trv])  # sequential (on scores)
    pred_c = head_c.predict_proba
    p5, p95 = np.percentile(S[trv], 5, axis=0), np.percentile(S[trv], 95, axis=0)
    base_c = S[te]
    reveal_c = p95[None, :] * Cgt[te] + p5[None, :] * (1 - Cgt[te])   # Koh 5/95 toward GT
    clip = curve(pred_c, base_c, reveal_c, resp_order(pred_c, base_c, reveal_c), y[te])

    def hdr(k):
        return "k=max" if k == "max" else f"k={k}"
    print(f"\n=== CUB-{N_CLASSES}: intervention accuracy (%), matched-capacity (MLP head) ===")
    print(f"{'concepts':22} " + "  ".join(f"{hdr(k):>6}" for k in KCOLS) + "   gain@max")
    for name, d in (("standard (independent)", std), ("label-free (LF-CBM)", clip)):
        row = "  ".join(f"{d[k]*100:6.1f}" for k in KCOLS)
        print(f"{name:22} {row}   {(d['max']-d[0])*100:+6.1f}")


if __name__ == "__main__":
    main()
