"""Faithful reproduction of the paper's CLIP/label-free (LFCBM) concept regime on
REAL Derm7pt data, to see whether interventions backfire (negative incacc) as they
do for the synthetic robots.

Mirrors scripts/robot_pipeline.py's machine/clip regime exactly:
  * concepts come from CLIP text similarity (LabelFreeCBM, experiments/lfcbm.py) --
    NOT from ground-truth annotations (label-free).
  * baseline accuracy `acc_det` is computed on CONTINUOUS concept activations
    (robot_pipeline.py:1041).
  * an intervention reveals concepts as HARD 0/1 values and the whole concept
    vector is binarized before the label head (robot_pipeline.py:887-888); the head
    was trained on continuous logit features, so hardening shifts the input.
  * concepts to reveal are chosen per-instance by responsiveness (flip effect),
    matching the KFlip ranking in robot_pipeline.py:844-861.

Unlike the robots (whose CLIP concepts have no ground-truth counterpart, so their
interventions correct toward Gemini re-labels), Derm7pt's 7 CLIP concepts ARE the 7
clinician concepts -- so we correct toward genuine human ground truth, no LLM needed.

Run:  ./venv/bin/python experiments/derm7pt_lfcbm_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig  # noqa: E402
from experiments.derm7pt_concept_noise import load_derm7pt, CONCEPTS, IMAGES  # noqa: E402

# concept texts for CLIP (melanoma-indicating / atypical variant, matching the
# melanoma-indicating binarization of the ground-truth human concepts)
CONCEPT_TEXTS = [
    "an atypical pigment network",
    "irregular streaks",
    "irregular pigmentation",
    "regression structures",
    "irregular dots and globules",
    "a blue-whitish veil",
    "atypical vascular structures",
]
BUDGETS = [0, 1, 2, 3, 5, 7]
INCACC_KS = [1, 2, 5]
SEEDS = [0, 1, 2]
CACHE = Path(__file__).resolve().parent / "_derm7pt_lfcbm_cache"


def acc(pred_proba, y):
    return float((pred_proba.argmax(1) == y.astype(int)).mean())


def run_seed(lf, C_true_kept, y, test, seed, mu):
    """One LFCBM. Baseline on continuous activations; interventions reveal hard
    ground-truth concepts (responsiveness-ranked) and harden the whole vector."""
    P_te = lf.concept_proba(test["paths"])                              # (N, Mk) continuous
    yt = y[test["idx"]]
    acc_det = acc(lf.predict_from_probs(P_te), yt)                       # continuous baseline

    base_hard = (P_te >= 0.5).astype(int)                               # hardened base
    base_prob = lf.predict_from_probs(base_hard)
    base_cls = base_prob.argmax(1)
    n, m = base_hard.shape
    rows = np.arange(n)

    # responsiveness: |change in predicted-class prob| when flipping each concept bit
    score = np.zeros((n, m))
    for j in range(m):
        flip = base_hard.copy(); flip[:, j] = 1 - flip[:, j]
        p = lf.predict_from_probs(flip)
        score[:, j] = np.abs(p[rows, base_cls] - base_prob[rows, base_cls])
    order = np.argsort(-score, axis=1)                                  # most responsive first

    Ck = C_true_kept[test["idx"]].astype(float)                         # hard human truth (kept concepts)
    Ck_int = Ck.astype(int)
    calib_val = mu[np.arange(m)[None, :], Ck_int]                        # in-distribution value for true state
    accs_hard = {0: acc_det}   # method #1: harden whole vector (paper's robot_pipeline.py:887)
    accs_soft = {0: acc_det}   # method #2: overwrite only revealed concepts hard 0/1, keep rest soft (Koh)
    accs_cal = {0: acc_det}    # method #3: revealed concept set to calibrated in-distribution value
    for k in [b for b in BUDGETS if b > 0]:
        reveal = np.zeros((n, m), dtype=bool)
        kk = min(k, m)
        reveal[rows[:, None], order[:, :kk]] = True
        accs_hard[k] = acc(lf.predict_from_probs(np.where(reveal, Ck, base_hard)), yt)
        accs_soft[k] = acc(lf.predict_from_probs(np.where(reveal, Ck, P_te)), yt)
        accs_cal[k] = acc(lf.predict_from_probs(np.where(reveal, calib_val, P_te)), yt)
    inc = lambda a: float(np.mean([a[k] - acc_det for k in INCACC_KS]))
    return accs_hard, accs_soft, accs_cal, inc(accs_hard), inc(accs_soft), inc(accs_cal), acc_det


def main():
    y, C_true, paths, train, valid, test = load_derm7pt()
    trainval = np.concatenate([train, valid])
    all_paths = list(paths)                                             # already absolute
    concept_set = LFConceptSet(keys=[f"c{i}" for i in range(len(CONCEPTS))], texts=CONCEPT_TEXTS)
    print(f"[data] mel={int(y.sum())} nevus={int((1-y).sum())} | "
          f"train+val={len(trainval)} test={len(test)}")

    results = []
    kept_report = None
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        boot = rng.choice(trainval, size=len(trainval), replace=True) if seed > 0 else trainval
        # split boot into train/valid for LFCBM
        cut = int(0.8 * len(boot)); tr, va = boot[:cut], boot[cut:]
        cfg = LFTrainingConfig(device=_dev(), seed=seed, cache_dir=CACHE)
        lf = LabelFreeCBM(cfg)
        stats = lf.fit(
            train_X=[all_paths[i] for i in tr], train_y=y[tr],
            valid_X=[all_paths[i] for i in va], valid_y=y[va],
            concept_set=concept_set,
        )
        keep = np.asarray(lf.keep_mask, dtype=bool)
        C_true_kept = C_true[:, keep]
        if kept_report is None:
            kept_report = (int(keep.sum()), [CONCEPTS[i] for i in range(len(CONCEPTS)) if keep[i]])
        # calibrated in-distribution value per concept per true state = Koh 2020 exact:
        # absent -> 5th percentile of the training concept-activation distribution, present -> 95th.
        P_tr = lf.concept_proba([all_paths[i] for i in trainval])
        m = P_tr.shape[1]
        mu = np.array([[np.percentile(P_tr[:, j], 5), np.percentile(P_tr[:, j], 95)]
                       for j in range(m)])
        test_d = {"idx": test, "paths": [paths[i] for i in test]}
        ah, asf, ac, inc_hard, inc_soft, inc_cal, acc_det = run_seed(lf, C_true_kept, y, test_d, seed, mu)
        results.append((ah, asf, ac, inc_hard, inc_soft, inc_cal, acc_det))
        print(f"  seed {seed}: kept {int(keep.sum())}/7 | acc_det {acc_det*100:.1f} | "
              f"incacc hard {inc_hard*100:+.1f} | soft {inc_soft*100:+.1f} | calib {inc_cal*100:+.1f}")

    maj = max(y[test].mean(), 1 - y[test].mean())
    print(f"\n=== Derm7pt LFCBM (CLIP label-free) regime -- majority {maj*100:.1f}% ===")
    print(f"kept concepts ({kept_report[0]}/7): {', '.join(kept_report[1])}")
    print(f"{'method':16} " + " ".join(f"acc@{k}" for k in BUDGETS) + "   incacc(k=1,2,5)")
    for label, hi, ii in (("hard-all(paper)", 0, 3), ("soft-rest(Koh)", 1, 4), ("calib-indist(#3)", 2, 5)):
        accm = {k: np.mean([r[hi][k] for r in results]) for k in BUDGETS}
        inc = [r[ii] for r in results]
        row = " ".join(f"{accm[k]*100:5.1f}" for k in BUDGETS)
        print(f"{label:16} {row}    {np.mean(inc)*100:+5.1f} +/- {np.std(inc)*100:.1f}")

    inc_soft_m = np.mean([r[4] for r in results])
    inc_cal_m = np.mean([r[5] for r in results])
    print("\n=== VERDICT ===")
    print(f"standard (hard revealed, rest soft): incacc = {inc_soft_m*100:+.1f}%")
    print(f"calibrated in-distribution revealed: incacc = {inc_cal_m*100:+.1f}%")
    if inc_cal_m < 0:
        print("Backfire SURVIVES even calibrated in-distribution interventions -> real, "
              "content-driven effect on real data (matches paper's misaligned-concept mechanism).")
    else:
        print("Backfire VANISHES with calibrated interventions -> it was the extreme-value "
              "(out-of-distribution) input, not concept content. Mechanism needs re-examining.")


def _dev():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
