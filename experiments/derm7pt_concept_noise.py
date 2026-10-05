"""Reproduce the paper's *On Concept Noise* finding on a REAL concept-annotated
dataset (Derm7pt).

Setup mirrors the synthetic robot experiment: a sequential CBM (concept detector
g(x)->C, then a label head f(C)->y trained on *predicted* concepts) with
test-time concept interventions.  Two arms share a frozen CLIP backbone and
differ ONLY in the source of the concept annotation:

    human : the 7-point dermoscopy checklist (clinician annotations)
    clip  : CLIP zero-shot labels for the same 7 concepts (machine annotation)

For each arm we report accuracy_gain(k) = acc_intervened(k) - acc(0) and
incacc = mean gain over k in {1,2,5} (matching robot_pipeline.py's metric).

Finding reproduces if the human arm's incacc > 0 (interventions help) while the
clip arm's incacc <= that (interventions do not help / backfire).

Run:  python experiments/derm7pt_concept_noise.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# reuse the paper's CLIP feature extractor
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lfcbm import _CLIPEncoder  # noqa: E402

DERM_ROOT = Path(os.path.expanduser("~/Downloads/Derm7pt"))
META = DERM_ROOT / "meta" / "meta.csv"
IMAGES = DERM_ROOT / "images"
MODEL_NAME, PRETRAINED = "ViT-B-32-quickgelu", "openai"
CACHE = Path(__file__).resolve().parent / f"_derm7pt_clip_feats_{MODEL_NAME}.npz"

CONCEPTS = [
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
]


def encode_concept(name, v):
    """Binarize to the melanoma-indicating (atypical/irregular) value per the
    7-point checklist -- NOT mere presence, which would conflate benign 'typical'
    variants with malignant 'atypical' ones and destroy the signal."""
    v = v.strip().lower()
    if name == "pigment_network":
        return int(v == "atypical")
    if name == "streaks":
        return int(v == "irregular")
    if name == "pigmentation":
        return int("irregular" in v)
    if name == "regression_structures":
        return int(v != "absent")
    if name == "dots_and_globules":
        return int(v == "irregular")
    if name == "blue_whitish_veil":
        return int(v == "present")
    if name == "vascular_structures":
        return int(v in {"dotted", "linear irregular"})
    raise ValueError(name)


# zero-shot text prompts (atypical variant vs not) for the CLIP machine annotator
_PFX = "a dermoscopy image of a melanocytic skin lesion "
PROMPTS = {
    "pigment_network": (_PFX + "with an atypical pigment network", _PFX + "without an atypical pigment network"),
    "streaks": (_PFX + "with irregular streaks", _PFX + "without irregular streaks"),
    "pigmentation": (_PFX + "with irregular pigmentation", _PFX + "with regular or absent pigmentation"),
    "regression_structures": (_PFX + "with regression structures", _PFX + "without regression structures"),
    "dots_and_globules": (_PFX + "with irregular dots and globules", _PFX + "without irregular dots and globules"),
    "blue_whitish_veil": (_PFX + "with a blue-whitish veil", _PFX + "without a blue-whitish veil"),
    "vascular_structures": (_PFX + "with atypical vascular structures", _PFX + "without atypical vascular structures"),
}

BUDGETS = [0, 1, 2, 3, 5, 7]
INCACC_KS = [1, 2, 5]
SEEDS = [0, 1, 2, 3, 4]


def load_derm7pt():
    rows = list(csv.DictReader(open(META)))
    keep, y, C_true, paths = [], [], [], []
    for i, r in enumerate(rows):
        d = r["diagnosis"].lower()
        if "melanoma" in d:
            lab = 1
        elif "nevus" in d:
            lab = 0
        else:
            continue
        keep.append(i)
        y.append(lab)
        C_true.append([encode_concept(c, r[c]) for c in CONCEPTS])
        paths.append(str(IMAGES / r["derm"]))
    idx_of_row = {row: k for k, row in enumerate(keep)}  # meta row -> kept position

    def read_split(name):
        vals = [int(r["indexes"]) for r in csv.DictReader(open(DERM_ROOT / "meta" / name))]
        return [idx_of_row[v] for v in vals if v in idx_of_row]

    train = read_split("train_indexes.csv")
    valid = read_split("valid_indexes.csv")
    test = read_split("test_indexes.csv")
    return (np.array(y), np.array(C_true, dtype=int), paths,
            np.array(train), np.array(valid), np.array(test))


def clip_features(paths):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if len(z["paths"]) == len(paths) and list(z["paths"]) == list(paths):
            return z["img"], z["prompts"].item()
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] encoding {len(paths)} images on {dev} ({MODEL_NAME}/{PRETRAINED})...")
    enc = _CLIPEncoder(MODEL_NAME, PRETRAINED, dev)
    img = enc.encode_images(paths, batch_size=64)
    prompt_feats = {}
    for c, (pos, neg) in PROMPTS.items():
        tp = enc.encode_texts([pos]); tn = enc.encode_texts([neg])
        prompt_feats[c] = (tp[0], tn[0])
    np.savez(CACHE, img=img, prompts=np.array(prompt_feats, dtype=object), paths=np.array(paths, dtype=object))
    return img, prompt_feats


def clip_concepts(img, prompt_feats, train):
    """Binary machine annotation: threshold pos-vs-neg similarity at train median."""
    C = np.zeros((img.shape[0], len(CONCEPTS)), dtype=int)
    for j, c in enumerate(CONCEPTS):
        tp, tn = prompt_feats[c]
        score = img @ tp - img @ tn
        thr = np.median(score[train])
        C[:, j] = (score > thr).astype(int)
    return C


def fit_detector(F_train, A_train, F_all, seed):
    """Per-concept probe predicting the arm's concept annotation from CLIP feats."""
    Chat = np.zeros((F_all.shape[0], A_train.shape[1]), dtype=int)
    for j in range(A_train.shape[1]):
        col = A_train[:, j]
        if col.min() == col.max():
            Chat[:, j] = col[0]
            continue
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(F_train, col)
        Chat[:, j] = clf.predict(F_all)
    return Chat


def intervene_greedy(head, c_pred, a_src, y, k):
    """Per-instance: reveal top-k concepts by responsiveness (flip effect), then predict."""
    if k == 0:
        return accuracy_score(y, head.predict(c_pred)), balanced_accuracy_score(y, head.predict(c_pred))
    n, m = c_pred.shape
    out = c_pred.copy()
    for i in range(n):
        cur = out[i].copy().astype(float)
        for _ in range(k):
            base_p = head.predict_proba(cur.reshape(1, -1))[0]
            base_cls = int(base_p.argmax())
            best_j, best_delta = -1, -1.0
            for j in range(m):
                if cur[j] == a_src[i, j]:
                    continue  # revealing changes nothing
                trial = cur.copy(); trial[j] = a_src[i, j]
                p = head.predict_proba(trial.reshape(1, -1))[0]
                delta = abs(p[base_cls] - base_p[base_cls])
                if delta > best_delta:
                    best_delta, best_j = delta, j
            if best_j < 0:
                break
            cur[best_j] = a_src[i, best_j]
        out[i] = cur.round().astype(int)
    return accuracy_score(y, head.predict(out)), balanced_accuracy_score(y, head.predict(out))


def run_arm(name, A, F, y, train, test, seed):
    """Clean independent CBM built entirely from ONE concept-annotation source A
    (human clinician labels, or CLIP machine labels). Detector predicts A; label
    head trained on A; standard responsiveness intervention reveals A's concepts.
    Nothing about the intervention differs between arms -- only the source A."""
    rng = np.random.default_rng(seed)
    boot = rng.choice(train, size=len(train), replace=True) if seed > 0 else train
    Chat = fit_detector(F[boot], A[boot], F, seed)          # detector g(x)->C predicts source A
    head = LogisticRegression(max_iter=1000, random_state=seed)
    head.fit(A[boot], y[boot])                              # head f(A)->y trained on source A
    accs = {}
    for k in BUDGETS:
        accs[k] = intervene_greedy(head, Chat[test], A[test], y[test], k)[0]
    gains = {k: accs[k] - accs[0] for k in BUDGETS}
    incacc = float(np.mean([gains[k] for k in INCACC_KS]))
    ceiling = accs[7]                                       # accuracy under perfect source concepts
    return accs, gains, incacc, ceiling


def main():
    y, C_true, paths, train, valid, test = load_derm7pt()
    trainval = np.concatenate([train, valid])  # more data for the detector/head
    print(f"[data] mel={int(y.sum())} nevus={int((1-y).sum())} | "
          f"train+val={len(trainval)} test={len(test)}")
    img, prompt_feats = clip_features(paths)
    C_clip = clip_concepts(img, prompt_feats, trainval)

    maj = max(y[test].mean(), 1 - y[test].mean())
    arms = {"human": C_true, "clip": C_clip}
    results = {a: [] for a in arms}
    for seed in SEEDS:
        for a, A in arms.items():
            accs, gains, incacc, ceiling = run_arm(a, A, img, y, trainval, test, seed)
            results[a].append((accs, gains, incacc, ceiling))

    print(f"\n=== Derm7pt: interventions under human vs machine (CLIP) concepts "
          f"(majority baseline {maj*100:.1f}%) ===")
    print(f"{'arm':6} | " + " ".join(f"acc@{k}" for k in BUDGETS) + " |  incacc(k=1,2,5)  ceiling")
    for a in arms:
        accm = {k: np.mean([r[0][k] for r in results[a]]) for k in BUDGETS}
        inc = [r[2] for r in results[a]]
        ceil = [r[3] for r in results[a]]
        row = " ".join(f"{accm[k]*100:5.1f}" for k in BUDGETS)
        print(f"{a:6} | {row} |  {np.mean(inc)*100:+5.1f} +/- {np.std(inc)*100:.1f}   "
              f"{np.mean(ceil)*100:.1f}")

    hum = np.mean([r[2] for r in results["human"]])
    clp = np.mean([r[2] for r in results["clip"]])
    hum_c = np.mean([r[3] for r in results["human"]])
    clp_c = np.mean([r[3] for r in results["clip"]])
    print("\n=== VERDICT ===")
    print(f"human-concept interventions:  incacc = {hum*100:+.1f}%   (perfect-concept ceiling {hum_c*100:.1f}%)")
    print(f"CLIP-concept  interventions:  incacc = {clp*100:+.1f}%   (perfect-concept ceiling {clp_c*100:.1f}%)")
    if hum > 0 and clp < hum:
        print("REPRODUCED: interventions help under human concepts but not under "
              "machine concepts (backfire/no-benefit) -- the paper's Concept-Noise "
              "finding holds on real data.")
    else:
        print("NOT reproduced as expected; inspect per-k table above.")


if __name__ == "__main__":
    main()
