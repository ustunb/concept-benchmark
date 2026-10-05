"""Concept-source result on real data (Derm7pt, CUB-25) with the paper's intervention policy or the cost-per-concept one.

For each dataset we train a CBM on (a) annotated concepts and (b) label-free concepts (LF-CBM, CLIP scores), as in
derm7pt_concept_noise.py / derm7pt_lfcbm_regime.py / cub_standard_vs_clip.py, and intervene toward the ground-truth
concepts with experiments.kflip.KFlipInterventionStrategy, only if the chance of changing the prediction reaches 0.2.
`--policy paper` (default) is the policy of scripts/robot_pipeline.py: up to k of all concepts, and every concept of the
selected images at k = max. `--policy cost` (exploratory) asks only about concepts the model is unsure about
(0.01 < p < 0.99), each of which must raise that chance by more than 0.01. Images with more than 12 candidates (CUB) are
searched greedily from the first concept, see KFlipInterventionStrategy. Label-free
interventions set a concept's score to the 5th or 95th percentile of its training scores; unintervened scores stay
continuous. Each LF-CBM gets its own CLIP cache folder (the caches are read by name, so sharing one across bootstrap
seeds would reuse the first seed's features).

    ./venv/bin/python experiments/real_data_up_to_k.py --out results/_incoming/real_data_up_to_k
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.intervention import InterventionBatch, InterventionConfig  # noqa: E402
from experiments.kflip import KFlipInterventionStrategy  # noqa: E402
from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig  # noqa: E402
from experiments.models import ConceptBasedModel, FrontEndModel  # noqa: E402

THRESHOLD = 0.2


class _RowPredictor(FrontEndModel):
    """Label predictor for one image, seen through its candidate concepts only."""

    supports_aligned_concept_replay = True
    _kflip_fast_path = False

    def __init__(self, predict, base_row, candidates, value_one, value_zero) -> None:
        super().__init__()
        self.predict, self.base_row, self.candidates = predict, base_row, candidates
        self.value_one, self.value_zero = value_one, value_zero

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        raise AssertionError("KFlip should use aligned replay for this predictor.")

    def predict_proba_from_concepts(self, concepts, *, row_indices=None, baseline_concepts=None, intervention_mask=None):
        full = np.repeat(self.base_row[None, :], len(concepts), axis=0)
        if intervention_mask is not None:
            values = np.where(np.asarray(concepts) >= 0.5, self.value_one, self.value_zero)
            full[:, self.candidates] = np.where(intervention_mask, values, full[:, self.candidates])
        return self.predict(full)


def up_to_k_mask(predict, base, probs, k, value_one, value_zero, policy="paper") -> np.ndarray:
    """Concepts KFlip asks about for each image at budget k (rows of `base` are what the label predictor reads)."""
    n, m = base.shape
    mask = np.zeros((n, m), dtype=bool)
    if policy == "paper":  # as in scripts/robot_pipeline.py: k = max corrects every concept of the images selected
        strategy = KFlipInterventionStrategy(use_exact_k=k >= m)
    else:  # unsure concepts only, each must add more than 0.01
        strategy = KFlipInterventionStrategy(concept_cost=0.01, uncertainty_band=(0.01, 0.99), exhaustive_size=3)
    for i in range(n):
        p = probs[i]
        cand = np.arange(m) if policy == "paper" else np.nonzero((p > 0.01) & (p < 0.99))[0]
        if cand.size == 0:
            continue
        model = ConceptBasedModel(
            label_predictor=_RowPredictor(predict, base[i].astype(float), cand, value_one[cand], value_zero[cand])
        )
        batch = InterventionBatch(C_pred=p[cand][None, :].astype(np.float32), C_true=np.zeros((1, cand.size), dtype=np.float32))
        proposal = strategy.propose(
            model, batch, InterventionConfig(per_instance_budget=int(min(k, cand.size)), score_threshold=THRESHOLD, random_state=0)
        )
        mask[i, cand] = proposal.mask[0]
    return mask


def curve(predict, base, probs, truth, y, budgets, value_one, value_zero, policy="paper") -> list[dict]:
    """Accuracy at k = 0 and each budget; `truth` holds the 0/1 ground-truth concepts."""
    rows = [{"budget": 0, "accuracy": float((predict(base).argmax(1) == y).mean()), "intervened": 0, "concepts_per_image": 0.0,
             "largest_subset": 0}]
    revealed = np.where(truth >= 0.5, value_one[None, :], value_zero[None, :])
    for k in budgets:
        kk = base.shape[1] if k == "max" else int(k)
        mask = up_to_k_mask(predict, base, probs, kk, value_one, value_zero, policy)
        after = predict(np.where(mask, revealed, base)).argmax(1)
        picked = mask.any(axis=1)
        rows.append({"budget": k, "accuracy": float((after == y).mean()), "intervened": int(picked.sum()),
                     "concepts_per_image": float(mask.sum() / max(1, picked.sum())),
                     "largest_subset": int(mask.sum(axis=1).max())})
        print(f"    k={k}: accuracy {rows[-1]['accuracy']:.3f}, images {rows[-1]['intervened']}, "
              f"concepts per image {rows[-1]['concepts_per_image']:.2f}, largest subset {rows[-1]['largest_subset']}", flush=True)
    return rows


def detector_probabilities(F_train, A_train, F_all, seed) -> np.ndarray:
    """Per-concept logistic probes on CLIP features (constant concepts keep their constant)."""
    P = np.zeros((F_all.shape[0], A_train.shape[1]))
    for j in range(A_train.shape[1]):
        col = A_train[:, j]
        if col.min() == col.max():
            P[:, j] = col[0]
        else:
            P[:, j] = LogisticRegression(max_iter=1000, random_state=seed).fit(F_train, col).predict_proba(F_all)[:, 1]
    return P


def run_derm7pt(out: Path, root: Path | None, policy: str) -> list[dict]:
    import experiments.derm7pt_concept_noise as derm
    from experiments.derm7pt_concept_noise import clip_features, load_derm7pt
    from experiments.derm7pt_lfcbm_regime import CONCEPT_TEXTS, _dev

    if root is not None:  # dataset copied elsewhere: read it there and keep this run's CLIP features apart
        derm.DERM_ROOT, derm.META, derm.IMAGES = root, root / "meta" / "meta.csv", root / "images"
        derm.CACHE = out / "cache/derm7pt_clip_feats.npz"
        derm.CACHE.parent.mkdir(parents=True, exist_ok=True)

    budgets = [1, 2, 5, "max"]
    y, C_true, paths, train, valid, test = load_derm7pt()
    trainval = np.concatenate([train, valid])
    F, _ = clip_features(paths)
    results = []
    for seed in range(5):  # clinician-annotated concepts, bootstrap over the training set
        rng = np.random.default_rng(seed)
        boot = rng.choice(trainval, size=len(trainval), replace=True) if seed > 0 else trainval
        P = detector_probabilities(F[boot], C_true[boot], F, seed)[test]
        head = LogisticRegression(max_iter=1000, random_state=seed).fit(C_true[boot], y[boot])
        m = C_true.shape[1]
        rows = curve(head.predict_proba, (P >= 0.5).astype(float), P, C_true[test], y[test], budgets, np.ones(m), np.zeros(m), policy)
        results += [{"dataset": "derm7pt", "concepts": "clinician", "seed": seed, **r} for r in rows]
        print("derm7pt clinician", seed, [round(100 * r["accuracy"], 1) for r in rows], flush=True)
    concept_set = LFConceptSet(keys=[f"c{i}" for i in range(len(CONCEPT_TEXTS))], texts=CONCEPT_TEXTS)
    for seed in range(3):  # label-free concepts
        rng = np.random.default_rng(seed)
        boot = rng.choice(trainval, size=len(trainval), replace=True) if seed > 0 else trainval
        cut = int(0.8 * len(boot))
        tr, va = boot[:cut], boot[cut:]
        lf = LabelFreeCBM(LFTrainingConfig(device=_dev(), seed=seed, cache_dir=out / f"cache/derm7pt_seed{seed}"))
        lf.fit(train_X=[paths[i] for i in tr], train_y=y[tr], valid_X=[paths[i] for i in va], valid_y=y[va], concept_set=concept_set)
        keep = np.asarray(lf.keep_mask, dtype=bool)
        P_tr = lf.concept_proba([paths[i] for i in trainval])
        P_te = lf.concept_proba([paths[i] for i in test])
        rows = curve(lf.predict_from_probs, P_te, P_te, C_true[test][:, keep], y[test], budgets,
                     np.percentile(P_tr, 95, axis=0), np.percentile(P_tr, 5, axis=0), policy)
        results += [{"dataset": "derm7pt", "concepts": "label_free", "seed": seed, **r} for r in rows]
        print("derm7pt label-free", seed, f"kept {int(keep.sum())}/7", [round(100 * r["accuracy"], 1) for r in rows], flush=True)
    return results


def run_cub(out: Path, root: Path | None, policy: str) -> list[dict]:
    import experiments.cub_lfcbm_regime as cub
    from experiments.cub_lfcbm_regime import N_CLASSES, _dev, load_cub_subset

    if root is not None:
        cub.CUB, cub.IMG_ROOT = root, root / "images"
    from experiments.cub_standard_vs_clip import FEAT_CACHE, HEAD, SEED
    from experiments.lfcbm import _CLIPEncoder

    budgets = [1, 10, "max"]
    paths, y, C_true, is_train, texts = load_cub_subset()
    rng = np.random.default_rng(SEED)
    tr_all, te = np.where(is_train)[0], np.where(~is_train)[0]
    perm = rng.permutation(tr_all)
    cut = int(0.8 * len(perm))
    fit, va = perm[:cut], perm[cut:]
    trv = np.r_[fit, va]
    F = np.load(FEAT_CACHE)["F"] if FEAT_CACHE.exists() else _CLIPEncoder("ViT-B-32", "laion2b_s34b_b79k", _dev()).encode_images(paths, batch_size=64)
    results = []
    # ground-truth attributes: probes on CLIP features, MLP head trained on the true attributes
    P = detector_probabilities(F[trv], C_true[trv], F, SEED)[te]
    head = MLPClassifier(**HEAD).fit(C_true[trv].astype(float), y[trv])
    m = C_true.shape[1]
    rows = curve(head.predict_proba, (P >= 0.5).astype(float), P, C_true[te], y[te], budgets, np.ones(m), np.zeros(m), policy)
    results += [{"dataset": f"cub{N_CLASSES}", "concepts": "ground_truth", "seed": SEED, **r} for r in rows]
    print("cub ground-truth", [round(100 * r["accuracy"], 1) for r in rows], flush=True)
    # label-free: LF-CBM concept scores, MLP head on the scores
    cset = LFConceptSet(keys=[f"a{i}" for i in range(len(texts))], texts=texts)
    lf = LabelFreeCBM(LFTrainingConfig(device=_dev(), seed=SEED, cache_dir=out / "cache/cub"))
    lf.fit(train_X=[paths[i] for i in fit], train_y=y[fit], valid_X=[paths[i] for i in va], valid_y=y[va], concept_set=cset)
    keep = np.asarray(lf.keep_mask, dtype=bool)
    S = lf.concept_proba(paths)
    head = MLPClassifier(**HEAD).fit(S[trv], y[trv])
    rows = curve(head.predict_proba, S[te], S[te], C_true[te][:, keep], y[te], budgets,
                 np.percentile(S[trv], 95, axis=0), np.percentile(S[trv], 5, axis=0), policy)
    results += [{"dataset": f"cub{N_CLASSES}", "concepts": "label_free", "seed": SEED, **r} for r in rows]
    print("cub label-free", f"kept {int(keep.sum())}/{len(texts)}", [round(100 * r["accuracy"], 1) for r in rows], flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["derm7pt", "cub"])
    ap.add_argument("--policy", choices=["paper", "cost"], default="paper")
    ap.add_argument("--derm-root", type=Path, default=None, help="Derm7pt folder (default ~/Downloads/Derm7pt)")
    ap.add_argument("--cub-root", type=Path, default=None, help="CUB_200_2011 folder holding images/ and attributes/")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        rows = (run_derm7pt(args.out, args.derm_root, args.policy) if name == "derm7pt"
                else run_cub(args.out, args.cub_root, args.policy))
        with (args.out / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out / f'{name}.csv'}", flush=True)


if __name__ == "__main__":
    main()
