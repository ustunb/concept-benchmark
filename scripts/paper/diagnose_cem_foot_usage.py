"""Does CEM's label predictor use the foot information its inputs carry for robots with an unlisted foot subtype?

Balanced rule, human_concepts, installed CEM per seed. Test robots are split into listed and unlisted foot subtypes.
  1. Glorp-logit gap: mean label logit (Glorp minus Drent) for pointy minus flat feet, on listed and on unlisted
     robots, before interventions. The rule adds +8 to the score for a pointy foot.
  2. Probe transfer: a logistic probe for pointy vs flat on the six FootShape embeddings, fit on the training robots
     (all listed) and applied to the unlisted test robots; for reference, the 5-fold probe within the unlisted robots.

    python scripts/paper/diagnose_cem_foot_usage.py --models <dir> --data-root <runs> --images <dir> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from _common import BALANCED_TAG
from concept_benchmark.ext.fileutils import load

SEEDS = (1014, 1015, 1016, 1017)


def split(root: Path, seed: int, preset: str, name: str, images: Path):
    folder = "foot_subtypes" if preset == "subconcept" else "ground_truth"
    d = load(
        root
        / f"final_d030e2/run_d0.30e2_s{seed}_{folder}/results/robot_image_4_{preset}_seed{seed}.data"
    )
    s = getattr(d, name)
    s.base_dir = images
    return s


def embeddings(det, data, foot_cols):
    label_probs, concept_probs, cache = det._run_official_model(data)
    p = concept_probs[:, foot_cols]
    pos, neg = (
        cache.pos_embeddings.numpy()[:, foot_cols],
        cache.neg_embeddings.numpy()[:, foot_cols],
    )
    mixed = pos * p[..., None] + neg * (1 - p[..., None])
    model = det._require_official_model().cpu().eval()
    allp = concept_probs
    bottleneck = cache.pos_embeddings * torch.as_tensor(allp).unsqueeze(
        -1
    ) + cache.neg_embeddings * (1 - torch.as_tensor(allp).unsqueeze(-1))
    with torch.no_grad():
        logits = model.c2y_model(torch.flatten(bottleneck, start_dim=1).float()).numpy()
    margin = logits[:, 1] - logits[:, 0] if logits.shape[1] == 2 else logits[:, 0]
    return mixed.reshape(len(mixed), -1), margin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for seed in SEEDS:
        det = load(
            next(args.models.glob(f"{BALANCED_TAG}__subconcept_cem_seed{seed}__*"))
        ).concept_detector._owner
        sets = {}
        for name in ("train", "test"):
            human = split(args.data_root, seed, "subconcept", name, args.images)
            true = split(args.data_root, seed, "ideal", name, args.images)
            foot_cols = [
                j for j, n in enumerate(human.concepts) if n.startswith("foot_shape")
            ]
            unlisted = np.asarray(human.C)[:, foot_cols].sum(axis=1) == 0
            pointy = (
                np.asarray(true.C)[:, list(true.concepts).index("foot_shape")] >= 0.5
            )
            emb, margin = embeddings(det, human, foot_cols)
            sets[name] = (emb, margin, unlisted, pointy)
        emb, margin, u, pointy = sets["test"]
        tr_emb, _, tr_u, tr_pointy = sets["train"]
        probe = LogisticRegression(max_iter=3000).fit(tr_emb, tr_pointy)
        rows.append(
            {
                "seed": seed,
                "train_unlisted_share": round(float(tr_u.mean()), 3),
                "logit_gap_listed": round(
                    float(margin[~u & pointy].mean() - margin[~u & ~pointy].mean()), 2
                ),
                "logit_gap_unlisted": round(
                    float(margin[u & pointy].mean() - margin[u & ~pointy].mean()), 2
                ),
                "probe_from_train_on_unlisted": round(
                    float((probe.predict(emb[u]) == pointy[u]).mean()), 3
                ),
                "probe_from_train_on_listed": round(
                    float((probe.predict(emb[~u]) == pointy[~u]).mean()), 3
                ),
                "probe_within_unlisted_cv": round(
                    float(
                        cross_val_score(
                            LogisticRegression(max_iter=3000), emb[u], pointy[u], cv=5
                        ).mean()
                    ),
                    3,
                ),
            }
        )
        print(rows[-1], flush=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
