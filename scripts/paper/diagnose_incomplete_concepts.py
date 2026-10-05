"""Diagnostics on the balanced rule: how CEM and ECBM handle robots whose foot subtype is not listed.

Test robots split by whether their foot subtype is listed in human_concepts (unlisted = every FootShape concept 0).
  CEM:  can pointy vs flat be read off what CEM passes to its label predictor on unlisted robots? 5-fold logistic
        probes on (a) the FootShape concept probabilities only (what a CBM passes), (b) the mixed FootShape
        embeddings before interventions, (c) the embeddings after every concept is set to its true value; plus CEM's
        label accuracy on listed vs unlisted robots.
  ECBM: label accuracy of the image-only prediction (lowest x->y energy) vs the joint gradient inference, on listed
        vs unlisted robots, for human_concepts and true_concepts.
Runs where the CEM/ECBM dependencies and a GPU are available; models from models/robot/balanced.

    python scripts/paper/diagnose_incomplete_concepts.py --models <dir> --data-root <runs> --images <dir> --out <csv>
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
from experiments.baselines.ecbm import _infer_labels_and_concepts

SEEDS = (1014, 1015, 1016, 1017)


def dataset(root: Path, seed: int, preset: str, images: Path):
    folder = "foot_subtypes" if preset == "subconcept" else "ground_truth"
    d = load(
        root
        / f"final_d030e2/run_d0.30e2_s{seed}_{folder}/results/robot_image_4_{preset}_seed{seed}.data"
    )
    d.test.base_dir = images
    return d.test


def probe(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(len(x), -1)
    return float(cross_val_score(LogisticRegression(max_iter=2000), x, y, cv=5).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for seed in SEEDS:
        human = dataset(args.data_root, seed, "subconcept", args.images)
        true = dataset(args.data_root, seed, "ideal", args.images)
        assert list(map(str, human.inputs)) == list(map(str, true.inputs))
        foot_cols = [
            j for j, n in enumerate(human.concepts) if n.startswith("foot_shape")
        ]
        unlisted = np.asarray(human.C)[:, foot_cols].sum(axis=1) == 0
        pointy = np.asarray(true.C)[:, list(true.concepts).index("foot_shape")] >= 0.5
        y = np.asarray(human.y).astype(int)

        # CEM
        model = load(
            next(args.models.glob(f"{BALANCED_TAG}__subconcept_cem_seed{seed}__*"))
        )
        det = (
            model.concept_detector._owner
        )  # the official CEM/ECBM wrapper behind the adapter
        label_probs, concept_probs, cache = det._run_official_model(human)
        pred = (
            label_probs.argmax(axis=1)
            if label_probs.ndim == 2
            else (label_probs >= 0.5).astype(int)
        )
        p = concept_probs[:, foot_cols]
        pos, neg = (
            cache.pos_embeddings.numpy()[:, foot_cols],
            cache.neg_embeddings.numpy()[:, foot_cols],
        )
        mixed = pos * p[..., None] + neg * (1 - p[..., None])
        truth = np.asarray(human.C)[:, foot_cols]
        corrected = pos * truth[..., None] + neg * (1 - truth[..., None])
        u = unlisted
        rows.append(
            {
                "seed": seed,
                "arch": "cem",
                "unlisted_share": round(float(u.mean()), 3),
                "pointy_share_unlisted": round(float(pointy[u].mean()), 3),
                "probe_probs": round(probe(p[u], pointy[u]), 3),
                "probe_embeddings": round(probe(mixed[u], pointy[u]), 3),
                "probe_embeddings_corrected": round(probe(corrected[u], pointy[u]), 3),
                "acc_listed": round(float((pred[~u] == y[~u]).mean()), 3),
                "acc_unlisted": round(float((pred[u] == y[u]).mean()), 3),
            }
        )
        print(rows[-1], flush=True)

        # ECBM, human and true concepts
        for preset, data in (("subconcept", human), ("ideal", true)):
            model = load(
                next(args.models.glob(f"{BALANCED_TAG}__{preset}_ecbm_seed{seed}__*"))
            )
            det = (
                model.concept_detector._owner
            )  # the official CEM/ECBM wrapper behind the adapter
            net = det._require_official_model()
            device = det._inference_device()
            net.to(device).eval()
            feats = []
            with torch.no_grad():
                for batch_x, _, _ in data.loader(shuffle=False, **det._loader_kwargs()):
                    feats.append(net.extract_features(batch_x.to(device).float()))
                features = torch.cat(feats)
                c0 = torch.zeros(len(features), net.n_concepts, device=device)
                xy, _, _ = net.training_energies(features, c0, augment=False)
            image_only = xy.argmin(dim=1).cpu().numpy()
            y_logits, _ = _infer_labels_and_concepts(
                net, features, det._inference_batch_size()
            )
            joint = y_logits.argmax(dim=1).cpu().numpy()
            yy = np.asarray(data.y).astype(int)
            rows.append(
                {
                    "seed": seed,
                    "arch": f"ecbm_{preset}",
                    "acc_image_only_listed": round(
                        float((image_only[~u] == yy[~u]).mean()), 3
                    ),
                    "acc_image_only_unlisted": round(
                        float((image_only[u] == yy[u]).mean()), 3
                    ),
                    "acc_joint_listed": round(float((joint[~u] == yy[~u]).mean()), 3),
                    "acc_joint_unlisted": round(float((joint[u] == yy[u]).mean()), 3),
                    "acc_image_only": round(float((image_only == yy).mean()), 3),
                    "acc_joint": round(float((joint == yy).mean()), 3),
                    "predicted_glorp_share_joint": round(float(joint.mean()), 3),
                }
            )
            print(rows[-1], flush=True)
    keys = sorted(
        {k for r in rows for k in r}, key=lambda k: (k not in ("seed", "arch"), k)
    )
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
