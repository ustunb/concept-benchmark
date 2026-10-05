"""Image-only vs joint label accuracy of one ECBM model on listed vs unlisted foot subtypes (balanced rule test set).

PYTHONPATH=. python scripts/paper/eval_ecbm_image_only.py --model <file> --seed 1015 --data-root <runs> --images <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from concept_benchmark.ext.fileutils import load
from diagnose_incomplete_concepts import dataset
from experiments.baselines.ecbm import _infer_labels_and_concepts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    args = ap.parse_args()
    human = dataset(args.data_root, args.seed, "subconcept", args.images)
    foot_cols = [j for j, n in enumerate(human.concepts) if n.startswith("foot_shape")]
    u = np.asarray(human.C)[:, foot_cols].sum(axis=1) == 0
    det = load(args.model).concept_detector._owner
    net = det._require_official_model()
    device = det._inference_device()
    net.to(device).eval()
    with torch.no_grad():
        feats = [
            net.extract_features(x.to(device).float())
            for x, _, _ in human.loader(shuffle=False, **det._loader_kwargs())
        ]
        features = torch.cat(feats)
        xy, _, _ = net.training_energies(
            features,
            torch.zeros(len(features), net.n_concepts, device=device),
            augment=False,
        )
    image_only = xy.argmin(dim=1).cpu().numpy()
    joint = (
        _infer_labels_and_concepts(net, features, det._inference_batch_size())[0]
        .argmax(dim=1)
        .cpu()
        .numpy()
    )
    y = np.asarray(human.y).astype(int)
    print(
        "RESULT",
        {
            "seed": args.seed,
            "lambdas": (net.lambda_xy, net.lambda_xc, net.lambda_cy),
            "image_only": round(float((image_only == y).mean()), 3),
            "image_only_listed": round(float((image_only[~u] == y[~u]).mean()), 3),
            "image_only_unlisted": round(float((image_only[u] == y[u]).mean()), 3),
            "joint": round(float((joint == y).mean()), 3),
            "glorp_share": round(float(image_only.mean()), 3),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
