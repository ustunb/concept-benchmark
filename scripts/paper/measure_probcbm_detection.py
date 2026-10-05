"""Does ProbCBM learn the rule concepts on the training robots but not generalize, or not learn them at all?

Concept detection accuracy and AUC on the training and test robots, true_concepts, for each seed's ProbCBM.
Good on training and chance on test = memorized; chance on both = never learned.

    cd <checkout> && PYTHONPATH=. python <repo>/scripts/paper/measure_probcbm_detection.py --pipeline-root <root> \
        --images <robot_images> --out <csv> [--seeds 1014,...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_intervention_response import model_and_data_files, set_cpu  # noqa: E402

from concept_benchmark.ext.fileutils import load  # noqa: E402

CONCEPTS = ("mouth_type", "has_knees", "foot_shape", "head_shape", "body_shape")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1014,1015,1016,1017")
    args = ap.parse_args()
    rows = []
    for seed in map(int, args.seeds.split(",")):
        model_file, data_file = model_and_data_files("probcbm", "true", seed, args.pipeline_root)
        data = load(data_file)
        model = load(model_file)
        set_cpu(model)
        for split in ("train", "test"):
            sample = getattr(data, split)
            sample.base_dir = args.images
            probs = model.concept_detector.predict_proba(sample)
            truth = np.asarray(sample.C) >= 0.5
            for j, name in enumerate(sample.concepts):
                if name in CONCEPTS:
                    rows.append({"seed": seed, "split": split, "concept": name,
                                 "accuracy": round(float(((probs[:, j] >= 0.5) == truth[:, j]).mean()), 4),
                                 "auc": round(float(roc_auc_score(truth[:, j], probs[:, j])), 4)})
        for r in rows[-2 * len(CONCEPTS):]:
            print(f"seed {seed} {r['split']:5} {r['concept']:11} acc {r['accuracy']:.3f}  AUC {r['auc']:.3f}", flush=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
