"""Why do some ProbCBM runs respond to a full intervention and others not? Inference on the saved grid models.

With every concept intervened on, ProbCBM's prediction no longer depends on the image: each concept becomes its
learned "present" or "absent" prototype (concept_vectors, L2-normalized), the label head maps the 7 prototypes to a
class embedding, and the class is the nearest learned class prototype (class_mean, scaled distance). So a run
that ignores a full intervention must differ in these parts. For each seed (true_concepts, sparse rule) this
reports, on the test robots with every concept set to its true value:
  * per concept: cosine similarity of its present and absent prototypes (1 = indistinguishable);
  * per concept: share of predictions that change when that one concept is flipped (sensitivity of the head);
  * spread of the predicted Glorp probability, the learned distance scale, and the distance between class means.

Run from a code checkout of `grid-seeded-lfcbm` with the cem package:
    cd <checkout> && PYTHONPATH=. python <repo>/scripts/paper/measure_probcbm_heads.py --pipeline-root <root> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from concept_benchmark.ext.fileutils import load
from measure_intervention_response import model_and_data_files, set_cpu
from measure_probcbm_response import find_official


def class_proba(official, concepts: torch.Tensor) -> torch.Tensor:
    """Class probabilities when every concept is replaced by its prototype (as in the intervention replay)."""
    protos = F.normalize(official.concept_vectors, p=2, dim=-1)  # [2, C, D]: 0 = absent, 1 = present
    emb = concepts.unsqueeze(-1) * protos[1].unsqueeze(0) + (1 - concepts).unsqueeze(-1) * protos[0].unsqueeze(0)
    emb = emb.reshape(emb.shape[0], 1, -1)  # one sample per robot, concepts concatenated
    with torch.no_grad():
        class_emb = official.head(emb)
        dist = torch.sqrt(((class_emb.unsqueeze(1) - official.class_mean.unsqueeze(1).unsqueeze(0)) ** 2).mean(-1) + 1e-10)
        if getattr(official, "use_scale", False):
            dist = official.class_negative_scale * dist
        return F.softmax(-dist, dim=1).mean(dim=-1)


def measure(seed: int, root: Path) -> list[dict]:
    model_file, data_file = model_and_data_files("probcbm", "true", seed, root)
    data = load(data_file)
    model = load(model_file)
    set_cpu(model)
    official = find_official(model).cpu().eval()
    test = data.test
    truth = torch.as_tensor(np.asarray(test.C, dtype=np.float32))
    y = np.asarray(test.y).astype(int)
    names = list(test.concepts)

    proba = class_proba(official, truth).numpy()
    pred = proba.argmax(axis=1)
    protos = F.normalize(official.concept_vectors.detach(), p=2, dim=-1)
    scale = getattr(official, "class_negative_scale", None)
    class_mean = official.class_mean.detach()
    common = {
        "seed": seed,
        "acc_all_true": round(float((pred == y).mean()), 4),
        "majority_share": round(float((pred == np.bincount(y).argmax()).mean()), 4),
        "p_glorp_std": round(float(proba[:, np.bincount(y).argmax()].std()), 5),
        "scale": round(float(scale.detach().mean()), 4) if scale is not None else "",
        "class_mean_dist": round(float((class_mean[0] - class_mean[1]).norm()), 4),
    }
    rows = []
    for j, name in enumerate(names):
        flipped = truth.clone()
        flipped[:, j] = 1 - flipped[:, j]
        changed = float((class_proba(official, flipped).numpy().argmax(axis=1) != pred).mean())
        rows.append({**common, "concept": name,
                     "proto_cosine": round(float(F.cosine_similarity(protos[0, j], protos[1, j], dim=0)), 4),
                     "flip_changes": round(changed, 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1014,1015,1016,1017")
    args = ap.parse_args()
    rows = []
    for seed in map(int, args.seeds.split(",")):
        r = measure(seed, args.pipeline_root)
        rows += r
        c = r[0]
        print(f"seed {seed}: acc(all true) {c['acc_all_true']}  majority share {c['majority_share']}  "
              f"P(Glorp) std {c['p_glorp_std']}  scale {c['scale']}  class-mean dist {c['class_mean_dist']}", flush=True)
        for x in r:
            print(f"    {x['concept']:12} present/absent cosine {x['proto_cosine']:+.3f}   flip changes {x['flip_changes']:.3f}",
                  flush=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
