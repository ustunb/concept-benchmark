"""Why does ProbCBM ignore single interventions? Inference-only checks on the saved grid models (sparse rule).

The official ProbCBM training step (cem/models/probcbm.py, `torch.rand_like(gt[..., :1, :1]) < intervention_prob`)
draws one coin per training example, so the label predictor sees either every concept replaced by its true value
or none. If that is why single interventions do nothing, accuracy should stay flat as we intervene on more
concepts and jump only when every concept is set; the CBM, whose label predictor is trained on true concept
values, should climb steadily. For each seed and concept set this script reports:
  * the training settings stored in the model (training mode, intervention probability, inference samples);
  * detection accuracy and AUC of the rule concepts (to check that chance-level detection is real);
  * accuracy and share of changed predictions after intervening on m random concepts per robot, m = 0..all,
    for ProbCBM (as the pipeline replays it, with sampled embeddings, and with mean embeddings as the official
    evaluation does) and for the CBM;
  * human_concepts only: accuracy after intervening on every concept, on robots whose foot subtype is listed in
    the concept set vs not.

Run from a code checkout of `grid-seeded-lfcbm` (the models are pickled with that code):
    cd <checkout> && PYTHONPATH=. python <repo>/scripts/paper/measure_probcbm_response.py --images <robot_images> \
        --pipeline-root <root with run_s<seed>/results> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from concept_benchmark.ext.fileutils import load
from measure_intervention_response import label_proba, model_and_data_files, set_cpu

RULE_CONCEPTS = ("mouth_type", "has_knees", "foot_shape")


def find_cache(obj, seen=None):
    """The ProbCBM prediction cache (holds the sampled-embedding inputs) somewhere inside the pickled model."""
    seen = seen if seen is not None else set()
    if id(obj) in seen or not hasattr(obj, "__dict__"):
        return None
    seen.add(id(obj))
    if hasattr(obj, "probcbm_pred_logsigma"):
        return obj
    for value in vars(obj).values():
        found = find_cache(value, seen)
        if found is not None:
            return found
    return None


def find_official(obj, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen or not hasattr(obj, "__dict__"):
        return None
    seen.add(id(obj))
    if type(obj).__name__ == "ProbCBM":
        return obj
    for value in vars(obj).values():
        found = find_official(value, seen)
        if found is not None:
            return found
    return None


def response_curve(model, arch, probs, truth, y, pred0, rng_seed=0):
    """Accuracy and changed share after intervening on m random concepts per robot, m = 0..all."""
    n, c = probs.shape
    order = np.argsort(np.random.default_rng(rng_seed).random((n, c)), axis=1)  # random concept order per robot
    out = []
    for m in range(c + 1):
        mask = np.zeros((n, c), dtype=bool)
        np.put_along_axis(mask, order[:, :m], True, axis=1)
        pred = label_proba(model, arch, np.where(mask, truth, probs), probs, mask if m else None).argmax(axis=1)
        out.append((m, float((pred == y).mean()), float((pred != pred0).mean())))
    return out


def measure(arch, concepts_name, seed, images, pipeline_root, listed):
    model_file, data_file = model_and_data_files(arch, concepts_name, seed, pipeline_root)
    data = load(data_file)
    test = data.test
    test.base_dir = Path(images)
    model = load(model_file)
    set_cpu(model)
    probs = model.concept_detector.predict_proba(test)
    truth = np.asarray(test.C, dtype=float)
    y = np.asarray(test.y).astype(int)
    names = list(test.concepts)
    pred0 = label_proba(model, arch, probs, probs, None).argmax(axis=1)
    common = {"arch": arch, "concepts": concepts_name, "seed": seed}
    rows = []

    if arch == "probcbm":
        official = find_official(model)
        settings = {k: getattr(official, k, None) for k in ("train_class_mode", "intervention_prob", "n_samples_inference")}
        print(f"  probcbm {concepts_name} {seed} settings: {settings}", flush=True)
        for j, name in enumerate(names):
            if name in RULE_CONCEPTS:
                t = truth[:, j] >= 0.5
                rows.append({**common, "measure": "detection", "mode": "", "concept": name,
                             "m": "", "value": round(float(((probs[:, j] >= 0.5) == t).mean()), 4),
                             "extra": round(float(roc_auc_score(t, probs[:, j])), 4)})

    modes = ["sampled", "mean"] if arch == "probcbm" else [""]
    cache = find_cache(model) if arch == "probcbm" else None
    logsigma = cache.probcbm_pred_logsigma if cache is not None else None
    for mode in modes:
        if cache is not None:
            cache.probcbm_pred_logsigma = logsigma if mode == "sampled" else None  # None -> mean embeddings
        for m, acc, changed in response_curve(model, arch, probs, truth, y, pred0):
            rows.append({**common, "measure": "intervene_m", "mode": mode, "concept": "", "m": m,
                         "value": round(acc, 4), "extra": round(changed, 4)})
        if concepts_name == "human":
            catalog = data.meta["catalog_df"]
            idx = catalog.index.get_indexer(test.meta["df_indices"])
            subtype = (catalog["foot_shape"].astype(str) + "_" + catalog["foot_shape_subtype"].astype(str)).to_numpy()[idx]
            is_listed = np.isin(subtype, list(listed))
            mask = np.ones_like(probs, dtype=bool)
            pred = label_proba(model, arch, truth, probs, mask).argmax(axis=1)
            for group, sel in (("listed", is_listed), ("unlisted", ~is_listed)):
                rows.append({**common, "measure": "all_concepts_by_subtype", "mode": mode, "concept": group, "m": "",
                             "value": round(float((pred[sel] == y[sel]).mean()), 4),
                             "extra": round(float((pred0[sel] == y[sel]).mean()), 4)})
    if cache is not None:
        cache.probcbm_pred_logsigma = logsigma
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--pipeline-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1014,1015,1016,1017")
    ap.add_argument("--archs", default="probcbm,cbm")
    args = ap.parse_args()
    fh = args.out.open("w", newline="")
    writer = None
    for seed in map(int, args.seeds.split(",")):
        human = load(model_and_data_files("cbm", "human", seed, args.pipeline_root)[1])
        listed = {c.removeprefix("foot_shape_") for c in human.train.concepts if c.startswith("foot_shape_")}
        for arch in args.archs.split(","):
            for concepts_name in ("true", "human"):
                rows = measure(arch, concepts_name, seed, args.images, args.pipeline_root, listed)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                fh.flush()
                curve = [r for r in rows if r["measure"] == "intervene_m" and r["mode"] in ("", "sampled")]
                print(f"{arch:8} {concepts_name:5} {seed}: acc by m " +
                      " ".join(f"{r['value']:.3f}" for r in curve), flush=True)
    fh.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
