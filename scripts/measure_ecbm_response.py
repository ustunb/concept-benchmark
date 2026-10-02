"""Why does a partial intervention lower ECBM's accuracy? Inference on trained models (true_concepts, sparse rule).

Without interventions, ECBM infers the label with energy weights 1/1/0.01 (image->label dominates); after any
intervention, the authors' procedure re-infers the label from the concept->label energy alone (0/0/3), using the
intervened concepts plus the model's own predictions for the rest. For each seed this reports, on the test robots:
  * accuracy of the image->label energy alone and of the usual prediction (no interventions);
  * per-concept detection accuracy and AUC of the inferred concepts;
  * "switch only": the intervention procedure with no concept changed (concept->label on the model's own concepts);
  * accuracy after intervening on m random concepts per robot, m = 0..all.

    cd <ecbm-original checkout> && PYTHONPATH=. python scripts/measure_ecbm_response.py --root <dir with run_ecbm_s<seed>> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from concept_benchmark.ext.fileutils import load
from experiments.baselines import ecbm as e


def find_ecbm(obj, seen=None):
    """The ECBMBenchmarkModel inside the pipeline's wrappers."""
    seen = seen if seen is not None else set()
    if isinstance(obj, e.ECBMBenchmarkModel):
        return obj
    if id(obj) in seen or not hasattr(obj, "__dict__"):
        return None
    seen.add(id(obj))
    for value in vars(obj).values():
        found = find_ecbm(value, seen)
        if found is not None:
            return found
    return None


def measure(seed: int, root: Path) -> list[dict]:
    res = root / f"run_ecbm_s{seed}" / "results"
    data = load(res / f"robot_image_4_ideal_seed{seed}.data")
    model = load(res / f"robot_image_stochastic_4_ideal_ecbm_seed{seed}_labels12345.model")
    model = find_ecbm(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval_config["device"] = str(device)
    test = data.test
    y = np.asarray(test.y).astype(int)
    truth = np.asarray(test.C, dtype=np.float32)
    proba, c_prob = model.predict_proba(test, return_concepts=True)
    cache = model._prediction_cache
    net = model._require_official_model().to(device).eval()
    feats = cache.ecbm_features.to(device)
    logits = cache.ecbm_concept_logits.to(device)
    bs = model._inference_batch_size()
    acc = lambda p: round(float((p == y).mean()), 4)
    rows = [{"seed": seed, "measure": "usual_prediction", "m": 0, "value": acc(proba.argmax(1))}]
    with torch.no_grad():
        xy, _, _ = net.training_energies(feats, torch.as_tensor(truth, device=device), augment=False)
    rows.append({"seed": seed, "measure": "image_to_label_only", "m": 0, "value": acc(xy.argmin(1).cpu().numpy())})
    for j, name in enumerate(test.concepts):
        t = truth[:, j] >= 0.5
        rows.append({"seed": seed, "measure": f"detection_acc:{name}", "m": "", "value": round(float(((c_prob[:, j] >= 0.5) == t).mean()), 4)})
        rows.append({"seed": seed, "measure": f"detection_auc:{name}", "m": "", "value": round(float(roc_auc_score(t, c_prob[:, j])), 4)})
    n, c = truth.shape
    order = np.argsort(np.random.default_rng(0).random((n, c)), axis=1)
    t_truth = torch.as_tensor(truth, device=device)
    for m in range(c + 1):
        mask = np.zeros((n, c), dtype=bool)
        np.put_along_axis(mask, order[:, :m], True, axis=1)
        y_logits = e._intervene(net, feats, logits, t_truth, torch.as_tensor(mask, device=device), bs)
        name = "switch_only" if m == 0 else "intervene_random"
        rows.append({"seed": seed, "measure": name, "m": m, "value": acc(y_logits.argmax(1).cpu().numpy())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1014,1015,1016,1017")
    args = ap.parse_args()
    rows = []
    for seed in map(int, args.seeds.split(",")):
        r = measure(seed, args.root)
        rows += r
        for x in r:
            print(f"seed {seed}  {x['measure']:28} m={x['m']!s:2}  {x['value']}", flush=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
