# scripts/demo3_text_genstyle.py

from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd

from concept_benchmark.paths import pkg_dir, results_dir, data_dir
from concept_benchmark.synthetic.helper.textgen import create_synthetic_dataset
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.models import FrontEndModel
from concept_benchmark.metrics import calc_metric
from concept_benchmark.synthetic.helper.text_concept_detector import TextConceptDetector

settings = {
    "out_dir": str(results_dir / "demo3_text"),
    "seed": 0,
    "n_per_combo": 1,
    "use_llm": False,
    "label_model": "(row['body_shape']=='square') and (row['has_antennae']=='true')",
    "device": "cuda",
    "det_epochs": 6,
    "det_batch": 64,
    "det_model": "distilbert/distilbert-base-uncased",
    "concept_mode": "hard",
    "split_by_robot": True,
    "couple_train": True,
    "couple_pair": "body_shape=square,has_antennae=true",
    "conflate_concept": "body_shape=square",
    "apply_conflation_at_test": False,
    "noise_targets": "has_antennae=true",
    "noise_prob_train": 0.0,
    "noise_prob_test": 0.2,
    "train_on_detected_gt": False,
    "intervene_concepts": "body_shape=square,has_antennae=true",
    "intervention_k": 2,
    "intervene_scope": "errors",
    "select_by": "uncertainty",
    "coverages": "0.0,0.1,0.25,0.5,1.0",
    "tau": 0.5
}

ap = argparse.ArgumentParser(add_help=False)
for k, v in settings.items():
    ap.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
ap.set_defaults(**settings)
args, _ = ap.parse_known_args([])
S = vars(args)

rng = np.random.default_rng(S["seed"])
out_dir = Path(S["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

concept_cols = ["head_shape","body_shape","has_knees","has_elbows","foot_shape","has_antennae","ears_shape","mouth_type","hand_shape"]
tpl_path = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "Templates.txt"
templates = [ln.strip() for ln in tpl_path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

from itertools import product
domain = {
    "head_shape": ["square","round"],
    "body_shape": ["square","round"],
    "has_knees": ["false","true"],
    "has_elbows": ["false","true"],
    "foot_shape": ["flat_4sided","flat_5sided","flat_lshaped","pointy_3sided","pointy_4sided","pointy_6sided"],
    "has_antennae": ["false","true"],
    "ears_shape": ["square","triangle"],
    "mouth_type": ["closed","open"],
    "hand_shape": ["round_circle","wide_oval","tall_oval","edgy_square","edgy_triangle","edgy_trapezoid"]
}
rows = [dict(zip(concept_cols, vals)) for vals in product(*[domain[c] for c in concept_cols])]
def _label(r):
    return 'glorp' if eval(S["label_model"], {"__builtins__": {}}, {"row": r}) else 'drent'
catalog_df = pd.DataFrame(rows, columns=concept_cols)
catalog_df["label"] = catalog_df.apply(lambda sr: _label(sr.to_dict()), axis=1)

ds = create_synthetic_dataset(
    source=catalog_df,
    templates=templates,
    variants_per_row=int(S["n_per_combo"]),
    include_color=False,
    rng_seed=int(S["seed"]),
    concept_cols=concept_cols,
    label_col="label",
    label_map={"drent": 0, "glorp": 1},
    text_mode="semi",
    llm_provider=("gemini" if S["use_llm"] else ""),
    llm_model="gemini-1.5-flash",
    llm_user_prompt=""
)

if ds.cvindices is None or getattr(ds.validation, "n", 0) == 0 or getattr(ds, "test", None) is None:
    idx = getattr(getattr(ds, "_full", ds), "meta", {}).get("row_index", np.arange(len(ds)))
    base = np.unique(idx); rng.shuffle(base)
    n_folds = 5
    assign = {int(b): i % n_folds for i, b in enumerate(base)}
    fold_arr = np.array([assign[int(x)] for x in idx], dtype=int)
    ds.cvindices = ds.cvindices or {}
    ds.cvindices["by_robot"] = fold_arr
    ds.split("by_robot", fold_num_validation=0, fold_num_test=1)

train_ds, val_ds, test_ds = ds.training, ds.validation, ds.test

def _concept_names():
    return [str(n) for n in list(ds.concepts)]
names = _concept_names()

def _name_to_col(nm: str):
    nm = (nm or "").strip()
    if nm in names:
        return names.index(nm)
    base = nm.split("=", 1)[0]
    cands = [(i, n) for i, n in enumerate(names) if (n == base) or n.startswith(base + "=")]
    if not cands:
        raise KeyError(nm)
    if len(cands) == 1:
        return cands[0][0]
    if "=" in nm:
        for i, n in cands:
            if n == nm:
                return i
    prefer_vals = ("=true", "=present", "=yes", "=on", "=1")
    for i, n in cands:
        if any(v in n for v in prefer_vals):
            return i
    avoid_vals = ("=false", "=absent", "=no", "=0")
    for i, n in cands:
        if not any(v in n for v in avoid_vals):
            return i
    return cands[0][0]

if S["couple_train"]:
    a_nm, b_nm = [x.strip() for x in S["couple_pair"].split(",")]
    a_idx, b_idx = _name_to_col(a_nm), _name_to_col(b_nm)
    Ctr = train_ds.C
    a, b = Ctr[:, a_idx], Ctr[:, b_idx]
    keep = ~(((a == 1) & (b == 0)) | ((a == 0) & (b == 1)))
    idx_keep = np.where(keep)[0]
    train_ds = ConceptDatasetSample(
        X=np.array(train_ds.X, dtype=object)[idx_keep],
        C=train_ds.C[idx_keep],
        y=train_ds.y[idx_keep],
        meta={"concepts": names, "classes": [0,1], "data_type": "text", "observed_mask": np.ones((idx_keep.size, len(names)), dtype=int)}
    )

det = TextConceptDetector(
    epochs=int(S["det_epochs"]),
    batch_size=int(S["det_batch"]),
    model_name=S["det_model"],
    output_mode="hard",
    validate=True,
    device=S["device"]
)
det.fit(train_ds, val_ds)

def _soft(det, ds_):
    old = det.output_mode; det.output_mode = "soft"
    P = det.predict(ds_)
    det.output_mode = old
    return P.astype(np.float32)

C_train_true = train_ds.C.astype(np.float32)
C_val_true   = val_ds.C.astype(np.float32)
C_test_true  = test_ds.C.astype(np.float32)

C_train_pred = _soft(det, train_ds)
C_val_pred   = _soft(det, val_ds)
C_test_pred  = _soft(det, test_ds)

def _apply_noise(C, targets, p, seed):
    if p <= 0 or not targets:
        return C
    rng = np.random.default_rng(seed)
    J = [_name_to_col(t) for t in targets]
    C2 = C.copy()
    flips = rng.random((C.shape[0], len(J))) < float(p)
    for k, j in enumerate(J):
        C2[:, j] = np.where(flips[:, k], 1.0 - C2[:, j], C2[:, j])
    return C2

noise_targets = [s.strip() for s in S["noise_targets"].split(",") if s.strip()]
C_train_pred_noisy = _apply_noise(C_train_pred, noise_targets, S["noise_prob_train"], S["seed"])
C_test_pred_noisy  = _apply_noise(C_test_pred,  noise_targets, S["noise_prob_test"],  S["seed"]+1)

def _apply_conflation_train(C, y, concept_name):
    if not concept_name:
        return C
    j = _name_to_col(concept_name)
    C2 = C.copy()
    C2[:, j] = y.astype(np.float32)
    return C2

C_train_machine_full = _apply_conflation_train(C_train_pred_noisy, train_ds.y.astype(int), S["conflate_concept"])
if S["apply_conflation_at_test"]:
    j = _name_to_col(S["conflate_concept"])
    C_test_pred_noisy[:, j] = test_ds.y.astype(np.float32)

fe_gt = FrontEndModel()
Xtr_gt = C_train_pred if S["train_on_detected_gt"] else C_train_true
fe_gt.fit(Xtr_gt, train_ds.y.astype(int))

fe_mc = FrontEndModel()
J_machine_names = ["body_shape=square", "has_antennae"]
J_machine = [_name_to_col(nm) for nm in J_machine_names]
fe_mc.fit(C_train_machine_full[:, J_machine], train_ds.y.astype(int))

def _eval(fe, X, y, tau):
    P = fe.predict_proba(X); yhat = np.argmax(P, axis=1)
    acc = float((yhat == y).mean())
    s = calc_metric(P.max(1), (yhat == y).astype(int), tau=tau)
    return acc, float(s["coverage"]), float(s["selective_accuracy"]), P, yhat

tau = float(S.get("tau", 0.5))
acc_gt, cov_gt, sel_gt, Pte_gt, yhat_gt = _eval(fe_gt, C_test_true,               test_ds.y.astype(int), tau)
acc_mc, cov_mc, sel_mc, Pte_mc, yhat_mc = _eval(fe_mc, C_test_pred_noisy[:, J_machine], test_ds.y.astype(int), tau)

elig_global = [_name_to_col(nm) for nm in S["intervene_concepts"].split(",") if nm.strip()]
elig_mc_local = []
for nm in S["intervene_concepts"].split(","):
    nm = nm.strip()
    if not nm:
        continue
    try:
        jg = _name_to_col(nm)
    except KeyError:
        continue
    if jg in J_machine:
        elig_mc_local.append(J_machine.index(jg))

covs = [float(x) for x in S["coverages"].split(",") if x.strip()]

def _sweep(fe, X0, y, C_true, elig_cols, k, coverages, scope, select_by, seed=0):
    P0 = fe.predict_proba(X0); y0 = np.argmax(P0, 1); pmax = P0.max(1)
    idx_all = np.arange(len(y)); errs = idx_all[y0 != y]
    if select_by == "uncertainty":
        order_all = np.argsort(pmax)
        order_err = errs[np.argsort(pmax[errs])]
    else:
        r = np.random.default_rng(seed)
        order_all = r.permutation(idx_all)
        order_err = r.permutation(errs)
    out = []
    for cov in coverages:
        if scope == "errors":
            msel = int(round(cov * len(errs))); sel = order_err[:msel]
        else:
            msel = int(round(cov * len(idx_all))); sel = order_all[:msel]
        X = X0.copy()
        if k > 0 and len(elig_cols) > 0 and len(sel) > 0:
            for i in sel:
                diffs = np.abs(X[i, elig_cols] - C_true[i, elig_cols])
                kk = min(int(k), diffs.shape[0])
                fix = np.argpartition(-diffs, kk - 1)[:kk]
                X[i, np.array(elig_cols)[fix]] = C_true[i, np.array(elig_cols)[fix]]
        P1 = fe.predict_proba(X); y1 = np.argmax(P1, 1)
        pre = float((y0 == y).mean()); post = float((y1 == y).mean())
        harm = float(((y0[sel] == y[sel]) & (y1[sel] != y[sel])).mean()) if len(sel) > 0 else 0.0
        out.append({"coverage": cov, "pre": pre, "post": post, "delta": post - pre, "harm_rate": harm, "n_intervened": int(len(sel))})
    return pd.DataFrame(out)

sweep_gt = _sweep(fe_gt, C_test_true,               test_ds.y.astype(int), C_test_true,               [_name_to_col(nm) for nm in S["intervene_concepts"].split(",") if nm.strip()], int(S["intervention_k"]), covs, S["intervene_scope"], S["select_by"], seed=S["seed"])
sweep_mc = _sweep(fe_mc, C_test_pred_noisy[:, J_machine], test_ds.y.astype(int), C_test_true[:, J_machine], elig_mc_local, int(S["intervention_k"]), covs, S["intervene_scope"], S["select_by"], seed=S["seed"]+7)

summary = {
    "settings": S,
    "concept_names": names,
    "results": [
        {"name": "ground_truth_frontend", "acc": acc_gt, "coverage": cov_gt, "selective_accuracy": sel_gt},
        {"name": "machine_frontend", "acc": acc_mc, "coverage": cov_mc, "selective_accuracy": sel_mc}
    ],
    "sweep_ground_truth": sweep_gt.to_dict(orient="records"),
    "sweep_machine": sweep_mc.to_dict(orient="records")
}
(Path(S["out_dir"]) / "summary.json").write_text(json.dumps(summary, indent=2))
sweep_gt.to_csv(Path(S["out_dir"]) / "sweep_ground_truth.csv", index=False)
sweep_mc.to_csv(Path(S["out_dir"]) / "sweep_machine.csv", index=False)
print(json.dumps(summary, indent=2))
