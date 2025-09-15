from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from concept_benchmark.synthetic.helper.textgen import create_synthetic_dataset
from concept_benchmark.paths import pkg_dir, results_dir, data_dir
from concept_benchmark.synthetic.helper.text_concept_detector import TextConceptDetector
from concept_benchmark.models import ConceptBasedModel, FrontEndModel
from concept_benchmark.ext.fileutils import save as save_obj
from concept_benchmark.metrics import calc_metric
from concept_benchmark.data import ConceptDatasetSample
from types import SimpleNamespace
import argparse, psutil
from itertools import product
import builtins, functools
print = functools.partial(builtins.print, end="\n\n")

tpl_path = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "Templates.txt"
with open(tpl_path, "r", encoding="utf-8-sig") as f:
    templates = [ln.strip() for ln in f if ln.strip()]

ROBOT_RUN_DIR = results_dir / "robot_text"
ROBOT_DATA_DIR = data_dir / "robot_text"
ROBOT_RUN_DIR.mkdir(parents=True, exist_ok=True)
ROBOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

settings = {
    "variant": "perfect",
    "imperfect_strategy": "",
    "heldout_concepts": [],
    "mask_p": 1.0,
    "test_label_prior": "",
    "seed": 1337,
    "concept_mode": "hard",
    "train_on_detected": False,
    "templates_file": "",
    "label_model_expr": "",
    "corr_pair": "",
    "train_corr": 1.0,
    "test_break": 1.0,
    "test_corr": -1.0,
    "budgets": "0,1,2,5,10",
    "target_acc_grid": "0.7,0.8,0.9,0.95",
    "target_acc_concepts": "",
    "intervene_allow": "",
    "human_acc": 1.0,
    "human_acc_concepts": "",
    "skew_concept": "",
    "make_plots": 0,
    "policy": "uncertainty",
    "concept_include": "",
    "concept_exclude": "",
    "blackbox_metrics": "",
    "concept_source": "detected",
    "machine_method": "kmeans",
    "machine_k": 16,
    "machine_soft": 1,
    "machine_seed": 0,
    "machine_upper_bound": 0,
    "mask_mode": "rowdrop",
    "mask_rate": 0.0,
}

def _csv_list(s: str) -> list[str]:
    s = str(s).strip()
    return [t.strip() for t in s.split(",")] if s else []

def _csv_kv_float(s: str) -> dict:
    out = {}
    if not s: return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part: continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except:
            pass
    return out

if psutil.Process(psutil.Process().ppid()).name().lower().startswith("pycharm"):
    args_obj = SimpleNamespace(**settings)
else:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--variant", choices=["perfect", "imperfect"], default=settings["variant"])
    ap.add_argument("--imperfect-strategy", choices=["missing_concepts", "label_prior_shift"], dest="imperfect_strategy", default=settings["imperfect_strategy"])
    ap.add_argument("--heldout-concepts", type=_csv_list, default=settings["heldout_concepts"])
    ap.add_argument("--mask-p", type=float, default=settings["mask_p"])
    ap.add_argument("--test-label-prior", type=str, default=settings["test_label_prior"])
    ap.add_argument("--seed", type=int, default=settings["seed"])
    ap.add_argument("--concept-mode", choices=["hard", "soft"], default=settings["concept_mode"])
    ap.add_argument("--train-on-detected", action="store_true", default=settings["train_on_detected"])
    ap.add_argument("--templates-file", type=str, default=settings["templates_file"])
    ap.add_argument("--label-model-expr", type=str, default=settings["label_model_expr"])
    ap.add_argument("--corr-pair", type=str, default=settings["corr_pair"])
    ap.add_argument("--train-corr", type=float, default=settings["train_corr"])
    ap.add_argument("--test-break", type=float, default=settings["test_break"])
    ap.add_argument("--test-corr", type=float, default=settings["test_corr"])
    ap.add_argument("--budgets", type=str, default=settings["budgets"])
    ap.add_argument("--target-acc-grid", type=str, default=settings["target_acc_grid"])
    ap.add_argument("--target-acc-concepts", type=str, default=settings["target_acc_concepts"])
    ap.add_argument("--intervene-allow", type=str, default=settings["intervene_allow"])
    ap.add_argument("--human-acc", type=float, default=settings["human_acc"])
    ap.add_argument("--human-acc-concepts", type=str, default=settings["human_acc_concepts"])
    ap.add_argument("--skew-concept", type=str, default=settings["skew_concept"])
    ap.add_argument("--make-plots", type=int, default=settings["make_plots"])
    ap.add_argument("--policy", choices=["uncertainty","oracle"], default=settings["policy"])
    ap.add_argument("--concept-include", type=str, default=settings["concept_include"])
    ap.add_argument("--concept-exclude", type=str, default=settings["concept_exclude"])
    ap.add_argument("--blackbox_metrics", type=str, default=settings["blackbox_metrics"])
    ap.add_argument("--concept-source", choices=["detected","gt","machine"], default=settings["concept_source"])
    ap.add_argument("--machine-method", choices=["kmeans"], default=settings["machine_method"])
    ap.add_argument("--machine-k", type=int, default=settings["machine_k"])
    ap.add_argument("--machine-soft", type=int, default=settings["machine_soft"])
    ap.add_argument("--machine-seed", type=int, default=settings["machine_seed"])
    ap.add_argument("--machine-upper-bound", type=int, default=settings["machine_upper_bound"])
    ap.add_argument("--mask-mode", choices=["rowdrop","mask"], default=settings["mask_mode"])
    ap.add_argument("--mask-rate", type=float, default=settings["mask_rate"])
    known, _ = ap.parse_known_args()
    merged = dict(settings)
    merged.update({
        "variant": known.variant,
        "imperfect_strategy": known.imperfect_strategy,
        "heldout_concepts": known.heldout_concepts if isinstance(known.heldout_concepts, list) else _csv_list(known.heldout_concepts),
        "mask_p": known.mask_p,
        "test_label_prior": known.test_label_prior,
        "seed": known.seed,
        "concept_mode": known.concept_mode,
        "train_on_detected": bool(known.train_on_detected),
        "templates_file": known.templates_file or "",
        "label_model_expr": known.label_model_expr or "",
        "corr_pair": known.corr_pair or "",
        "train_corr": float(known.train_corr),
        "test_break": float(known.test_break),
        "test_corr": float(known.test_corr),
        "budgets": known.budgets,
        "target_acc_grid": known.target_acc_grid,
        "target_acc_concepts": known.target_acc_concepts,
        "intervene_allow": known.intervene_allow,
        "human_acc": float(known.human_acc),
        "human_acc_concepts": known.human_acc_concepts,
        "skew_concept": known.skew_concept,
        "make_plots": int(known.make_plots),
        "policy": known.policy,
        "concept_include": known.concept_include,
        "concept_exclude": known.concept_exclude,
        "blackbox_metrics": known.blackbox_metrics,
        "concept_source": known.concept_source,
        "machine_method": known.machine_method,
        "machine_k": int(known.machine_k),
        "machine_soft": int(known.machine_soft),
        "machine_seed": int(known.machine_seed),
        "machine_upper_bound": int(known.machine_upper_bound),
        "mask_mode": known.mask_mode,
        "mask_rate": float(known.mask_rate),
    })
    if merged["test_corr"] is not None and merged["test_corr"] >= 0:
        merged["test_break"] = max(0.0, min(1.0, 1.0 - float(merged["test_corr"])))
    args_obj = SimpleNamespace(**merged)

if args_obj.templates_file:
    with open(Path(args_obj.templates_file), "r", encoding="utf-8-sig") as f:
        templates = [ln.strip() for ln in f if ln.strip()]

VARIANT = args_obj.variant
IMPERFECT_STRATEGY = args_obj.imperfect_strategy
HELDOUT_CONCEPTS = args_obj.heldout_concepts
MASK_P = float(args_obj.mask_p)
TEST_LABEL_PRIOR = args_obj.test_label_prior
SEED = int(args_obj.seed)
CONCEPT_MODE = args_obj.concept_mode
train_on_detected = bool(args_obj.train_on_detected)

params = {
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "foot_shape": ["flat_4sided","flat_5sided","flat_lshaped","pointy_3sided","pointy_4sided","pointy_6sided"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": ["round_circle","wide_oval","tall_oval","edgy_square","edgy_triangle","edgy_trapezoid"],
    },
    "model": "'glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')) - 2 >= 0) else 'drent'",
}
if args_obj.label_model_expr:
    params["model"] = args_obj.label_model_expr

def enumerate_concepts(params, shuffle=True, seed=0):
    cols = list(params["concepts"].keys())
    grids = [params["concepts"][c] for c in cols]
    combos = list(product(*grids))
    df = pd.DataFrame(combos, columns=cols)
    if shuffle:
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df

def sample_concepts(params, n=50, seed=0):
    rng = np.random.default_rng(seed)
    cols = list(params["concepts"].keys())
    rows = []
    for _ in range(n):
        r = {}
        for c in cols:
            r[c] = rng.choice(params["concepts"][c])
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)

def compute_label(df: pd.DataFrame, model_expr: str) -> pd.Series:
    SAFE_GLOBALS = {"__builtins__": None, "int": int, "str": str, "float": float, "bool": bool, "any": any, "all": all}
    def eval_one(sr):
        row = sr.to_dict()
        return eval(model_expr, SAFE_GLOBALS, {"row": row})
    return df.apply(eval_one, axis=1)

def _subset_sample(sample: ConceptDatasetSample, keep_idx: np.ndarray, concepts, classes) -> ConceptDatasetSample:
    keep_idx = np.asarray(keep_idx, dtype=int)
    X = [str(x) for x in np.array(sample.X, dtype=object)[keep_idx]]
    C = sample.C[keep_idx]
    y = sample.y[keep_idx]
    return ConceptDatasetSample(X=X, C=C, y=y, meta={"concepts": concepts, "classes": classes, "data_type": "text"})

def _apply_missing_concepts(train_sample: ConceptDatasetSample, concepts: list[str], heldout: list[str], mask_p: float, seed: int) -> ConceptDatasetSample:
    if not heldout:
        return train_sample
    name_to_idx = {n: i for i, n in enumerate(concepts)}
    cols = []
    for spec in heldout:
        if spec in name_to_idx:
            cols.append(name_to_idx[spec]); continue
        key = spec.split("=", 1)[0].strip()
        cols.extend([i for i, n in enumerate(concepts) if n.startswith(key)])
    cols = sorted(set(cols))
    if not cols:
        return train_sample
    C = train_sample.C.astype(np.float32)
    active = (C[:, cols] > 0.5).any(axis=1)
    if mask_p >= 1.0:
        keep = ~active
    else:
        rng = np.random.default_rng(seed)
        drop = active & (rng.random(active.shape[0]) < mask_p)
        keep = ~drop
    keep_idx = np.where(keep)[0]
    if keep_idx.size == 0:
        keep_idx = np.where(~active)[0]
    return _subset_sample(train_sample, keep_idx, concepts, train_sample.meta.get("classes", []))

def _parse_label_prior(spec: str, classes) -> dict:
    if not spec:
        return {}
    out = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        k, v = p.split(":")
        k = k.strip(); v = float(v.strip())
        if k.isdigit():
            k = int(k)
        else:
            if k in classes:
                k = int(np.where(np.array(classes, dtype=object) == k)[0][0])
            else:
                continue
        out[k] = v
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] /= s
    return out

def _apply_label_prior_shift(val_sample: ConceptDatasetSample, prior: dict, seed: int) -> ConceptDatasetSample:
    if not prior:
        return val_sample
    rng = np.random.default_rng(seed)
    y = val_sample.y.astype(int)
    classes = sorted(np.unique(y).tolist())
    n = len(y)
    target_counts = {c: int(round(prior.get(c, (y == c).mean()) * n)) for c in classes}
    chosen_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        k = target_counts[c]
        if idx.size == 0:
            continue
        if k <= idx.size:
            sel = rng.choice(idx, size=k, replace=False)
        else:
            sel = rng.choice(idx, size=k, replace=True)
        chosen_idx.append(sel)
    if not chosen_idx:
        return val_sample
    keep_idx = np.concatenate(chosen_idx)
    rng.shuffle(keep_idx)
    return _subset_sample(val_sample, keep_idx, val_sample.meta.get("concepts", []), val_sample.meta.get("classes", []))

def _group_indices(names, key):
    return [i for i, n in enumerate(names) if n.startswith(key + "=")]

def _corr_equal_mask(sample: ConceptDatasetSample, a: str, b: str) -> np.ndarray:
    names = list(sample.concepts)
    ai = _group_indices(names, a)
    bi = _group_indices(names, b)
    T = sample.C.astype(int)
    aa = T[:, ai].argmax(1)
    bb = T[:, bi].argmax(1)
    return aa == bb

def _enforce_corr(sample: ConceptDatasetSample, pair: str, frac_corr: float, seed: int) -> ConceptDatasetSample:
    if not pair:
        return sample
    a, b = [t.strip() for t in pair.split(",")]
    m = _corr_equal_mask(sample, a, b)
    idx_corr = np.where(m)[0]
    idx_brk = np.where(~m)[0]
    n = sample.n
    n_corr = min(idx_corr.size, int(round(frac_corr * n)))
    n_brk = min(idx_brk.size, n - n_corr)
    rng = np.random.default_rng(seed)
    sel_corr = rng.choice(idx_corr, size=n_corr, replace=False) if n_corr > 0 else np.array([], dtype=int)
    sel_brk = rng.choice(idx_brk, size=n_brk, replace=False) if n_brk > 0 else np.array([], dtype=int)
    keep = np.concatenate([sel_corr, sel_brk])
    rng.shuffle(keep)
    return _subset_sample(sample, keep, list(sample.concepts), sample.meta.get("classes", []))

def _degrade_to_acc(H: np.ndarray, T: np.ndarray, target: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = H.copy()
    correct = (H == T).reshape(-1)
    cur = float(correct.mean())
    if target >= cur:
        return H
    need = int(round((cur - target) * H.size))
    idx = np.where(correct)[0]
    if need > idx.size:
        need = idx.size
    if need > 0:
        sel = rng.choice(idx, size=need, replace=False)
        flat = H.reshape(-1)
        flat[sel] = 1 - flat[sel]
        H = flat.reshape(H.shape)
    return H

def _indices_for(names, spec):
    out = []
    spec = spec.strip()
    if "=" in spec:
        if spec in names:
            out.append(names.index(spec))
    else:
        out.extend([i for i, n in enumerate(names) if n.startswith(spec + "=")])
    return sorted(set(out))

def _apply_per_concept_degrade(H: np.ndarray, T: np.ndarray, names: list[str], mapping: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = H.copy()
    for k, v in mapping.items():
        idxs = _indices_for(names, k)
        if not idxs: continue
        for j in idxs:
            pred = out[:, j]
            truth = T[:, j]
            cur = (pred == truth).mean()
            if v >= cur: continue
            need = int(round((cur - v) * pred.size))
            correct_idx = np.where(pred == truth)[0]
            if need > correct_idx.size: need = correct_idx.size
            if need > 0:
                sel = rng.choice(correct_idx, size=need, replace=False)
                pred2 = pred.copy()
                pred2[sel] = 1 - pred2[sel]
                out[:, j] = pred2
    return out

def _apply_human_edit(p_row, truth_row, sel_idxs, names, acc_default, acc_map, rng):
    for j in sel_idxs:
        name = names[j]
        base = name.split("=", 1)[0]
        acc = acc_map.get(name, acc_map.get(base, acc_default))
        if rng.random() < acc:
            p_row[j] = truth_row[j]
        else:
            p_row[j] = 1 - truth_row[j]
    return p_row

def _allowed_indices(names, allow_spec):
    if not allow_spec: return np.arange(len(names), dtype=int)
    out = []
    for tok in _csv_list(allow_spec):
        out.extend(_indices_for(names, tok))
    return np.array(sorted(set(out)), dtype=int)

def _apply_skew(sample: ConceptDatasetSample, spec: str, seed: int) -> ConceptDatasetSample:
    if not spec: return sample
    parts = [t.strip() for t in spec.split(",")]
    if len(parts) != 3: return sample
    key, val, p = parts[0], parts[1], float(parts[2])
    names = list(sample.concepts)
    idxs = _indices_for(names, f"{key}={val}")
    if not idxs: return sample
    j = idxs[0]
    T = sample.C.astype(int)
    idx_pos = np.where(T[:, j] == 1)[0]
    idx_neg = np.where(T[:, j] == 0)[0]
    n = sample.n
    n_pos = min(idx_pos.size, int(round(p * n)))
    n_neg = min(idx_neg.size, n - n_pos)
    rng = np.random.default_rng(seed)
    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    keep = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(keep)
    return _subset_sample(sample, keep, names, sample.meta.get("classes", []))

def _select_concept_columns(sample: ConceptDatasetSample, keep_idx: np.ndarray) -> ConceptDatasetSample:
    keep_idx = np.asarray(keep_idx, dtype=int)
    X = sample.X
    C = sample.C[:, keep_idx]
    y = sample.y
    names = list(sample.concepts)
    keep_names = [names[i] for i in keep_idx]
    return ConceptDatasetSample(X=X, C=C, y=y, meta={"concepts": keep_names, "classes": sample.meta.get("classes", []), "data_type": "text"})

def _tfidf_fit(texts, seed):
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000, dtype=np.float32)
    X = vec.fit_transform([str(t) for t in texts])
    return vec, X

def _kmeans_fit(X, k, seed):
    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed))
    km.fit(X)
    return km

def _kmeans_soft(X, km):
    D = km.transform(X)
    S = np.exp(-D)
    S_sum = S.sum(axis=1, keepdims=True)
    S_sum[S_sum == 0] = 1.0
    return S / S_sum

def _machine_truth_map(H_train_hard, C_train_true):
    J = H_train_hard.shape[1]
    mapping = []
    for j in range(J):
        col = H_train_hard[:, j].astype(int)
        best = 0
        best_idx = 0
        for t in range(C_train_true.shape[1]):
            c = C_train_true[:, t].astype(int)
            agree = int(((col == 1) & (c == 1)).sum() + ((col == 0) & (c == 0)).sum())
            if agree > best:
                best = agree; best_idx = t
        mapping.append(best_idx)
    return np.array(mapping, dtype=int)

cols = list(params["concepts"].keys())
catalog_df = pd.DataFrame([dict(zip(cols, vals)) for vals in product(*[params["concepts"][c] for c in cols])], columns=cols)
catalog_df["label"] = compute_label(catalog_df, params["model"])

concept_cols = list(params["concepts"].keys())
llm_user_prompt = "Using the provided attributes, write a natural spoken description (1–3 sentences) that sounds like a person describing an image they saw. Do not invent locations or scenarios; focus only on what the attributes imply."

ds = create_synthetic_dataset(
    source=catalog_df,
    templates=templates,
    variants_per_row=3,
    include_color=False,
    rng_seed=0,
    concept_cols=concept_cols,
    label_col="label",
    label_map={"drent": 0, "glorp": 1},
    text_mode="semi",
    llm_provider="gemini",
    llm_model="gemini-1.5-flash",
    llm_user_prompt=llm_user_prompt,
)

ds_path = ROBOT_DATA_DIR / "robot_text_dataset.pkl"
save_obj(ds, ds_path, overwrite=True)

print("SAMPLE CAPTIONS:")
for x in ds.X[:6]:
    print("-", x)

print("\nCONCEPT NAMES:", ds.concepts)
print("CLASSES:", ds.classes)
print("N samples:", len(ds))

out_csv = ROBOT_RUN_DIR / "text_samples.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
row_index = getattr(getattr(ds, "_full", ds), "meta", {}).get("row_index", np.arange(len(ds)))
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    cols = ["text"] + concept_cols + ["label"]
    w.writerow(cols)
    for i, x in enumerate(ds.X):
        src_idx = int(row_index[i])
        row_vals = catalog_df.loc[src_idx, concept_cols].tolist()
        w.writerow([x] + row_vals + [catalog_df.loc[src_idx, "label"]])

print(f"\nWrote {len(ds)} rows to {out_csv}")

if ds.cvindices is None or getattr(ds.validation, "n", 0) == 0 or getattr(ds, "test", None) is None:
    n_folds = 5
    rng = np.random.default_rng(0)
    base_ids = np.unique(row_index)
    rng.shuffle(base_ids)
    assign = {int(rid): i % n_folds for i, rid in enumerate(base_ids)}
    fold_arr = np.array([assign[int(r)] for r in row_index], dtype=int)
    if ds.cvindices is None:
        ds.cvindices = {}
    if "by_robot" not in ds.cvindices:
        ds.cvindices["by_robot"] = fold_arr
    ds.split(fold_id="by_robot", fold_num_validation=0, fold_num_test=1)
    print(f"Split sizes → train: {ds.training.n}, val: {ds.validation.n}, test: {ds.test.n}")

print(f"Variant: {VARIANT} | Strategy: {IMPERFECT_STRATEGY}")
train_ds = ds.training
val_ds = ds.validation
test_ds = ds.test

if args_obj.corr_pair:
    train_ds = _enforce_corr(train_ds, args_obj.corr_pair, float(args_obj.train_corr), SEED)
    test_ds = _enforce_corr(test_ds, args_obj.corr_pair, max(0.0, 1.0 - float(args_obj.test_break)), SEED + 1)

if args_obj.skew_concept:
    train_ds = _apply_skew(train_ds, args_obj.skew_concept, SEED)
    test_ds = _apply_skew(test_ds, args_obj.skew_concept, SEED + 2)

if VARIANT == "imperfect":
    if IMPERFECT_STRATEGY == "missing_concepts":
        if args_obj.mask_mode == "rowdrop":
            train_ds = _apply_missing_concepts(train_ds, ds.concepts, HELDOUT_CONCEPTS, MASK_P, SEED)
        else:
            pass
    elif IMPERFECT_STRATEGY == "label_prior_shift":
        prior = _parse_label_prior(TEST_LABEL_PRIOR, ds.classes)
        val_ds = _apply_label_prior_shift(val_ds, prior, SEED)

names_all = list(train_ds.concepts)
inc_idx = []
exc_idx = []
if args_obj.concept_include:
    for tok in _csv_list(args_obj.concept_include):
        inc_idx.extend(_indices_for(names_all, tok))
if args_obj.concept_exclude:
    for tok in _csv_list(args_obj.concept_exclude):
        exc_idx.extend(_indices_for(names_all, tok))
if inc_idx or exc_idx:
    if not inc_idx:
        inc_idx = list(range(len(names_all)))
    keep = sorted(set(inc_idx) - set(exc_idx))
    if keep:
        train_ds = _select_concept_columns(train_ds, np.array(keep, dtype=int))
        val_ds = _select_concept_columns(val_ds, np.array(keep, dtype=int))
        test_ds = _select_concept_columns(test_ds, np.array(keep, dtype=int))

detector = TextConceptDetector(
    embed_dim=128,
    hidden_dim=192,
    epochs=6,
    batch_size=64,
    use_bigrams=True,
    lr=2e-3,
    dropout=0.1,
    pos_weight="auto",
    output_mode=CONCEPT_MODE,
    threshold_mode="auto",
    pooling="attn",
    group_unknown_threshold=0.50,
    validate=True,
)

label_mask = None
if VARIANT == "imperfect" and IMPERFECT_STRATEGY == "missing_concepts" and args_obj.mask_mode == "mask" and HELDOUT_CONCEPTS:
    names = list(train_ds.concepts)
    J = len(names)
    label_mask = np.ones((train_ds.C.shape[0], J), dtype=np.int32)
    cols = []
    for spec in HELDOUT_CONCEPTS:
        cols.extend(_indices_for(names, spec))
    cols = sorted(set(cols))
    if cols:
        rngm = np.random.default_rng(SEED + 123)
        for j in cols:
            m = rngm.random(train_ds.C.shape[0]) < float(args_obj.mask_rate)
            label_mask[m, j] = 0

print("Fitting detector")
try:
    detector.fit(train_ds, val_ds, label_mask=label_mask)
except TypeError:
    detector.fit(train_ds, val_ds)

cbm = ConceptBasedModel(concept_detector=detector, front_end_model=FrontEndModel(), propagate=False)

C_train = train_ds.C
y_train = train_ds.y

if train_on_detected:
    old_mode = getattr(detector, "output_mode", None)
    try:
        if hasattr(detector, "output_mode"):
            detector.output_mode = "soft"
        C_train_used = detector.predict(train_ds)
    finally:
        if hasattr(detector, "output_mode") and old_mode is not None:
            detector.output_mode = old_mode
else:
    C_train_used = C_train

cbm.front_end_model.fit(C_train_used, y_train)

with np.errstate(invalid="ignore"):
    C_val_true = val_ds.C.astype(np.float32)

_old = detector.output_mode
detector.output_mode = "soft"
C_val_scores = detector.predict(val_ds)
detector.output_mode = _old

concept_names = list(ds.concepts)
per = {}
for j, name in enumerate(concept_names):
    yt = C_val_true[:, j]
    ys = C_val_scores[:, j]
    try:
        auprc = average_precision_score(yt, ys)
    except Exception:
        auprc = float("nan")
    try:
        rocauc = roc_auc_score(yt, ys) if len(np.unique(yt)) == 2 else float("nan")
    except Exception:
        rocauc = float("nan")
    per[name] = {"auprc": float(auprc), "roc_auc": float(rocauc)}

auprc_macro = float(np.nanmean([d["auprc"] for d in per.values()])) if per else float("nan")
roc_macro = float(np.nanmean([d["roc_auc"] for d in per.values()])) if per else float("nan")

print("Macro concept metrics:", {"auprc_macro": auprc_macro, "roc_auc_macro": roc_macro})
print("Sample per-concept metrics (first 5):", {k: per[k] for k in list(per.keys())[:5]})

texts_demo = [str(x) for x in ds.X[:3]]
dummy_C = np.zeros((len(texts_demo), len(ds.concepts)), dtype=np.float32)
dummy_y = np.zeros((len(texts_demo),), dtype=int)
demo_ds = ConceptDatasetSample(X=texts_demo, C=dummy_C, y=dummy_y, meta={"concepts": ds.concepts, "classes": ds.classes, "data_type": "text"})
proba_demo = detector.predict(demo_ds)
print("Concept order:", concept_names)
print(f"Concept outputs (mode={CONCEPT_MODE}) shape:", proba_demo.shape)
print("First row outputs:", proba_demo[0])

y_train_true = train_ds.y.astype(int)
y_train_proba = cbm.predict_proba(train_ds)
y_train_pred = np.argmax(y_train_proba, axis=1)
acc_train = accuracy_score(y_train_true, y_train_pred)
try:
    cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    roc_train = roc_auc_score(y_train_true, y_train_proba[:, cls_index_1]) if len(np.unique(y_train_true)) == 2 else float("nan")
except Exception:
    roc_train = float("nan")
print("Label model metrics (train):", {"accuracy": float(acc_train), "roc_auc": float(roc_train)})

y_val = val_ds.y.astype(int)
y_val_proba = cbm.predict_proba(val_ds)
y_val_pred = np.argmax(y_val_proba, axis=1)
acc = accuracy_score(y_val, y_val_pred)
try:
    cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    roc = roc_auc_score(y_val, y_val_proba[:, cls_index_1]) if len(np.unique(y_val)) == 2 else float("nan")
except Exception:
    roc = float("nan")
print("Label model metrics (validation):", {"accuracy": float(acc), "roc_auc": float(roc)})

y_test = test_ds.y.astype(int)
y_test_proba = cbm.predict_proba(test_ds)
y_test_pred = np.argmax(y_test_proba, axis=1)
acc_test = accuracy_score(y_test, y_test_pred)
try:
    cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    roc_test = roc_auc_score(y_test, y_test_proba[:, cls_index_1]) if len(np.unique(y_test)) == 2 else float("nan")
except Exception:
    roc_test = float("nan")
print("Label model metrics (test):", {"accuracy": float(acc_test), "roc_auc": float(roc_test)})

_old_tr = detector.output_mode
detector.output_mode = "soft"
C_train_scores = detector.predict(train_ds)
detector.output_mode = _old_tr

C_train_true = train_ds.C.astype(np.float32)
sel_covs_tr, sel_accs_tr, aucs_tr, auprcs_tr = [], [], [], []
for j in range(C_train_true.shape[1]):
    m = calc_metric(C_train_scores[:, j], C_train_true[:, j], tau=0.5)
    sel_covs_tr.append(m["coverage"])
    sel_accs_tr.append(m["selective_accuracy"])
    try:
        if len(np.unique(C_train_true[:, j])) == 2:
            aucs_tr.append(roc_auc_score(C_train_true[:, j], C_train_scores[:, j]))
            auprcs_tr.append(average_precision_score(C_train_true[:, j], C_train_scores[:, j]))
    except Exception:
        pass
concept_train_metrics = {
    "selective_cov_mean": float(np.nanmean(sel_covs_tr)) if sel_covs_tr else float("nan"),
    "selective_acc_mean": float(np.nanmean(sel_accs_tr)) if sel_accs_tr else float("nan"),
    "auroc_macro": float(np.nanmean(aucs_tr)) if aucs_tr else float("nan"),
    "auprc_macro": float(np.nanmean(auprcs_tr)) if auprcs_tr else float("nan"),
    "tau": 0.5,
}
print("Concept metrics (train):", concept_train_metrics)

_old2 = detector.output_mode
detector.output_mode = "soft"
C_test_scores = detector.predict(test_ds)
detector.output_mode = _old2

C_test_true = test_ds.C.astype(np.float32)
sel_covs_t, sel_accs_t, aucs_t, auprcs_t = [], [], [], []
for j in range(C_test_true.shape[1]):
    m = calc_metric(C_test_scores[:, j], C_test_true[:, j], tau=0.5)
    sel_covs_t.append(m["coverage"])
    sel_accs_t.append(m["selective_accuracy"])
    try:
        if len(np.unique(C_test_true[:, j])) == 2:
            aucs_t.append(roc_auc_score(C_test_true[:, j], C_test_scores[:, j]))
            auprcs_t.append(average_precision_score(C_test_true[:, j], C_test_scores[:, j]))
    except Exception:
        pass
concept_test_metrics = {
    "selective_cov_mean": float(np.nanmean(sel_covs_t)) if sel_covs_t else float("nan"),
    "selective_acc_mean": float(np.nanmean(sel_accs_t)) if sel_accs_t else float("nan"),
    "auroc_macro": float(np.nanmean(aucs_t)) if aucs_t else float("nan"),
    "auprc_macro": float(np.nanmean(auprcs_t)) if auprcs_t else float("nan"),
    "tau": 0.5,
}
print("Concept metrics (test):", concept_test_metrics)

def _groups(names):
    g, singles = {}, []
    for j, n in enumerate(names):
        if "=" in n:
            k = n.split("=", 1)[0]
            g.setdefault(k, []).append(j)
        else:
            singles.append(j)
    return {k: v for k, v in g.items() if len(v) > 1}, singles

def _bin_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, (prec + rec))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1)}

def _concept_error_report(split_name, sample):
    names = list(sample.concepts)
    G, singles = _groups(names)
    oldm = detector.output_mode
    detector.output_mode = "hard"
    H = detector.predict(sample)
    detector.output_mode = oldm
    T = sample.C.astype(int)
    per_concept = {n: _bin_metrics(T[:, j], H[:, j]) for j, n in enumerate(names)}
    per_group = {}
    for k, idxs in G.items():
        t = T[:, idxs]
        h = H[:, idxs]
        unk = (h.sum(1) == 0)
        pred = np.where(unk, -1, h.argmax(1))
        true = t.argmax(1)
        known = ~unk
        acc_known = float((pred[known] == true[known]).mean()) if known.any() else float("nan")
        per_group[k] = {"acc_known": acc_known, "unknown_rate": float(unk.mean())}
    worst = dict(sorted(per_concept.items(), key=lambda kv: kv[1]["acc"])[:5])
    print(f"=== Concept error report [{split_name}] ===")
    print("Per-group:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_group.items()})
    print("Worst 5 concepts by acc:", {k: round(v["acc"], 4) for k, v in worst.items()})
    return {"per_concept": per_concept, "per_group": per_group}

_ = _concept_error_report("train", train_ds)
_ = _concept_error_report("val", val_ds)
_ = _concept_error_report("test", test_ds)

all_probs = cbm.predict_proba(ds)
all_preds = np.argmax(all_probs, axis=1)
label_names = list(ds.classes)
pred_labels = [label_names[i] for i in all_preds]

metrics_out = {}
try:
    if len(np.unique(y_val)) == 2:
        cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
        lbl_sel = calc_metric(y_val_proba[:, cls_index_1], y_val, tau=0.5)
    else:
        lbl_sel = {"coverage": float("nan"), "selective_accuracy": float("nan"), "tau": 0.5}
except Exception:
    lbl_sel = {"coverage": float("nan"), "selective_accuracy": float("nan"), "tau": 0.5}

metrics_out["label"] = {"accuracy": float(acc), "roc_auc": float(roc), "selective": lbl_sel}
metrics_out["label_test"] = {"accuracy": float(acc_test), "roc_auc": float(roc_test)}
metrics_out["label_train"] = {"accuracy": float(acc_train), "roc_auc": float(roc_train)}
metrics_out["label_val"] = metrics_out["label"]

n_concepts = C_val_true.shape[1]
sel_covs, sel_accs, aucs, auprcs = [], [], [], []
for j in range(n_concepts):
    m = calc_metric(C_val_scores[:, j], C_val_true[:, j], tau=0.5)
    sel_covs.append(m["coverage"])
    sel_accs.append(m["selective_accuracy"])
    try:
        if len(np.unique(C_val_true[:, j])) == 2:
            aucs.append(roc_auc_score(C_val_true[:, j], C_val_scores[:, j]))
            auprcs.append(average_precision_score(C_val_true[:, j], C_val_scores[:, j]))
    except Exception:
        pass
concept_metrics = {
    "selective_cov_mean": float(np.nanmean(sel_covs)) if sel_covs else float("nan"),
    "selective_acc_mean": float(np.nanmean(sel_accs)) if sel_accs else float("nan"),
    "auroc_macro": float(np.nanmean(aucs)) if aucs else float("nan"),
    "auprc_macro": float(np.nanmean(auprcs)) if auprcs else float("nan"),
    "tau": 0.5,
}
metrics_out["concepts"] = concept_metrics
metrics_out["concepts_test"] = concept_test_metrics
metrics_out["concepts_train"] = concept_train_metrics
metrics_out["concepts_val"] = metrics_out["concepts"]

run_info = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "concept_mode": CONCEPT_MODE,
    "pos_weight": "auto",
    "n_samples": int(len(ds.X)),
    "n_concepts": int(n_concepts),
    "classes": list(ds.classes),
    "concept_names": list(ds.concepts),
    "variant": VARIANT,
    "strategy": IMPERFECT_STRATEGY,
    "heldout_concepts": HELDOUT_CONCEPTS,
    "mask_p": MASK_P,
    "test_label_prior": TEST_LABEL_PRIOR,
    "seed": SEED,
}

run_meta = {}
if getattr(detector, "thresholds_", None) is not None:
    run_meta["thresholds"] = detector.thresholds_.astype(float).tolist()
    np.savetxt(ROBOT_RUN_DIR / "thresholds.csv", detector.thresholds_, delimiter=",")
    np.save(ROBOT_RUN_DIR / "thresholds.npy", detector.thresholds_)
if getattr(detector, "concept_acc_", None) is not None:
    run_meta["concept_acc_per_concept"] = detector.concept_acc_.astype(float).tolist()
if getattr(detector, "alignment_", None) is not None:
    run_meta["alignment"] = {k: float(v) for k, v in detector.alignment_.items()}
if getattr(detector, "cross_auroc_", None) is not None:
    A = detector.cross_auroc_
    np.savetxt(ROBOT_RUN_DIR / "cross_auroc.csv", A, delimiter=",")
    run_meta["cross_auroc_diag_mean"] = float(np.nanmean(np.diag(A))) if A.size else float("nan")
metrics_out["detector_run_meta"] = run_meta

payload = {"run": run_info, "metrics": metrics_out}
with open(ROBOT_RUN_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

save_obj({"cbm": cbm, "detector": detector, "train_variant": VARIANT, "strategy": IMPERFECT_STRATEGY}, ROBOT_RUN_DIR / "model.pkl", overwrite=True)
print("Saved dataset:", ds_path)
print("Saved model:", ROBOT_RUN_DIR / "model.pkl")
print("Saved metrics:", ROBOT_RUN_DIR / "metrics.json")

from concept_benchmark.ext.fileutils import load as load_obj
MODEL = ROBOT_RUN_DIR / "model.pkl"
DATA  = ROBOT_DATA_DIR / "robot_text_dataset.pkl"
obj = load_obj(MODEL)
detector = obj["detector"]
ds = load_obj(DATA)

def ensure_split(ds):
    try:
        if ds.validation.n == 0 and ds.test.n == 0:
            ds.reset()
            ds.split(fold_id=0, fold_num_validation=1, fold_num_test=2)
    except Exception:
        pass

def pick_split(ds, name):
    d = {"train": ds.training, "val": ds.validation, "test": ds.test}[name]
    return d if d.n > 0 else ds

def groups(names):
    g = {}
    for j,n in enumerate(names):
        if "=" in n:
            g.setdefault(n.split("=",1)[0], []).append(j)
    return {k:v for k,v in g.items() if len(v)>1}

def metrics(names, hard, C):
    out = {}
    for k, idxs in groups(names).items():
        t = C[:, idxs]; p = hard[:, idxs]
        if t.shape[0] == 0: out[k] = {"acc_known": float("nan"), "unknown_rate": float("nan")}; continue
        unk = (p.sum(1) == 0)
        known = ~unk
        acc_known = (p.argmax(1)[known] == t.argmax(1)[known]).mean() if known.any() else float("nan")
        out[k] = {"acc_known": float(acc_known), "unknown_rate": float(unk.mean())}
    return out

def worst_examples(split_ds, names, hard, C, concept_key, k=10):
    j = names.index(concept_key)
    fn_idx = np.where((C[:,j]==1)&(hard[:,j]==0))[0]
    fp_idx = np.where((C[:,j]==0)&(hard[:,j]==1))[0]
    return {"n_FN": int(fn_idx.size), "n_FP": int(fp_idx.size), "FN": [str(split_ds.X[i]) for i in fn_idx[:k]], "FP": [str(split_ds.X[i]) for i in fp_idx[:k]]}

def run(name):
    ensure_split(ds)
    split = pick_split(ds, name)
    names = list(split.concepts)
    C = split.C.astype(int)
    H = detector.predict(split)
    print(f"=== {name.upper()} (n={split.n}) ===")
    print("Before:", metrics(names, H, C))
    ex = worst_examples(split, names, H, C, "head_shape=square", k=5)
    print("head=square FNs:", ex["n_FN"], "FPs:", ex["n_FP"])
    print("FN examples:", ex["FN"][:3])
    print("FP examples:", ex["FP"][:3])

run("val"); run("test")

_old = detector.output_mode
detector.output_mode = "soft"
C_test_scores = detector.predict(test_ds).astype(float)
detector.output_mode = "hard"
H_test = detector.predict(test_ds).astype(int)
detector.output_mode = _old

T_test = test_ds.C.astype(int)
y_test_true = test_ds.y.astype(int)
budgets = [int(x) for x in _csv_list(args_obj.budgets)]
acc_grid = [float(x) for x in _csv_list(args_obj.target_acc_grid)]
target_acc_concepts = _csv_kv_float(args_obj.target_acc_concepts)

vec = None
km = None
P_tr_m = None
P_te_m = None
H_te_m = None
names_m = None
truth_map = None
if args_obj.concept_source == "machine":
    vec, Xtr = _tfidf_fit(train_ds.X, int(args_obj.machine_seed) if int(args_obj.machine_seed) > 0 else SEED)
    Xte = vec.transform([str(t) for t in test_ds.X])
    km = _kmeans_fit(Xtr, int(args_obj.machine_k), int(args_obj.machine_seed) if int(args_obj.machine_seed) > 0 else SEED+11)
    P_tr_m = _kmeans_soft(Xtr, km)
    P_te_m = _kmeans_soft(Xte, km)
    H_tr_m = np.eye(int(args_obj.machine_k), dtype=int)[np.argmin(km.transform(Xtr), axis=1)]
    H_te_m = np.eye(int(args_obj.machine_k), dtype=int)[np.argmin(km.transform(Xte), axis=1)]
    names_m = [f"machine_{j}" for j in range(int(args_obj.machine_k))]
    if int(args_obj.machine_upper_bound):
        truth_map = _machine_truth_map(H_tr_m, train_ds.C.astype(int))
    fe_machine = FrontEndModel()
    if int(args_obj.machine_soft):
        fe_machine.fit(P_tr_m, train_ds.y.astype(int))
    else:
        fe_machine.fit(H_tr_m, train_ds.y.astype(int))

acc_map = _csv_kv_float(args_obj.human_acc_concepts)
rows = []
rng = np.random.default_rng(SEED)

def _choose_source():
    if args_obj.concept_source == "detected":
        names_vec = list(test_ds.concepts)
        U_full = C_test_scores * (1 - C_test_scores)
        H_base = H_test
        T_truth = T_test
        fe = cbm.front_end_model
        return names_vec, U_full, H_base, T_truth, fe
    if args_obj.concept_source == "gt":
        names_vec = list(test_ds.concepts)
        U_full = np.zeros_like(T_test, dtype=float)
        H_base = T_test.copy()
        T_truth = T_test
        fe_gt = FrontEndModel()
        fe_gt.fit(train_ds.C.astype(int), train_ds.y.astype(int))
        return names_vec, U_full, H_base, T_truth, fe_gt
    names_vec = names_m
    U_full = (P_te_m if int(args_obj.machine_soft) else H_te_m.astype(float)) * (1.0 - (P_te_m if int(args_obj.machine_soft) else H_te_m.astype(float)))
    H_base = (H_te_m if not int(args_obj.machine_soft) else (P_te_m > 0.5).astype(int))
    if int(args_obj.machine_upper_bound) and truth_map is not None:
        T_truth = T_test[:, truth_map]
    else:
        T_truth = H_base.copy()
    fe = fe_machine
    return names_vec, U_full, H_base, T_truth, fe

names_vec, U_full_src, H_test_src, T_truth_src, fe_src = _choose_source()
allow_idxs = _allowed_indices(names_vec, args_obj.intervene_allow)

bb_acc = None
if args_obj.blackbox_metrics and Path(args_obj.blackbox_metrics).is_file():
    _m = json.loads(Path(args_obj.blackbox_metrics).read_text())
    bb_acc = float(_m.get("accuracy", _m.get("acc_test", 0.0)))

for ta in acc_grid:
    H0 = _degrade_to_acc(H_test_src, T_truth_src, ta, SEED)
    if target_acc_concepts:
        H0 = _apply_per_concept_degrade(H0, T_truth_src, names_vec, target_acc_concepts, SEED+99)
    for k in budgets:
        Hm = H0.copy()
        if k > 0:
            if args_obj.policy == "uncertainty":
                U = U_full_src.copy()
                mask = np.ones(U.shape[1], dtype=bool)
                if allow_idxs.size > 0:
                    mask[:] = False
                    mask[allow_idxs] = True
                U[:, ~mask] = -1.0
                topk = np.argsort(-U, axis=1)[:, :k]
                for i in range(Hm.shape[0]):
                    sel = [j for j in topk[i] if j < Hm.shape[1] and (allow_idxs.size == 0 or j in allow_idxs)]
                    Hm[i] = _apply_human_edit(Hm[i], T_truth_src[i], sel, names_vec, float(args_obj.human_acc), acc_map, rng)
            else:
                mask = np.ones(Hm.shape[1], dtype=bool)
                if allow_idxs.size > 0:
                    mask[:] = False
                    mask[allow_idxs] = True
                for i in range(Hm.shape[0]):
                    errs = np.where((Hm[i] != T_truth_src[i]) & mask)[0]
                    if errs.size > 0:
                        sel = errs[:k]
                        Hm[i] = _apply_human_edit(Hm[i], T_truth_src[i], sel, names_vec, float(args_obj.human_acc), acc_map, rng)
        y_proba = fe_src.predict_proba(Hm)
        y_pred = np.argmax(y_proba, axis=1)
        acc_k = float(accuracy_score(y_test_true, y_pred))
        rec = {"target_acc": ta, "budget": k, "acc_cbm_intv": acc_k, "concept_checks": int(k * Hm.shape[0]), "concept_source": args_obj.concept_source}
        if bb_acc is not None:
            rec["delta_vs_blackbox"] = acc_k - bb_acc
        rows.append(rec)

viab = pd.DataFrame(rows)
viab_path = ROBOT_RUN_DIR / "viability_metrics.csv"
viab.to_csv(viab_path, index=False)
print("Saved intervention metrics:", viab_path)

if int(args_obj.make_plots):
    for ta in acc_grid:
        for src in sorted(viab["concept_source"].unique()):
            sub = viab[(viab["target_acc"] == ta) & (viab["concept_source"] == src)]
            xs = sub["concept_checks"].values
            ys = sub["acc_cbm_intv"].values
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("ConceptChecks")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy vs ConceptChecks (target_acc={ta}) [{src}]")
            plt.tight_layout()
            plt.savefig(ROBOT_RUN_DIR / f"acc_vs_checks_{ta}_{src}.png")
            plt.close()
        if bb_acc is not None:
            for src in sorted(viab["concept_source"].unique()):
                sub = viab[(viab["target_acc"] == ta) & (viab["concept_source"] == src)]
                xs = sub["concept_checks"].values
                ys = sub["delta_vs_blackbox"].values
                plt.figure()
                plt.plot(xs, ys, marker="o")
                plt.xlabel("ConceptChecks")
                plt.ylabel("CBM − BlackBox")
                plt.title(f"Gain vs ConceptChecks (target_acc={ta}) [{src}]")
                plt.tight_layout()
                plt.savefig(ROBOT_RUN_DIR / f"gain_vs_checks_{ta}_{src}.png")
                plt.close()
