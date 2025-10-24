import argparse, json
import numpy as np
import torch
from torchvision import transforms
from transformers import ViTModel
from pathlib import Path
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from concept_benchmark.models import ConceptDetector, ConceptBasedModel, FrontEndModel
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_robot_image_dataset
from concept_benchmark.data import ConceptDatasetSample

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

settings = {
    "samples_per_instance": 1,
    "draw": False,
    "output_directory": str(results_dir / "robots"),
    "size": "large",
    "color_mode": "color",
    "verbose": True,
    "epochs": 10,
    "freeze": True,
    "embed_device": "auto",
    "fit_device": "cpu",
    "seed": 42,
    "label_model_expr": "",
    "corr_pair": "",
    "train_corr": 1.0,
    "test_break": 1.0,
    "test_corr": -1.0,
    "target_acc": -1.0,
    "target_acc_grid": "0.7,0.8,0.9,0.95",
    "target_acc_concepts": "",
    "intervene_allow": "",
    "human_acc": 1.0,
    "human_acc_concepts": "",
    "skew_concept": "",
    "budgets": "0,1,2,5,10",
    "make_plots": 1,
    "policy": "uncertainty",
    "concept_source": "detected",
    "machine_k": 16,
    "machine_soft": 1,
    "machine_upper_bound": 0,
    "blackbox_metrics": ""
}

def _csv_list(s):
    s = str(s).strip()
    return [t.strip() for t in s.split(",")] if s else []

def _csv_kv_float(s):
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

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--samples-per-instance", type=int)
p.add_argument("--draw", type=int)
p.add_argument("--output-directory", type=str)
p.add_argument("--size", type=str)
p.add_argument("--color-mode", type=str)
p.add_argument("--verbose", type=int)
p.add_argument("--epochs", type=int)
p.add_argument("--freeze", type=int)
p.add_argument("--embed-device", type=str)
p.add_argument("--fit-device", type=str)
p.add_argument("--seed", type=int)
p.add_argument("--label-model-expr", type=str)
p.add_argument("--corr-pair", type=str)
p.add_argument("--train-corr", type=float)
p.add_argument("--test-break", type=float)
p.add_argument("--test-corr", type=float)
p.add_argument("--target-acc", type=float)
p.add_argument("--target-acc-grid", type=str)
p.add_argument("--target-acc-concepts", type=str)
p.add_argument("--intervene-allow", type=str)
p.add_argument("--human-acc", type=float)
p.add_argument("--human-acc-concepts", type=str)
p.add_argument("--skew-concept", type=str)
p.add_argument("--budgets", type=str)
p.add_argument("--make-plots", type=int)
p.add_argument("--policy", choices=["uncertainty","oracle"])
p.add_argument("--concept-source", choices=["detected","gt","machine"])
p.add_argument("--machine-k", type=int)
p.add_argument("--machine-soft", type=int)
p.add_argument("--machine-upper-bound", type=int)
p.add_argument("--blackbox_metrics", type=str)
args, _ = p.parse_known_args()
for k, v in vars(args).items():
    if v is not None:
        settings[k.replace("-", "_")] = v if k not in ("draw",) else bool(v)
if float(settings.get("test_corr", -1.0)) >= 0:
    settings["test_break"] = max(0.0, min(1.0, 1.0 - float(settings["test_corr"])))

params = {
    "samples_per_instance": int(settings["samples_per_instance"]),
    "draw": bool(settings["draw"]),
    "output_directory": Path(settings["output_directory"]),
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": ["round_circle", "round_oval", "round_oval2", "edgy_triangle", "edgy_square", "edgy_trapezoid"],
        "foot_shape": ["flat_4sided", "flat_5sided", "flat_lshaped", "pointy_3sided", "pointy_4sided", "pointy_6sided"],
    },
    "spurious_features": ["has_elbows", "hand_shape"],
    "model": "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
    "model_type": "deterministic",
    "size": settings["size"],
    "color_mode": settings["color_mode"],
    "train_concept_detector": False,
    "epochs": int(settings["epochs"]),
    "verbose": bool(settings["verbose"]),
}
if settings["label_model_expr"]:
    params["model"] = settings["label_model_expr"]

IMG_SIZE = 224
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

rng = np.random.default_rng(int(settings["seed"]))
data = create_robot_image_dataset(**params)
data.transform = tf
data.generate_cvindices(seed=int(settings["seed"]))
data.split("K05N01", fold_num_validation=4, fold_num_test=5)

def _subset_sample(sample: ConceptDatasetSample, keep_idx: np.ndarray) -> ConceptDatasetSample:
    keep_idx = np.asarray(keep_idx, dtype=int)
    X = [str(x) for x in np.array(sample.X, dtype=object)[keep_idx]]
    C = sample.C[keep_idx]
    y = sample.y[keep_idx]
    return ConceptDatasetSample(X=X, C=C, y=y, meta=sample.meta)

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
    return _subset_sample(sample, keep)

def _indices_for(names, spec):
    out = []
    spec = str(spec).strip()
    if "=" in spec:
        if spec in names:
            out.append(names.index(spec))
    else:
        out.extend([i for i, n in enumerate(names) if n.startswith(spec + "=")])
    return sorted(set(out))

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
    return _subset_sample(sample, keep)

train_sample = data.training
val_sample = data.validation
test_sample = data.test

outp = Path(settings["output_directory"]); outp.mkdir(parents=True, exist_ok=True)
cols = list(params["concepts"].keys())
catalog_df = pd.DataFrame(list(product(*[params["concepts"][c] for c in cols])), columns=cols)
catalog_csv = outp / "robots_catalog.csv"; catalog_df.to_csv(catalog_csv, index=False)

def _row_vals(sample, r):
    T = sample.C.astype(int)[r]
    vals = []
    for key in cols:
        idx = [i for i, n in enumerate(sample.concepts) if n.startswith(key + "=")]
        j = int(T[idx].argmax()); vals.append(sample.concepts[idx[j]].split("=", 1)[1])
    return tuple(vals)

index_map = {tuple(row): i for i, row in catalog_df[cols].itertuples(index=False, name=None)}

def _ids(sample):
    ids = []
    for r in range(sample.C.shape[0]):
        ids.append(index_map[_row_vals(sample, r)])
    return sorted(set(int(x) for x in ids))

df_indices = {"train": _ids(train_sample), "valid": _ids(val_sample), "test": _ids(test_sample)}
meta = {"catalog_csv": str(catalog_csv.resolve()), "df_indices": df_indices}
meta_path = outp / "image_meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Wrote image_meta:", meta_path)


if settings["corr_pair"]:
    train_sample = _enforce_corr(train_sample, settings["corr_pair"], float(settings["train_corr"]), int(settings["seed"]))
    test_sample = _enforce_corr(test_sample, settings["corr_pair"], max(0.0, 1.0 - float(settings["test_break"])), int(settings["seed"]) + 1)

if settings["skew_concept"]:
    train_sample = _apply_skew(train_sample, settings["skew_concept"], int(settings["seed"]))
    test_sample = _apply_skew(test_sample, settings["skew_concept"], int(settings["seed"]) + 2)

class ViTWrapper(torch.nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.vit = model if model else ViTModel.from_pretrained("google/vit-base-patch16-224")
    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]

model = ConceptDetector(embedding_model=ViTWrapper())

embed_dev = device if settings["embed_device"] in ("auto", "", None) else torch.device(settings["embed_device"])
fit_dev = device if settings["fit_device"] in ("auto", "", None) else torch.device(settings["fit_device"])

model.fit(
    train_sample,
    val_sample,
    freeze=bool(settings["freeze"]),
    embed_params={"device": embed_dev},
    fit_params={"epochs": int(settings["epochs"]), "device": str(fit_dev)}
)

probs_tr = model.predict(train_sample, embed_params={"device": embed_dev}).astype(float)
probs = model.predict(test_sample, embed_params={"device": embed_dev}).astype(float)
H_test = (probs > 0.5).astype(int)
T_test = test_sample.C.astype(int)

def _degrade_to_acc(H: np.ndarray, T: np.ndarray, target: float, seed: int) -> np.ndarray:
    H = H.copy()
    correct = (H == T).reshape(-1)
    cur = float(correct.mean())
    if target < 0 or target >= cur:
        return H
    need = int(round((cur - target) * H.size))
    idx = np.where(correct)[0]
    if need > idx.size:
        need = idx.size
    if need > 0:
        rng = np.random.default_rng(seed)
        sel = rng.choice(idx, size=need, replace=False)
        flat = H.reshape(-1)
        flat[sel] = 1 - flat[sel]
        H = flat.reshape(H.shape)
    return H

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

def _allowed_indices(names, allow_spec):
    if not allow_spec: return np.arange(len(names), dtype=int)
    out = []
    for tok in _csv_list(allow_spec):
        out.extend(_indices_for(names, tok))
    return np.array(sorted(set(out)), dtype=int)

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

def _machine_from_probs(P_tr, P_te, K):
    km = KMeans(n_clusters=int(K), n_init=10, random_state=int(settings["seed"])+31)
    km.fit(P_tr)
    Dtr = km.transform(P_tr); Dte = km.transform(P_te)
    Str = np.exp(-Dtr); Ste = np.exp(-Dte)
    Str /= (Str.sum(1, keepdims=True) + 1e-12)
    Ste /= (Ste.sum(1, keepdims=True) + 1e-12)
    Htr = np.eye(int(K))[np.argmin(Dtr, axis=1)]
    Hte = np.eye(int(K))[np.argmin(Dte, axis=1)]
    names = [f"machine_{i}" for i in range(int(K))]
    return Str, Ste, Htr.astype(int), Hte.astype(int), names

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

pred_bin = _degrade_to_acc(H_test, T_test, float(settings["target_acc"]), int(settings["seed"]))
acc_per_concept = (pred_bin == T_test).mean(axis=0)
print("Concept-wise accuracy:", acc_per_concept)

cbm = ConceptBasedModel(concept_detector=model, front_end_model=FrontEndModel(), propagate=False)
cbm.front_end_model.fit(train_sample.C, train_sample.y)
y_test_true = test_sample.y.astype(int)

P_tr_m, P_te_m, H_tr_m, H_te_m, names_m = None, None, None, None, None
truth_map = None
if settings["concept_source"] == "machine":
    P_tr_m, P_te_m, H_tr_m, H_te_m, names_m = _machine_from_probs(probs_tr, probs, int(settings["machine_k"]))
    if int(settings["machine_upper_bound"]):
        truth_map = _machine_truth_map(H_tr_m, train_sample.C.astype(int))
    fe_m = FrontEndModel()
    if int(settings["machine_soft"]):
        fe_m.fit(P_tr_m, train_sample.y.astype(int))
    else:
        fe_m.fit(H_tr_m, train_sample.y.astype(int))

target_acc_concepts = _csv_kv_float(settings["target_acc_concepts"])
acc_map = _csv_kv_float(settings["human_acc_concepts"])
rng2 = np.random.default_rng(int(settings["seed"]) + 123)
names_vec = list(test_sample.concepts) if settings["concept_source"] in ("detected","gt") else [f"machine_{j}" for j in range(int(settings["machine_k"]))]
acc_grid = [float(x) for x in _csv_list(settings["target_acc_grid"])]
budgets = [int(x) for x in _csv_list(settings["budgets"])]

def _choose_source():
    if settings["concept_source"] == "detected":
        U_full = probs * (1 - probs)
        H_base = (probs > 0.5).astype(int)
        T_truth = T_test
        fe = cbm.front_end_model
        return names_vec, U_full, H_base, T_truth, fe
    if settings["concept_source"] == "gt":
        U_full = np.zeros_like(T_test, dtype=float)
        H_base = T_test.copy()
        T_truth = T_test
        fe_gt = FrontEndModel()
        fe_gt.fit(train_sample.C.astype(int), train_sample.y.astype(int))
        return names_vec, U_full, H_base, T_truth, fe_gt
    U_full = (P_te_m if int(settings["machine_soft"]) else H_te_m.astype(float)) * (1.0 - (P_te_m if int(settings["machine_soft"]) else H_te_m.astype(float)))
    H_base = (H_te_m if not int(settings["machine_soft"]) else (P_te_m > 0.5).astype(int))
    if int(settings["machine_upper_bound"]) and truth_map is not None:
        T_truth = T_test[:, truth_map]
    else:
        T_truth = H_base.copy()
    fe = fe_m
    return names_vec, U_full, H_base, T_truth, fe

names_vec, U_full_src, H_test_src, T_truth_src, fe_src = _choose_source()
allow_idxs = _allowed_indices(names_vec, settings["intervene_allow"])

rows = []
for ta in acc_grid:
    H0 = _degrade_to_acc(H_test_src, T_truth_src, ta, int(settings["seed"]))
    if target_acc_concepts:
        H0 = _apply_per_concept_degrade(H0, T_truth_src, names_vec, target_acc_concepts, int(settings["seed"]) + 77)
    for k in budgets:
        Hm = H0.copy()
        if k > 0:
            if settings["policy"] == "uncertainty":
                U = U_full_src.copy()
                mask = np.ones(U.shape[1], dtype=bool)
                if allow_idxs.size > 0:
                    mask[:] = False
                    mask[allow_idxs] = True
                U[:, ~mask] = -1.0
                topk = np.argsort(-U, axis=1)[:, :k]
                for i in range(Hm.shape[0]):
                    sel = [j for j in topk[i] if mask[j]]
                    Hm[i] = _apply_human_edit(Hm[i], T_truth_src[i], sel, names_vec, float(settings["human_acc"]), acc_map, rng2)
            else:
                mask = np.ones(Hm.shape[1], dtype=bool)
                if allow_idxs.size > 0:
                    mask[:] = False
                    mask[allow_idxs] = True
                for i in range(Hm.shape[0]):
                    errs = np.where((Hm[i] != T_truth_src[i]) & mask)[0]
                    if errs.size > 0:
                        sel = errs[:k]
                        Hm[i] = _apply_human_edit(Hm[i], T_truth_src[i], sel, names_vec, float(settings["human_acc"]), acc_map, rng2)
        y_proba = fe_src.predict_proba(Hm)
        y_pred = np.argmax(y_proba, axis=1)
        acc_k = float((y_pred == y_test_true).mean())
        rows.append({"target_acc": ta, "budget": k, "acc_cbm_intv": acc_k, "concept_checks": int(k * Hm.shape[0]), "concept_source": settings["concept_source"]})

viab = pd.DataFrame(rows)
viab_path = Path(settings["output_directory"]) / "viability_metrics.csv"
viab.to_csv(viab_path, index=False)
print("Saved intervention metrics:", viab_path)

if int(settings["make_plots"]):
    outp = Path(settings["output_directory"])
    outp.mkdir(parents=True, exist_ok=True)
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
            plt.savefig(outp / f"acc_vs_checks_{ta}_images_{src}.png")
            plt.close()
