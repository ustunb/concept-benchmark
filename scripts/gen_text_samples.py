import csv
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import datetime

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from concept_benchmark.synthetic.helper.textgen import create_synthetic_dataset
from concept_benchmark.paths import pkg_dir, results_dir, data_dir
from concept_benchmark.synthetic.helper.text_concept_detector import TextConceptDetector
from concept_benchmark.models import ConceptBasedModel, FrontEndModel
from concept_benchmark.ext.fileutils import save as save_obj
from concept_benchmark.metrics import calc_metric
from concept_benchmark.data import ConceptDatasetSample

tpl_path = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "Templates.txt"
with open(tpl_path, "r", encoding="utf-8-sig") as f:
    templates = [ln.strip() for ln in f if ln.strip()]

ROBOT_RUN_DIR = results_dir / "robot_text"
ROBOT_DATA_DIR = data_dir / "robot_text"
ROBOT_RUN_DIR.mkdir(parents=True, exist_ok=True)
ROBOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

CONCEPT_MODE = os.environ.get("CONCEPT_MODE", "hard").strip().lower()
if CONCEPT_MODE not in {"hard", "soft"}:
    CONCEPT_MODE = "hard"

_train_on_detected = os.environ.get("TRAIN_ON_DETECTED", "false").strip().lower()
train_on_detected = _train_on_detected in {"1", "true", "yes", "y"}

params = {
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "foot_shape": [
            "flat_4sided",
            "flat_5sided",
            "flat_lshaped",
            "pointy_3sided",
            "pointy_4sided",
            "pointy_6sided",
        ],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": [
            "round_circle",
            "wide_oval",
            "tall_oval",
            "edgy_square",
            "edgy_triangle",
            "edgy_trapezoid",
        ],
    },
    "model": "'glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')) - 2 >= 0) else 'drent'",
}


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
    SAFE_GLOBALS = {
        "__builtins__": None,
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
        "any": any,
        "all": all,
    }

    def eval_one(sr):
        row = sr.to_dict()
        return eval(model_expr, SAFE_GLOBALS, {"row": row})

    return df.apply(eval_one, axis=1)


catalog_df = sample_concepts(params, n=10000, seed=0)
catalog_df["label"] = compute_label(catalog_df, params["model"])

concept_cols = list(params["concepts"].keys())
llm_user_prompt = (
    "Using the provided attributes, write a natural spoken description (1–3 sentences) "
    "that sounds like a person describing an image they saw. Do not invent locations "
    "or scenarios; focus only on what the attributes imply."
)

ds = create_synthetic_dataset(
    source=catalog_df,
    templates=templates,
    variants_per_row=1,
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

if ds.cvindices is None or getattr(ds.validation, "n", 0) == 0:
    ds.generate_cvindices(total_folds_for_cv=[5], replicates=1, seed=0)
    ds.split(fold_id=list(ds.cvindices.keys())[0], fold_num_validation=1, fold_num_test=None)

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
    validate=True,
)


print("Fitting detector")
detector.fit(ds.training, ds.validation)

cbm = ConceptBasedModel(concept_detector=detector, front_end_model=FrontEndModel(), propagate=False)

C_train = ds.training.C
y_train = ds.training.y

if train_on_detected:
    print("Training front-end on detected (noisy) concepts (train_on_detected=True).")
    old_mode = getattr(detector, "output_mode", None)
    try:
        if hasattr(detector, "output_mode"):
            detector.output_mode = "soft"
        C_train_used = detector.predict(ds.training)
    finally:
        if hasattr(detector, "output_mode") and old_mode is not None:
            detector.output_mode = old_mode
else:
    print("Training front-end on ground-truth concepts (train_on_detected=False).")
    C_train_used = C_train

cbm.front_end_model.fit(C_train_used, y_train)

with np.errstate(invalid="ignore"):
    C_val_true = ds.validation.C.astype(np.float32)

_old = detector.output_mode
detector.output_mode = "soft"
C_val_scores = detector.predict(ds.validation)
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
demo_ds = ConceptDatasetSample(
    X=texts_demo,
    C=dummy_C,
    y=dummy_y,
    meta={"concepts": ds.concepts, "classes": ds.classes, "data_type": "text"},
)

proba = detector.predict(demo_ds)
print("Concept order:", concept_names)
print(f"Concept outputs (mode={CONCEPT_MODE}) shape:", proba.shape)
print("First row outputs:", proba[0])

y_val = ds.validation.y.astype(int)
y_val_proba = cbm.predict_proba(ds.validation)
y_val_pred = np.argmax(y_val_proba, axis=1)

acc = accuracy_score(y_val, y_val_pred)
try:
    cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    roc = roc_auc_score(y_val, y_val_proba[:, cls_index_1]) if len(np.unique(y_val)) == 2 else float("nan")
except Exception:
    roc = float("nan")

print("Label model metrics (validation):", {"accuracy": float(acc), "roc_auc": float(roc)})

all_probs = cbm.predict_proba(ds)
all_preds = np.argmax(all_probs, axis=1)
label_names = list(ds.classes)

pred_labels = [label_names[i] for i in all_preds]
print("Pred labels:", pred_labels)
print("Class order in probs:", label_names)
print("First row class probs:", all_probs[0])

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

n_concepts = C_val_true.shape[1]

sel_covs, sel_accs = [], []
aucs, auprcs = [], []
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

run_info = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "concept_mode": CONCEPT_MODE,
    "pos_weight": "auto",
    "n_samples": int(len(ds.X)),
    "n_concepts": int(n_concepts),
    "classes": list(ds.classes),
    "concept_names": list(ds.concepts),
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

save_obj({"cbm": cbm, "detector": detector}, ROBOT_RUN_DIR / "model.pkl", overwrite=True)

print("Saved dataset:", ds_path)
print("Saved model:", ROBOT_RUN_DIR / "model.pkl")
print("Saved metrics:", ROBOT_RUN_DIR / "metrics.json")
