import csv
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from concept_benchmark.synthetic.robot_concepts.textgen import create_synthetic_dataset
from concept_benchmark.paths import pkg_dir
from concept_benchmark.synthetic.robot_concepts.text_concept_detector import TextConceptDetector
from concept_benchmark.models import ConceptBasedModel, FrontEndModel

tpl_path = pkg_dir / "synthetic" / "robot_concepts" / "static" / "text_templates" / "Templates.txt"
with open(tpl_path, "r", encoding="utf-8-sig") as f:
    templates = [ln.strip() for ln in f if ln.strip()]

params = {
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "foot_shape": ["flat_4sided", "flat_5sided", "flat_lshaped", "pointy_3sided", "pointy_4sided", "pointy_6sided"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": ["round_circle","wide_oval", "tall oval", "edgy_square","edgy_triangle","edgy_trapezoid"], #round oval is oval1 and wide oval is oval2
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
    SAFE_GLOBALS = {"__builtins__": None, "int": int, "str": str, "float": float, "bool": bool, "any": any, "all": all}
    def eval_one(sr):
        row = sr.to_dict()
        return eval(model_expr, SAFE_GLOBALS, {"row": row})
    return df.apply(eval_one, axis=1)

catalog_df = sample_concepts(params, n=500, seed=0)
catalog_df["label"] = compute_label(catalog_df, params["model"])

concept_cols = list(params["concepts"].keys())
llm_user_prompt = "Using the provided attributes, write a natural spoken description (1–3 sentences) that sounds like a person describing an image they saw. Do not invent locations or scenarios; focus only on what the attributes imply."

ds = create_synthetic_dataset(source=catalog_df, templates=templates, variants_per_row=1, include_color=False, rng_seed=0, concept_cols=concept_cols, label_col="label", label_map={"drent": 0, "glorp": 1}, text_mode="semi", llm_provider="gemini", llm_model="gemini-1.5-flash", llm_user_prompt=llm_user_prompt)

print("SAMPLE CAPTIONS:")
for x in ds.X[:6]:
    print("-", x)

print("\nCONCEPT NAMES:", ds.concepts)
print("CLASSES:", ds.classes)
print("N samples:", len(ds))

out_csv = Path("outputs/text_samples.csv")
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

detector = TextConceptDetector(embed_dim=128, hidden_dim=192, epochs=6, batch_size=64, use_bigrams=True, lr=2e-3, dropout=0.1)

cbm = ConceptBasedModel(concept_detector=detector, front_end_model=FrontEndModel(), propagate=False)

cbm.fit(ds.training, ds.validation)

with np.errstate(invalid="ignore"):
    C_val_true = ds.validation.C.astype(np.float32)
    C_val_prob = detector.predict(ds.validation)

concept_names = list(ds.concepts)
per = {}
for j, name in enumerate(concept_names):
    yt = C_val_true[:, j]
    yp = C_val_prob[:, j]
    try:
        auprc = average_precision_score(yt, yp)
    except Exception:
        auprc = float("nan")
    try:
        rocauc = roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else 0.5
    except Exception:
        rocauc = float("nan")
    per[name] = {"auprc": float(auprc), "roc_auc": float(rocauc)}

auprc_macro = float(np.nanmean([d["auprc"] for d in per.values()])) if per else float("nan")
roc_macro = float(np.nanmean([d["roc_auc"] for d in per.values()])) if per else float("nan")

print("Macro metrics:", {"auprc_macro": auprc_macro, "roc_auc_macro": roc_macro})
first5_keys = list(per.keys())[:5]
print("Sample per-concept metric (first 5):", {k: per[k] for k in first5_keys})

texts_demo = [str(x) for x in ds.X[:3]]
from concept_benchmark.data import ConceptDatasetSample
dummy_C = np.zeros((len(texts_demo), len(ds.concepts)), dtype=np.float32)
dummy_y = np.zeros((len(texts_demo),), dtype=int)
demo_ds = ConceptDatasetSample(X=texts_demo, C=dummy_C, y=dummy_y, meta={"concepts": ds.concepts, "classes": ds.classes, "data_type": "text"})

proba = detector.predict(demo_ds)
print("Concept order:", concept_names)
print("Proba shape:", proba.shape)
print("First row probs:", proba[0])

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