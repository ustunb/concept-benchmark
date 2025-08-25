import csv
from pathlib import Path
import pandas as pd
import numpy as np
from concept_benchmark.synthetic.robot_concepts.textgen import create_robot_text_dataset
from concept_benchmark.paths import repo_dir, pkg_dir
from concept_benchmark.synthetic.robot_concepts.train_text_concept_detector import train_concept_detector_text_multi, predict_proba_text_multi

tpl_path = pkg_dir/ "synthetic"/  "robot_concepts" / "static" / "text_templates" / "Templates.txt"
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
        "hand_shape": ["edgy_square", "edgy_triangle", "round_circle", "round_oval"],
    },
    "model": "'glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')) - 2 >= 0) else 'drent'",
}

def sample_concepts(params, n=100, seed=0):
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

catalog_df = sample_concepts(params, n=5000, seed=0)
catalog_df["label"] = compute_label(catalog_df, params["model"])

concept_cols = list(params["concepts"].keys())

llm_user_prompt = "Using the provided attributes, write a natural spoken description (1–3 sentences) that sounds like a person describing an image they saw. Do not invent locations or scenarios; focus only on what the attributes imply."

ds = create_robot_text_dataset(
    source=catalog_df,
    templates=templates,
    variants_per_row=1,
    include_color=False,
    rng_seed=0,
    concept_cols=concept_cols,
    label_col="label",
    label_map={"drent": 0, "glorp": 1},
    text_mode="semi",                      # "structured" | "semi" | "llm"
    llm_provider="gemini",
    llm_model="gemini-1.5-flash",
    llm_user_prompt=llm_user_prompt,
)

print("SAMPLE CAPTIONS:")
for x in ds.X[:6]:
    print("-", x)

print("\nCONCEPT NAMES:", ds.concepts)
print("CLASSES:", ds.classes)
print("N samples:", len(ds))

out_csv = Path("outputs/text_samples.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
row_index = getattr(ds, "_full").meta.get("row_index")

with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    cols = ["text"] + concept_cols + ["label"]
    w.writerow(cols)
    for i, x in enumerate(ds.X):
        src_idx = int(row_index[i])
        row_vals = catalog_df.loc[src_idx, concept_cols].tolist()
        w.writerow([x] + row_vals + [catalog_df.loc[src_idx, "label"]])

print(f"\nWrote {len(ds)} rows to {out_csv}")


if ds.cvindices is None or ds.validation.n == 0:
    ds.generate_cvindices(total_folds_for_cv=[5], replicates=1, seed=0)
    ds.split(fold_id=list(ds.cvindices.keys())[0], fold_num_validation=1, fold_num_test=None)

res = train_concept_detector_text_multi(
    dataset=ds,
    eval_split="validation",
    embed_dim=128,
    hidden_dim=192,
    epochs=6,
    batch_size=64,
    use_bigrams=True,
    lr=2e-3,
)

print("Macro metrics:", res.metrics_macro)
print("Sample per-concept metric (first 5):", {k:res.metrics_per_concept[k] for k in res.concepts[:5]})

proba = predict_proba_text_multi(res.model, res.vocab, [str(x) for x in ds.X[:3]])
print("Proba shape:", proba.shape)
print("First row probs (concept order = ds.concepts):", proba[0])