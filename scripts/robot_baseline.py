from __future__ import annotations

import argparse, json, time, random, re
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import torch.nn.functional as F

from concept_benchmark.paths import results_dir, pkg_dir
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.synthetic.helper.textgen import create_synthetic_dataset as make_text_ds

settings = {
    "modality": "text",
    "n": 5000,
    "seed": 1337,
    "out_dir": str(results_dir / "robot_baseline"),
    "label_model_expr": "",
    "label_model_type": "deterministic",
    "label_model_alpha": 10.0,
    "label_model_bias": -0.2,
    "corr_pair": "",
    "train_corr": 1.0,
    "test_break": 1.0,
    "epochs": 3,
    "batch_size": 16,
    "lr": 5e-5,
    "text_model": "distilbert-base-uncased",
    "image_model": "google/vit-base-patch16-224-in21k",
    "image_size": 224,
    "color_mode": "rgb",
    "samples_per_instance": 3,
    "draw": 0,
    "run_name": "",
    "templates_file": "",
    "template_difficulty": "hard",
}

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def compute_label(df: pd.DataFrame, model_expr: str,
                  label_model_type: str = "deterministic",
                  alpha: float = 10.0, bias: float = -0.2, seed: int = 0) -> pd.Series:
    SAFE_GLOBALS = {
        "__builtins__": None,
        "int": int, "str": str, "float": float, "bool": bool,
        "any": any, "all": all, "np": np,
        "min": min, "max": max
    }
    rng = np.random.default_rng(int(seed))

    def _cond_to_score(expr: str) -> str | None:
        m = re.search(r"\bif\s+(?P<cond>.+?)\s+else\b", expr)
        cond = m.group("cond").strip() if m else expr.strip()
        m2 = re.search(r"^(?P<lhs>.+?)(?:\s*(?:>=|<=|>|<)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$",
                       cond, flags=re.IGNORECASE)
        lhs = m2.group("lhs").strip() if m2 else cond
        while lhs.startswith("(") and lhs.endswith(")"):
            lvl = 0; ok = True
            for ch in lhs:
                if ch == "(": lvl += 1
                elif ch == ")":
                    lvl -= 1
                    if lvl < 0: ok = False; break
            if ok and lvl == 0:
                lhs = lhs[1:-1].strip()
            else:
                break
        return lhs or None

    score_expr = _cond_to_score(model_expr) if label_model_type == "stochastic" else None

    def eval_one(sr):
        row = sr.to_dict()
        if label_model_type is None or label_model_type == "deterministic":
            return eval(model_expr, SAFE_GLOBALS, {"row": row})
        score = None
        if score_expr:
            try:
                score = float(eval(score_expr, SAFE_GLOBALS, {"row": row}))
            except Exception:
                score = None
        if score is None:
            try:
                hard = eval(model_expr, SAFE_GLOBALS, {"row": row})
                score = 1.0 if str(hard).strip().lower() == "glorp" else 0.0
            except Exception:
                score = 0.0
        p = 1.0 / (1.0 + float(np.exp(-float(alpha) * (float(score) - float(bias)))))
        return "glorp" if rng.random() < p else "drent"

    return df.apply(eval_one, axis=1).astype(str)

def _signals_from_row(row: dict) -> dict:
    head = str(row["head_shape"])
    body = str(row["body_shape"])
    return {
        "head_body_same": (head == body),
        "has_antennae_bool": (str(row["has_antennae"]).lower() == "true"),
        "corners_head": (head == "square"),
        "corners_body": (body == "square"),
        "rounded_head": (head == "round"),
        "rounded_body": (body == "round"),
        "ears_shape": str(row["ears_shape"]),
        "mouth_type": str(row["mouth_type"]),
    }

def _nat_from_tokens(row: dict) -> dict:
    head_nat = {"square": "boxy", "round": "dome-like"}[str(row["head_shape"])]
    body_nat = {"square": "sharp-cornered", "round": "barrel-smooth"}[str(row["body_shape"])]
    ears_nat = {"square": "square", "triangle": "pointy"}[str(row["ears_shape"])]
    mouth_nat = {"closed": "shut", "open": "open"}[str(row["mouth_type"])]
    hands_nat_map = {
        "round_circle": "round mitts", "wide_oval": "broad ovals", "tall_oval": "long ovals",
        "edgy_square": "square claws", "edgy_triangle": "triangular grippers", "edgy_trapezoid": "trapezoid claws",
    }
    feet_nat_map = {
        "flat_4sided": "flat four-sided pads", "flat_5sided": "pentagonal pads", "flat_lshaped": "L-shaped feet",
        "pointy_3sided": "three-point feet", "pointy_4sided": "four-point feet", "pointy_6sided": "hex-point feet",
    }
    hands_nat = hands_nat_map[str(row["hand_shape"])]
    feet_nat = feet_nat_map[str(row["foot_shape"])]
    ant_nat = "with antennae" if str(row["has_antennae"]).lower() == "true" else "no antennae"
    knees_nat = "has knees" if str(row["has_knees"]).lower() == "true" else "no knees"
    elbows_nat = "has elbows" if str(row["has_elbows"]).lower() == "true" else "no elbows"
    return {
        "HEAD_NAT": head_nat, "BODY_NAT": body_nat, "EARS_NAT": ears_nat, "MOUTH_NAT": mouth_nat,
        "HANDS_NAT": hands_nat, "FEET_NAT": feet_nat, "ANT_NAT": ant_nat,
        "KNEES_NAT": knees_nat, "ELBOWS_NAT": elbows_nat,
    }

def _line_matches(sig: dict, cond: dict) -> bool:
    for k, v in cond.items():
        if k == "any":
            continue
        if k not in sig:
            return False
        if isinstance(v, bool):
            if bool(sig[k]) != v:
                return False
        else:
            if str(sig[k]) != str(v):
                return False
    return True

def _load_jsonl(p: Path) -> list[dict]:
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"HardCorpus file is empty: {p}")
    if text.startswith("["):
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError("Top-level JSON is not a list")
        return arr
    items, plain_lines = [], []
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith("```"):
            continue
        try:
            items.append(json.loads(s))
        except json.JSONDecodeError:
            plain_lines.append(s)
    if items:
        return items
    if plain_lines:
        return [{"id": f"pt_{i:04d}", "when": {"any": True}, "text": s} for i, s in enumerate(plain_lines, 1)]
    raise ValueError(f"No valid JSON or plain-text lines found in {p}.")

def _render_from_corpus(row: dict, corpus: list[dict], seed: int) -> str:
    sig = _signals_from_row(row)
    cand = [it for it in corpus if _line_matches(sig, it.get("when", {}))]
    if not cand:
        cand = corpus
    key = f'{seed}:{row["head_shape"]}:{row["body_shape"]}:{row["foot_shape"]}:{row["ears_shape"]}:{row["mouth_type"]}:{row["hand_shape"]}:{row["has_antennae"]}:{row["has_knees"]}:{row["has_elbows"]}'
    idx = abs(hash(key)) % len(cand)
    txt = str(cand[idx]["text"])
    nat = _nat_from_tokens(row)
    for k, v in nat.items():
        ph = "{" + k + "}"
        if ph in txt:
            txt = txt.replace(ph, v)
    raw_map = {
        "head_shape": str(row["head_shape"]),
        "body_shape": str(row["body_shape"]),
        "ears_shape": str(row["ears_shape"]),
        "mouth_type": str(row["mouth_type"]),
        "hand_shape": str(row["hand_shape"]),
        "foot_shape": str(row["foot_shape"]),
        "has_antennae": str(row["has_antennae"]),
        "has_knees": str(row["has_knees"]),
        "has_elbows": str(row["has_elbows"]),
    }
    for k, v in raw_map.items():
        ph = "{" + k + "}"
        if ph in txt:
            txt = txt.replace(ph, v)
    return txt

def _names_from_concepts(concepts: dict) -> list[str]:
    names = []
    for k, vals in concepts.items():
        for v in vals:
            names.append(f"{k}={v}")
    return names

def _onehot_for_row(row: dict, concepts: dict, names: list[str]) -> np.ndarray:
    J = len(names)
    vec = np.zeros((J,), dtype=np.float32)
    pos = 0
    for k, vals in concepts.items():
        for v in vals:
            if str(row[k]) == str(v):
                vec[pos] = 1.0
            pos += 1
    return vec

class TextDS(Dataset):
    def __init__(self, X, y, tok, max_length=256):
        self.X = list(map(str, X))
        self.y = np.asarray(y, dtype=int)
        self.tok = tok
        self.max_length = max_length
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        enc = self.tok(
            self.X[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(self.y[i], dtype=torch.long)
        return enc, y

class ImageDS(Dataset):
    def __init__(self, X_paths, y, proc):
        self.X = [str(p) for p in X_paths]
        self.y = np.asarray(y, dtype=int)
        self.proc = proc
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        img = Image.open(self.X[i]).convert("RGB")
        enc = self.proc(images=img, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(self.y[i], dtype=torch.long)
        return enc, y

def _ensure_binary(y):
    u = np.unique(y)
    if u.size < 2:
        raise ValueError("Training set is single-class")

def train_eval_text(X_tr, y_tr, X_te, y_te, model_id, epochs, batch_size, lr, device):
    _ensure_binary(y_tr)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    ds_tr = TextDS(X_tr, y_tr, tok)
    ds_te = TextDS(X_te, y_te, tok)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in dl_tr:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb, labels=yb)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb)
            pred = out.logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    acc = correct / total if total > 0 else 0.0
    return float(acc), tok, model

def train_eval_image(paths_tr, y_tr, paths_te, y_te, model_id, size, epochs, batch_size, lr, device):
    _ensure_binary(y_tr)
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id, num_labels=2, ignore_mismatched_sizes=True)
    ds_tr = ImageDS(paths_tr, y_tr, proc)
    ds_te = ImageDS(paths_te, y_te, proc)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in dl_tr:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb, labels=yb)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb)
            pred = out.logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    acc = correct / total if total > 0 else 0.0
    return float(acc), proc, model

def _eval_text_metrics(X, y, tok, model, device):
    ds = TextDS(X, np.asarray(y, dtype=int), tok)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            out = model(**xb)
            p = out.logits.softmax(dim=-1)
            preds.append(p.argmax(dim=-1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
    y_true = np.asarray(y, dtype=int)
    y_pred = np.concatenate(preds) if preds else np.zeros_like(y_true)
    proba1 = np.concatenate(probs) if probs else np.zeros_like(y_true, dtype=float)
    acc = float((y_pred == y_true).mean()) if y_true.size else 0.0
    ba = float(balanced_accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0
    roc = float(roc_auc_score(y_true, proba1)) if np.unique(y_true).size == 2 else float("nan")
    return {"accuracy": acc, "balanced_acc": ba, "ber": float(1.0 - ba), "f1": f1, "roc_auc": roc}

def _eval_image_metrics(paths, y, proc, model, device):
    ds = ImageDS(paths, np.asarray(y, dtype=int), proc)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            out = model(**xb)
            p = out.logits.softmax(dim=-1)
            preds.append(p.argmax(dim=-1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
    y_true = np.asarray(y, dtype=int)
    y_pred = np.concatenate(preds) if preds else np.zeros_like(y_true)
    proba1 = np.concatenate(probs) if probs else np.zeros_like(y_true, dtype=float)
    acc = float((y_pred == y_true).mean()) if y_true.size else 0.0
    ba = float(balanced_accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0
    roc = float(roc_auc_score(y_true, proba1)) if np.unique(y_true).size == 2 else float("nan")
    return {"accuracy": acc, "balanced_acc": ba, "ber": float(1.0 - ba), "f1": f1, "roc_auc": roc}


p = argparse.ArgumentParser(add_help=False)
p.add_argument("--modality", choices=["text", "image"])
p.add_argument("--n", type=int)
p.add_argument("--seed", type=int)
p.add_argument("--out_dir")
p.add_argument("--label_model_expr")
p.add_argument("--label-model-expr", dest="label_model_expr")
p.add_argument("--label-model-type", dest="label_model_type", choices=["deterministic","stochastic"])
p.add_argument("--label-model-alpha", dest="label_model_alpha", type=float)
p.add_argument("--label-model-bias", dest="label_model_bias", type=float)
p.add_argument("--corr_pair")
p.add_argument("--train_corr", type=float)
p.add_argument("--test_break", type=float)
p.add_argument("--epochs", type=int)
p.add_argument("--batch_size", type=int)
p.add_argument("--lr", type=float)
p.add_argument("--text_model")
p.add_argument("--image_model")
p.add_argument("--image_size", type=int)
p.add_argument("--color_mode")
p.add_argument("--samples_per_instance", type=int)
p.add_argument("--variants-per-row-minority", dest="variants_per_row_minority", type=int)
p.add_argument("--variants-per-row-majority", dest="variants_per_row_majority", type=int)
p.add_argument("--minority_mult", dest="minority_mult", type=float)
p.add_argument("--draw", type=int)
p.add_argument("--run-name", dest="run_name", type=str)
p.add_argument("--templates-file", type=str)
p.add_argument("--template-difficulty", choices=["easy","medium","hard"])
p.add_argument("--redact-concepts", type=str)
p.add_argument("--redact-splits", type=str)
args, _ = p.parse_known_args()
for k, v in vars(args).items():
    if v is not None:
        settings[k] = v if k != "draw" else bool(v)

set_seed(int(settings["seed"]))
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

modality = settings["modality"]
seed_tag = f"seed{int(settings['seed'])}"
model_id = settings["text_model"] if modality == "text" else settings["image_model"]
model_tag = model_id.split("/")[-1]
ts = time.strftime("%Y%m%d_%H%M%S")
run_name = settings.get("run_name", "").strip()
run_folder = run_name if run_name else ts

out_dir = Path(settings["out_dir"]) / modality / run_folder
out_dir.mkdir(parents=True, exist_ok=True)

def build_text_ds_hard(catalog_df: pd.DataFrame,
                       concepts: dict,
                       corpus_path: Path,
                       variants_per_row: int,
                       seed: int,
                       row_variants: list[int] | None = None) -> ConceptDatasetSample:
    corpus = _load_jsonl(corpus_path)
    names = _names_from_concepts(concepts)
    classes = [0, 1]
    X, C, y, row_index = [], [], [], []
    for i, sr in catalog_df.iterrows():
        row = {k: sr[k] for k in concepts.keys()}
        row["label"] = sr["label"]
        _vpr_i = int(row_variants[i]) if (row_variants is not None and i < len(row_variants)) else int(variants_per_row)
        for v in range(max(1, _vpr_i)):
            text = _render_from_corpus(row, corpus, seed + v)
            X.append(text)
            C.append(_onehot_for_row(row, concepts, names))
            y.append(1 if str(row["label"]) == "glorp" else 0)
            row_index.append(i)
    ds = ConceptDatasetSample(
        X=X,
        C=np.asarray(C, dtype=np.float32),
        y=np.asarray(y, dtype=int),
        meta={"concepts": tuple(names), "classes": tuple(classes), "data_type": "text"}
    )
    setattr(ds, "_full", type("Full", (), {"meta": {"row_index": np.asarray(row_index, dtype=int)}}))
    return ds

if modality == "text":
    concepts = {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "foot_shape": ["flat_4sided","flat_5sided","flat_lshaped","pointy_3sided","pointy_4sided","pointy_6sided"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": ["round_circle","wide_oval","tall_oval","edgy_square","edgy_triangle","edgy_trapezoid"],
    }
    cols = list(concepts.keys())
    catalog_df = pd.DataFrame([dict(zip(cols, vals)) for vals in product(*[concepts[c] for c in cols])], columns=cols)
    label_expr = settings["label_model_expr"] or "'glorp' if (min(int(str(row['has_antennae']).lower()=='true'), int(row['body_shape']=='square')) >= 1) else 'drent'"
    catalog_df["label"] = compute_label(
        catalog_df,
        label_expr,
        label_model_type=settings.get("label_model_type", "deterministic"),
        alpha=float(settings.get("label_model_alpha", 10.0)),
        bias=float(settings.get("label_model_bias", -0.2)),
        seed=int(settings.get("seed", 0)),
    )
    _lbl = catalog_df["label"].astype(str)
    print("Label distribution (catalog_df):", {
        "glorp": int((_lbl == "glorp").sum()),
        "drent": int((_lbl == "drent").sum()),
        "total": int(len(_lbl)),
        "pos_frac": round((_lbl == "glorp").mean(), 4),
    })
    tpl_path = None
    if settings.get("templates_file"):
        tpl_path = Path(settings["templates_file"])
    else:
        if settings.get("template_difficulty","medium") == "hard":
            cand = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "HardCorpus.jsonl"
            tpl_path = cand if cand.is_file() else None
        if tpl_path is None:
            template_file_name = "Templates.txt" if settings.get("template_difficulty","medium") == "medium" else "Templates_simple.txt"
            tpl_path = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / template_file_name
    is_jsonl = str(tpl_path).lower().endswith(".jsonl")
    if is_jsonl:
        _base_vpr = int(settings["samples_per_instance"])
        _labels = catalog_df["label"].astype(str).tolist()
        _vals, _cnts = np.unique(_labels, return_counts=True)
        _minority_label = _vals[int(np.argmin(_cnts))] if len(_vals) > 0 else None
        _vpr_min = int(settings.get("variants_per_row_minority") or max(1, int(round(_base_vpr * float(settings.get("minority_mult", 1.0))))))
        _vpr_maj = int(settings.get("variants_per_row_majority") or _base_vpr)
        _row_variants = [(_vpr_min if (lab == _minority_label) else _vpr_maj) for lab in _labels]
        ds = build_text_ds_hard(
            catalog_df=catalog_df,
            concepts=concepts,
            corpus_path=tpl_path,
            variants_per_row=_base_vpr,
            seed=int(settings["seed"]),
            row_variants=_row_variants,
        )
    else:
        with open(tpl_path, "r", encoding="utf-8-sig") as f:
            templates = [ln.strip() for ln in f if ln.strip()]
        ds = make_text_ds(
            source=catalog_df,
            templates=templates,
            variants_per_row=max(1, settings["n"] // max(1, len(catalog_df))),
            include_color=False,
            rng_seed=int(settings["seed"]),
            concept_cols=cols,
            label_col="label",
            label_map={"drent": 0, "glorp": 1},
            text_mode="semi",
            llm_provider="gemini",
            llm_model="gemini-1.5-flash",
            llm_user_prompt="Describe the robot based only on attributes.",
        )
    row_index = getattr(getattr(ds, "_full", None), "meta", {}).get("row_index", None)
    if isinstance(row_index, np.ndarray) and len(row_index) == len(ds.X):
        rng = np.random.default_rng(0)
        base_ids = np.unique(row_index)
        rng.shuffle(base_ids)
        n_ids = len(base_ids)
        n_val_ids = int(np.floor(0.15 * n_ids))
        n_te_ids = int(np.floor(0.15 * n_ids))
        val_ids = set(base_ids[:n_val_ids])
        te_ids = set(base_ids[n_val_ids:n_val_ids + n_te_ids])
        mask_val = np.array([rid in val_ids for rid in row_index], dtype=bool)
        mask_te = np.array([rid in te_ids for rid in row_index], dtype=bool)
        mask_tr = ~(mask_val | mask_te)
        tr = np.where(mask_tr)[0]
        va = np.where(mask_val)[0]
        te = np.where(mask_te)[0]
    else:
        n = len(ds.X)
        idx = np.arange(n)
        rng = np.random.default_rng(int(settings["seed"]))
        rng.shuffle(idx)
        tr_end = int(0.70 * n)
        va_end = int(0.85 * n)
        tr, va, te = idx[:tr_end], idx[tr_end:va_end], idx[va_end:]
    def _subset(ds_obj, take):
        X = [ds_obj.X[i] for i in take]
        C = ds_obj.C[take]
        y = ds_obj.y[take]
        return ConceptDatasetSample(
            X=X, C=C, y=y,
            meta={"concepts": ds_obj.concepts, "classes": ds_obj.classes, "data_type": "text"}
        )
    ds.training = _subset(ds, tr)
    ds.validation = _subset(ds, va)
    ds.test = _subset(ds, te)
    rc = str(settings.get("redact_concepts", "") or "").strip().lower()
    rs = str(settings.get("redact_splits", "") or "").strip().lower()
    if rc and ("has_antennae" in {t.strip() for t in rc.split(",") if t.strip()}) and rs:
        if is_jsonl:
            base_jsonl = Path(settings.get("templates_file")) if settings.get("templates_file") else (
                        pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "HardCorpus.jsonl")
            cand = [
                base_jsonl.with_name(base_jsonl.stem + "_noANT" + base_jsonl.suffix),
                base_jsonl.parent / "HardCorpus_noANT.jsonl",
            ]
            tpl_noant = next((c for c in cand if c.is_file()), None)
            if tpl_noant is not None:
                corpus_noant = _load_jsonl(tpl_noant)
                newX = []
                for j, i_abs in enumerate(te):
                    rid = int(row_index[i_abs])
                    row = {k: catalog_df.loc[rid, k] for k in concepts.keys()}
                    txt = _render_from_corpus(row, corpus_noant, int(settings["seed"]) + j)
                    newX.append(txt)
                ds.test = ConceptDatasetSample(X=newX, C=ds.test.C, y=ds.test.y, meta=ds.test.meta)
        pat = re.compile(r"(?i)\b(?:with|has)\s+antennae\b|\bno\s+antennae\b|\bantenna(?:e|s)?\b")


        def _redact(lst):
            out = []
            for s in lst:
                z = re.sub(pat, " ", str(s))
                z = re.sub(r"\s{2,}", " ", z)
                z = re.sub(r"\s+([,.;:!?])", r"\1", z).strip()
                out.append(z)
            return out


        targets = {t.strip() for t in rs.split(",") if t.strip()}
        if "test" in targets:
            ds.test = ConceptDatasetSample(X=_redact(ds.test.X), C=ds.test.C, y=ds.test.y, meta=ds.test.meta)
        if "val" in targets:
            ds.validation = ConceptDatasetSample(X=_redact(ds.validation.X), C=ds.validation.C, y=ds.validation.y,
                                                 meta=ds.validation.meta)
        if "train" in targets:
            ds.training = ConceptDatasetSample(X=_redact(ds.training.X), C=ds.training.C, y=ds.training.y,
                                               meta=ds.training.meta)

    yt = np.asarray(ds.training.y, dtype=int)
    yv = np.asarray(ds.validation.y, dtype=int)
    yte = np.asarray(ds.test.y, dtype=int)
    print("Split sizes →", {"train": int(ds.training.n), "val": int(ds.validation.n), "test": int(ds.test.n)})
    print("Label distribution (train):", {
        "glorp": int((yt == 1).sum()),
        "drent": int((yt == 0).sum()),
        "total": int(yt.size),
        "pos_frac": round((yt == 1).mean() if yt.size else 0.0, 4),
    })
    print("Label distribution (val):", {
        "glorp": int((yv == 1).sum()),
        "drent": int((yv == 0).sum()),
        "total": int(yv.size),
        "pos_frac": round((yv == 1).mean() if yv.size else 0.0, 4),
    })
    print("Label distribution (test):", {
        "glorp": int((yte == 1).sum()),
        "drent": int((yte == 0).sum()),
        "total": int(yte.size),
        "pos_frac": round((yte == 1).mean() if yte.size else 0.0, 4),
    })

    pat_ant = re.compile(r"(?i)\bantenna(?:e|s)?(?:-like)?\b")
    counts = {
        "train": int(sum(1 for s in ds.training.X if pat_ant.search(str(s)))),
        "val": int(sum(1 for s in ds.validation.X if pat_ant.search(str(s)))),
        "test": int(sum(1 for s in ds.test.X if pat_ant.search(str(s))))
    }
    split_dump = {}
    for name, part in [("train", ds.training), ("val", ds.validation), ("test", ds.test)]:
        p = out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_split_{name}.csv"
        pd.DataFrame({"text": list(map(str, part.X)), "label": np.asarray(part.y, dtype=int)}).to_csv(p, index=False)
        split_dump[name] = str(p)
    print(json.dumps({
        "split_sizes": {"train": int(ds.training.n), "val": int(ds.validation.n), "test": int(ds.test.n)},
        "antenna_mentions": counts,
        "split_files": split_dump,
        "run_dir": str(out_dir)
    }, indent=2))
    if counts["test"] > 0:
        raise SystemExit(2)

    Xtr = ds.training.X
    ytr = ds.training.y.astype(int)
    Xte = ds.test.X
    yte = ds.test.y.astype(int)
    acc, tok_or_proc, model = train_eval_text(
        Xtr, ytr, Xte, yte,
        model_id=model_id,
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        lr=float(settings["lr"]),
        device=device,
    )
    _train_metrics = _eval_text_metrics(Xtr, ytr, tok_or_proc, model, device)
    _val_metrics = _eval_text_metrics(ds.validation.X, ds.validation.y.astype(int), tok_or_proc, model, device)
    _test_metrics = _eval_text_metrics(Xte, yte, tok_or_proc, model, device)
    print("Baseline test metrics:", {
        "accuracy": round(_test_metrics["accuracy"], 4),
        "balanced_acc": round(_test_metrics["balanced_acc"], 4),
        "ber": round(_test_metrics["ber"], 4),
        "f1": round(_test_metrics["f1"], 4),
        "roc_auc": (round(_test_metrics["roc_auc"], 4) if not np.isnan(_test_metrics["roc_auc"]) else "nan"),
    })


else:
    imgs_dir = pkg_dir / "synthetic" / "helper" / "static" / "robot_images_small"
    meta = pd.read_csv(imgs_dir / "meta.csv")
    meta = meta.sample(frac=1.0, random_state=int(settings["seed"])).reset_index(drop=True)
    label_expr = settings["label_model_expr"] or "'glorp' if (min(int(str(row['has_antennae']).lower()=='true'), int(row['body_shape']=='square')) >= 1) else 'drent'"
    meta["label"] = compute_label(
        meta,
        label_expr,
        label_model_type=settings.get("label_model_type", "deterministic"),
        alpha=float(settings.get("label_model_alpha", 10.0)),
        bias=float(settings.get("label_model_bias", -0.2)),
        seed=int(settings.get("seed", 0)),
    )
    _lbl_img = meta["label"].astype(str)
    print("Label distribution (images, full):", {
        "glorp": int((_lbl_img == "glorp").sum()),
        "drent": int((_lbl_img == "drent").sum()),
        "total": int(len(_lbl_img)),
        "pos_frac": round((_lbl_img == "glorp").mean(), 4),
    })
    n = min(int(settings["n"]), len(meta))
    meta = meta.iloc[:n].reset_index(drop=True)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    tr = meta.iloc[:n_tr]
    va = meta.iloc[n_tr:n_tr + n_va]
    te = meta.iloc[n_tr + n_va:]
    paths_tr = [imgs_dir / p for p in tr["path"]]
    ytr = tr["label"].map({"drent": 0, "glorp": 1}).astype(int).values
    paths_va = [imgs_dir / p for p in va["path"]]
    yva = va["label"].map({"drent": 0, "glorp": 1}).astype(int).values
    paths_te = [imgs_dir / p for p in te["path"]]
    yte = te["label"].map({"drent": 0, "glorp": 1}).astype(int).values
    print("Split sizes →", {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))})
    print("Label distribution (train):", {
        "glorp": int((ytr == 1).sum()),
        "drent": int((ytr == 0).sum()),
        "total": int(ytr.size),
        "pos_frac": round((ytr == 1).mean() if ytr.size else 0.0, 4),
    })
    print("Label distribution (val):", {
        "glorp": int((yva == 1).sum()),
        "drent": int((yva == 0).sum()),
        "total": int(yva.size),
        "pos_frac": round((yva == 1).mean() if yva.size else 0.0, 4),
    })
    print("Label distribution (test):", {
        "glorp": int((yte == 1).sum()),
        "drent": int((yte == 0).sum()),
        "total": int(yte.size),
        "pos_frac": round((yte == 1).mean() if yte.size else 0.0, 4),
    })
    acc, tok_or_proc, model = train_eval_image(
        paths_tr, ytr, paths_te, yte,
        model_id=model_id,
        size=int(settings["image_size"]),
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        lr=float(settings["lr"]),
        device=device,
    )
    _test_metrics = _eval_image_metrics(paths_te, yte, tok_or_proc, model, device)
    print("Baseline test metrics:", {
        "accuracy": round(_test_metrics["accuracy"], 4),
        "balanced_acc": round(_test_metrics["balanced_acc"], 4),
        "ber": round(_test_metrics["ber"], 4),
        "f1": round(_test_metrics["f1"], 4),
        "roc_auc": (round(_test_metrics["roc_auc"], 4) if not np.isnan(_test_metrics["roc_auc"]) else "nan"),
    })

metrics = {
    "accuracy": float(acc),
    "balanced_acc": float(_test_metrics.get("balanced_acc", float("nan"))),
    "ber": float(_test_metrics.get("ber", float("nan"))),
    "f1": float(_test_metrics.get("f1", float("nan"))),
    "roc_auc": float(_test_metrics.get("roc_auc", float("nan"))),
    "seed": int(settings["seed"]),
    "modality": modality,
    "model": model_id,
    "n": int(settings["n"]),
}

processor_or_tok = tok_or_proc
out_dir = Path(settings["out_dir"]) / modality / run_folder
out_dir.mkdir(parents=True, exist_ok=True)

legacy_metrics = out_dir / f"baseline_metrics_seed{int(settings['seed'])}.json"
legacy_metrics.write_text(json.dumps(metrics, indent=2))

named_metrics = out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_metrics.json"
named_metrics.write_text(json.dumps(metrics, indent=2))

model_dir = out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_model"
model_dir.mkdir(parents=True, exist_ok=True)
processor_or_tok.save_pretrained(model_dir)
model.save_pretrained(model_dir)

print(json.dumps({
    "run_dir": str(out_dir),
    "metrics_path": str(legacy_metrics),
    "metrics_named_path": str(named_metrics),
    "model_dir": str(model_dir),
}, indent=2))

split_files = {
    "train": out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_metrics_train.json",
    "val":   out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_metrics_val.json",
    "test":  out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_metrics_test.json",
}
for name, path in split_files.items():
    m = {"train": _train_metrics, "val": _val_metrics, "test": _test_metrics}[name]
    path.write_text(json.dumps(m, indent=2))
