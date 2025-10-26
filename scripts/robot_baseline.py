from __future__ import annotations

import argparse, json, time, random, re, hashlib
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
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

from concept_benchmark.paths import results_dir, pkg_dir
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.synthetic.helper.textgen import create_synthetic_dataset as make_text_ds

settings = {
    "modality": "text",
    "n": 5000,
    "seed": 1337,
    "seed_cv": 1338,
    "seed_deploy": 1339,
    "seed_balance_train": 1901,
    "seed_balance_val": 1902,
    "seed_balance_test": 1903,
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
    "generic_rate": 0.5,
    "generic_tol": 0.02,
    "generic_what": "ears",
    "train_balance_enable": 0,
    "train_target_pos_frac": -1.0,
    "train_target_generic_frac": 0.5,
    "train_balance_within_label": 1,
    "val_balance_enable": 0,
    "val_target_generic_frac": 0.5,
    "val_balance_within_label": 1,
    "test_balance_enable": 0,
    "test_target_generic_frac": 0.5,
    "test_balance_within_label": 1,
    "cv_k": 5,
    "cv_fold": 0,
    "dev_size": 1000,
    "deployment_size": 10000,
    "calibrate": "platt",
    "abstain": "conf",
    "tau": None,
    "tau_target": 0.99,
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
    SAFE_GLOBALS = {"__builtins__": None, "int": int, "str": str, "float": float, "bool": bool, "any": any, "all": all, "np": np, "min": min, "max": max}
    rng = np.random.default_rng(int(seed))
    def _cond_to_score(expr: str) -> str | None:
        m = re.search(r"\bif\s+(?P<cond>.+?)\s+else\b", expr)
        cond = m.group("cond").strip() if m else expr.strip()
        m2 = re.search(r"^(?P<lhs>.+?)(?:\s*(?:>=|<=|>|<)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$", cond, flags=re.IGNORECASE)
        lhs = m2.group("lhs").strip() if m2 else cond
        while lhs.startswith("(") and lhs.endswith(")"):
            lvl = 0; ok = True
            for ch in lhs:
                if ch == "(": lvl += 1
                elif ch == ")":
                    lvl -= 1
                    if lvl < 0: ok = False; break
            if ok and lvl == 0: lhs = lhs[1:-1].strip()
            else: break
        return lhs or None
    score_expr = _cond_to_score(model_expr) if label_model_type == "stochastic" else None
    def eval_one(sr):
        row = sr.to_dict()
        if label_model_type is None or label_model_type == "deterministic":
            return eval(model_expr, SAFE_GLOBALS, {"row": row})
        score = None
        if score_expr:
            try: score = float(eval(score_expr, SAFE_GLOBALS, {"row": row}))
            except Exception: score = None
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

def _load_jsonl(p: Path) -> list[dict]:
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text: raise ValueError(f"HardCorpus file is empty: {p}")
    if text.startswith("["):
        arr = json.loads(text)
        if not isinstance(arr, list): raise ValueError("Top-level JSON is not a list")
        return arr
    items, plain_lines = [], []
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith("```"): continue
        try: items.append(json.loads(s))
        except json.JSONDecodeError: plain_lines.append(s)
    if items: return items
    if plain_lines: return [{"id": f"pt_{i:04d}", "when": {"any": True}, "text": s} for i, s in enumerate(plain_lines, 1)]
    raise ValueError(f"No valid JSON or plain-text lines found in {p}.")

def _nat_from_tokens(row: dict) -> dict:
    head_nat = {"square": "boxy", "round": "dome-like"}[str(row["head_shape"])]
    body_nat = {"square": "sharp-cornered", "round": "barrel-smooth"}[str(row["body_shape"])]
    ears_nat = {"square": "square", "triangle": "pointy"}[str(row["ears_shape"])]
    mouth_nat = {"closed": "shut", "open": "open"}[str(row["mouth_type"])]
    hands_nat_map = {"round_circle": "round mitts","wide_oval":"broad ovals","tall_oval":"long ovals","edgy_square":"square claws","edgy_triangle":"triangular grippers","edgy_trapezoid":"trapezoid claws"}
    feet_nat_map = {
        "flat_4sided": "flat four-sided pads",
        "flat_5sided": "pentagonal pads",
        "flat_lshaped": "L-shaped feet",
        "pointy_3sided": "three-point feet",
        "pointy_4sided": "four-point feet",
        "pointy_6sided": "hex-point feet",
    }
    hands_nat = hands_nat_map.get(str(row["hand_shape"]), hands_nat_map["round_circle"])
    fs = str(row["foot_shape"])
    feet_nat = feet_nat_map.get(fs)
    if feet_nat is None:
        if "trapezoid" in fs:
            feet_nat = "trapezoid feet"
        elif fs.startswith("flat"):
            feet_nat = "flat feet"
        elif fs.startswith("pointy"):
            feet_nat = "pointed feet"
        else:
            feet_nat = fs.replace("_", " ") + " feet"

    ant_nat = "with antennae" if str(row["has_antennae"]).lower() == "true" else "no antennae"
    knees_nat = "has knees" if str(row["has_knees"]).lower() == "true" else "no knees"
    elbows_nat = "has elbows" if str(row["has_elbows"]).lower() == "true" else "no elbows"
    return {"HEAD_NAT": head_nat,"BODY_NAT": body_nat,"EARS_NAT": ears_nat,"MOUTH_NAT": mouth_nat,"HANDS_NAT": hands_nat,"FEET_NAT": feet_nat,"ANT_NAT": ant_nat,"KNEES_NAT": knees_nat,"ELBOWS_NAT": elbows_nat}

def _line_matches(sig: dict, cond: dict) -> bool:
    for k, v in cond.items():
        if k == "any": continue
        if k not in sig: return False
        if isinstance(v, bool):
            if bool(sig[k]) != v: return False
        else:
            if str(sig[k]) != str(v): return False
    return True

def _render_from_corpus(row: dict, corpus: list[dict], seed: int) -> str:
    sig = _signals_from_row(row)
    cand = [it for it in corpus if _line_matches(sig, it.get("when", {}))]
    if not cand: cand = corpus
    key = f'{seed}:{row["head_shape"]}:{row["body_shape"]}:{row["foot_shape"]}:{row["ears_shape"]}:{row["mouth_type"]}:{row["hand_shape"]}:{row["has_antennae"]}:{row["has_knees"]}:{row["has_elbows"]}'
    idx = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(cand)
    txt = str(cand[idx]["text"])
    nat = _nat_from_tokens(row)
    for k, v in nat.items():
        ph = "{" + k + "}"
        if ph in txt: txt = txt.replace(ph, v)
    raw_map = {"head_shape": str(row["head_shape"]),"body_shape": str(row["body_shape"]),"ears_shape": str(row["ears_shape"]),"mouth_type": str(row["mouth_type"]),"hand_shape": str(row["hand_shape"]),"foot_shape": str(row["foot_shape"]),"has_antennae": str(row["has_antennae"]),"has_knees": str(row["has_knees"]),"has_elbows": str(row["has_elbows"])}
    for k, v in raw_map.items():
        ph = "{" + k + "}"
        if ph in txt: txt = txt.replace(ph, v)
    return txt

def _names_from_concepts(concepts: dict) -> list[str]:
    names = []
    for k, vals in concepts.items():
        for v in vals: names.append(f"{k}={v}")
    return names

def _onehot_for_row(row: dict, concepts: dict, names: list[str]) -> np.ndarray:
    J = len(names)
    vec = np.zeros((J,), dtype=np.float32)
    pos = 0
    for k, vals in concepts.items():
        for v in vals:
            if str(row[k]) == str(v): vec[pos] = 1.0
            pos += 1
    return vec

class TextDS(Dataset):
    def __init__(self, X, y, tok, max_length=256):
        self.X = list(map(str, X))
        self.y = np.asarray(y, dtype=int)
        self.tok = tok
        self.max_length = max_length
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        enc = self.tok(self.X[i], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(self.y[i], dtype=torch.long)
        return enc, y

class ImageDS(Dataset):
    def __init__(self, X_paths, y, proc):
        self.X = [str(p) for p in X_paths]
        self.y = np.asarray(y, dtype=int)
        self.proc = proc
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        img = Image.open(self.X[i]).convert("RGB")
        enc = self.proc(images=img, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(self.y[i], dtype=torch.long)
        return enc, y

def _ensure_binary(y):
    u = np.unique(y)
    if u.size < 2: raise ValueError("Training set is single-class")

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

def _fit_platt(X, y, tok, model, device):
    ds = TextDS(X, np.asarray(y, dtype=int), tok)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    z_list = []
    y_list = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            out = model(**xb)
            z = (out.logits[:, 1] - out.logits[:, 0]).detach().cpu().numpy()
            z_list.append(z)
            y_list.append(yb.numpy())
    Z = np.concatenate(z_list).reshape(-1, 1) if z_list else np.zeros((0, 1), dtype=float)
    Y = np.concatenate(y_list).astype(int) if y_list else np.zeros((0,), dtype=int)
    if Y.size == 0 or np.unique(Y).size < 2: return None
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(Z, Y)
    return lr

def _eval_text_metrics(X, y, tok, model, device, calibrator=None, abstain=None, tau=None):
    ds = TextDS(X, np.asarray(y, dtype=int), tok)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            out = model(**xb)
            if calibrator is not None:
                z = (out.logits[:, 1] - out.logits[:, 0]).detach().cpu().numpy()
                p1 = calibrator.predict_proba(z.reshape(-1, 1))[:, 1]
                probs.append(p1)
                preds.append((p1 >= 0.5).astype(int))
            else:
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
    out = {"accuracy": acc, "balanced_acc": ba, "ber": float(1.0 - ba), "f1": f1, "roc_auc": roc}
    if abstain == "conf" and (tau is not None):
        conf = np.where(y_pred == 1, proba1, 1.0 - proba1)
        m = conf >= float(tau)
        cov = float(m.mean()) if m.size else 0.0
        if m.any(): sel_acc = float((y_pred[m] == y_true[m]).mean())
        else: sel_acc = float("nan")
        out["selective_accuracy"] = sel_acc
        out["coverage"] = cov
    return out

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
p.add_argument("--seed-cv", dest="seed_cv", type=int)
p.add_argument("--seed-deploy", dest="seed_deploy", type=int)
p.add_argument("--seed-balance-train", dest="seed_balance_train", type=int)
p.add_argument("--seed-balance-val", dest="seed_balance_val", type=int)
p.add_argument("--seed-balance-test", dest="seed_balance_test", type=int)
p.add_argument("--cv-k", dest="cv_k", type=int)
p.add_argument("--cv-fold", dest="cv_fold", type=int)
p.add_argument("--dev-size", dest="dev_size", type=int)
p.add_argument("--deployment-size", dest="deployment_size", type=int)
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
p.add_argument("--calibrate", choices=["none","platt"])
p.add_argument("--abstain", choices=["none","conf"])
p.add_argument("--tau", type=float)
p.add_argument("--tau-target", dest="tau_target", type=float)
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
p.add_argument("--redact-masked-only", dest="redact_masked_only", type=int)
p.add_argument("--generic-rate", dest="generic_rate", type=float)
p.add_argument("--generic-tol", dest="generic_tol", type=float)
p.add_argument("--generic-what", dest="generic_what", choices=["ears","foot","mouth","footmouth"])
p.add_argument("--train-balance-enable", dest="train_balance_enable", type=int)
p.add_argument("--train-target-pos-frac", dest="train_target_pos_frac", type=float)
p.add_argument("--train-target-generic-frac", dest="train_target_generic_frac", type=float)
p.add_argument("--train-balance-within-label", dest="train_balance_within_label", type=int)
p.add_argument("--val-balance-enable", dest="val_balance_enable", type=int)
p.add_argument("--val-target-generic-frac", dest="val_target_generic_frac", type=float)
p.add_argument("--val-balance-within-label", dest="val_balance_within_label", type=int)
p.add_argument("--test-balance-enable", dest="test_balance_enable", type=int)
p.add_argument("--test-target-generic-frac", dest="test_target_generic_frac", type=float)
p.add_argument("--test-balance-within-label", dest="test_balance_within_label", type=int)
p.add_argument("--image-meta", dest="image_meta", type=str)
p.add_argument("--image-meta-catalog", dest="image_meta_catalog", choices=["auto","base","spurious"])
p.add_argument("--skip-generic-leak-check", dest="skip_generic_leak_check", action="store_true")
p.add_argument("--debug-dump", dest="debug_dump", type=int)

args, _ = p.parse_known_args()
for k, v in vars(args).items():
    if v is not None: settings[k] = v if k != "draw" else bool(v)

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
                       row_variants: list[int] | None = None,
                       generic_path: Path | None = None,
                       generic_rate: float = 0.5) -> ConceptDatasetSample:
    corpus_spec = _load_jsonl(corpus_path)
    corpus_gen = _load_jsonl(generic_path) if (generic_path is not None and Path(generic_path).is_file()) else []
    names = _names_from_concepts(concepts)
    classes = [0, 1]
    X, C, y, row_index, ears_generic = [], [], [], [], []
    for i, sr in catalog_df.iterrows():
        row = {k: sr[k] for k in concepts.keys()}
        row["label"] = sr["label"]
        _vpr_i = int(row_variants[i]) if (row_variants is not None and i < len(row_variants)) else int(variants_per_row)
        for v in range(max(1, _vpr_i)):
            _gen_target = str(settings.get("generic_what","ears")).lower()
            key = f"{seed}:{i}:{v}:{_gen_target}_generic"
            h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            use_gen = (len(corpus_gen) > 0) and ((h % 1000000) < int(max(0.0, min(1.0, float(generic_rate))) * 1000000))
            corpus = corpus_gen if use_gen else corpus_spec
            text = _render_from_corpus(row, corpus, seed + v)
            X.append(text)
            C.append(_onehot_for_row(row, concepts, names))
            y.append(1 if str(row["label"]) == "glorp" else 0)
            row_index.append(i)
            ears_generic.append(bool(use_gen))
    ds = ConceptDatasetSample(X=X, C=np.asarray(C, dtype=np.float32), y=np.asarray(y, dtype=int), meta={"concepts": tuple(names), "classes": tuple(classes), "data_type": "text"})
    setattr(ds, "_full", type("Full", (), {"meta": {"row_index": np.asarray(row_index, dtype=int)}}))
    _gen_target = str(settings.get("generic_what", "ears")).lower()
    _mask = np.asarray(ears_generic, dtype=bool)
    setattr(ds, f"{_gen_target}_generic_mask", _mask)
    if _gen_target == "footmouth":
        setattr(ds, "foot_generic_mask", _mask)
        setattr(ds, "mouth_generic_mask", _mask)
    ds.ears_generic_mask = _mask
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

    if str(settings.get("image_meta","")).strip():
        _im = json.loads(Path(str(settings["image_meta"])).read_text())
        _cat_base = _im.get("catalog_csv", "")
        _cat_spu  = _im.get("catalog_csv_spurious", "")
        _kind = str(settings.get("image_meta_catalog","auto")).lower()

        def _exists(p): return isinstance(p,str) and p and Path(p).is_file()
        if _kind == "base":
            _choice = _cat_base
        elif _kind == "spurious":
            _choice = _cat_spu
        else:
            _choice = _cat_spu if _exists(_cat_spu) else _cat_base
        if not _exists(_choice):
            raise SystemExit(f"--image-meta provided but selected catalog missing or unreadable: {_choice}")

        catalog_df = pd.read_csv(_choice)

        def _normalize_catalog_schema(df: pd.DataFrame, seed: int) -> pd.DataFrame:
            df = df.copy()
            if "hand_shape_subtype" in df.columns:
                df["hand_shape"] = df["hand_shape_subtype"].astype(str)
            if "foot_shape_subtype" in df.columns:
                df["foot_shape"] = df["foot_shape_subtype"].astype(str)

            def _canon_hand(v, i):
                t = str(v)
                m = {
                    "round": "round_circle",
                    "circle": "round_circle",
                    "oval": "wide_oval",
                    "oval2": "tall_oval",
                    "wide": "wide_oval",
                    "tall": "tall_oval",
                    "tall_oval": "tall_oval",
                    "square": "edgy_square",
                    "triangle": "edgy_triangle",
                    "trapezoid": "edgy_trapezoid",
                    "edgy_square": "edgy_square",
                    "edgy_triangle": "edgy_triangle",
                    "edgy_trapezoid": "edgy_trapezoid",
                }
                return m.get(t, t)
            if "hand_shape" in df.columns:
                df["hand_shape"] = [_canon_hand(v, i) for i, v in enumerate(df["hand_shape"])]

            def _canon_foot(v, i):
                s = str(v)
                if s in {"flat", "flat_generic"}:
                    return ["flat_4sided","flat_5sided","flat_lshaped"][i % 3]
                if s in {"pointy", "pointy_generic"}:
                    return ["pointy_3sided","pointy_4sided","pointy_6sided"][i % 3]
                return s
            if "foot_shape" in df.columns:
                df["foot_shape"] = [_canon_foot(v, i + int(settings.get("seed",0))) for i, v in enumerate(df["foot_shape"])]

            for b in ["has_antennae","has_knees","has_elbows"]:
                if b in df.columns:
                    df[b] = df[b].astype(str).str.lower()
            return df

        catalog_df = _normalize_catalog_schema(catalog_df, seed=int(settings.get("seed",0)))
    else:
        catalog_df = pd.DataFrame([dict(zip(cols, vals)) for vals in product(*[concepts[c] for c in cols])], columns=cols)

    default_expr = "'glorp' if (min(int(str(row['has_antennae']).lower()=='true'), int(row['body_shape']=='square')) >= 1) else 'drent'"
    expr = str(settings.get("label_model_expr", "") or "").strip()
    if ("label" in catalog_df.columns) and not expr:
        catalog_df["label"] = catalog_df["label"].astype(str)
    else:
        label_expr = expr or default_expr
        catalog_df["label"] = compute_label(
            catalog_df,
            label_expr,
            label_model_type=settings.get("label_model_type", "deterministic"),
            alpha=float(settings.get("label_model_alpha", 10.0)),
            bias=float(settings.get("label_model_bias", -0.2)),
            seed=int(settings.get("seed", 0)),
        )
    _lbl = catalog_df["label"].astype(str)
    print("Label distribution (catalog_df):", {"glorp": int((_lbl == "glorp").sum()), "drent": int((_lbl == "drent").sum()), "total": int(len(_lbl)), "pos_frac": round((_lbl == "glorp").mean(), 4)})
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
        _gen_target = str(settings.get("generic_what", "ears")).lower()
        _suffix = {"ears":"EarsGeneric","foot":"FootGeneric","mouth":"MouthGeneric","footmouth":"FootMouthGeneric"}[_gen_target]
        _cands = [
            tpl_path.with_name(f"{tpl_path.stem}_{_suffix}.jsonl"),
            tpl_path.with_name(f"HardCorpus_{_suffix}.jsonl"),
        ]
        gen_jsonl = next((p for p in _cands if p.is_file()), None) if is_jsonl else None
        ds = build_text_ds_hard(
            catalog_df=catalog_df,
            concepts=concepts,
            corpus_path=tpl_path,
            variants_per_row=_base_vpr,
            seed=int(settings["seed"]),
            row_variants=_row_variants,
            generic_path=gen_jsonl,
            generic_rate=float(settings.get("generic_rate", 0.5)),
        )
    else:
        with open(tpl_path, "r", encoding="utf-8-sig") as f:
            templates = [ln.strip() for ln in f if ln.strip()]
        ds = make_text_ds(source=catalog_df, templates=templates, variants_per_row=max(1, settings["n"] // max(1, len(catalog_df))), include_color=False, rng_seed=int(settings["seed"]), concept_cols=cols, label_col="label", label_map={"drent": 0, "glorp": 1}, text_mode="semi", llm_provider="gemini", llm_model="gemini-1.5-flash", llm_user_prompt="Describe the robot based only on attributes.")

    _split_applied = False
    if str(settings.get("image_meta","")).strip():
        _im = json.loads(Path(str(settings["image_meta"])).read_text())
        _dfi = _im.get("df_indices", {})
        if _dfi:
            row_index_full = getattr(getattr(ds, "_full", ds), "meta", {}).get("row_index", np.arange(len(ds.X)))
            idx_map = {int(r): i for i, r in enumerate(np.asarray(row_index_full, dtype=int))}

            def _subset_by_ids(ids):
                sel = [idx_map[int(x)] for x in ids if int(x) in idx_map]
                if not sel: return None
                sel = np.asarray(sel, dtype=int)
                X = [ds.X[i] for i in sel]
                C = ds.C[sel]
                y = ds.y[sel]
                sub = ConceptDatasetSample(
                    X=X, C=C, y=y,
                    meta={"concepts": ds.concepts, "classes": ds.classes, "data_type": "text",
                          "df_indices": [int(row_index_full[i]) for i in sel]}
                )
                # copy any available generic masks
                for attr, val in ds.__dict__.items():
                    if attr.endswith("_generic_mask"):
                        setattr(sub, attr, np.asarray(val)[sel])
                return sub

            tr = _subset_by_ids(_dfi.get("train", []))
            va = _subset_by_ids(_dfi.get("valid", _dfi.get("val", [])))
            te = _subset_by_ids(_dfi.get("test", []))
            if tr is not None and va is not None and te is not None:
                ds.training, ds.validation, ds.test = tr, va, te
                ds.deployment = ds.test
                _split_applied = True

    if not _split_applied:
        K = int(settings.get("cv_k", 5))
        seed_cv = int(settings.get("seed_cv", int(settings.get("seed", 0)) + 1))
        val_fold = int(settings.get("cv_fold", 0)) or ((seed_cv % K) + 1)
        dev_size = int(settings.get("dev_size", 1000))
        rng_cv = np.random.default_rng(seed_cv)
        n_all = len(ds.X)
        idx_all = np.arange(n_all)
        if dev_size > n_all: dev_size = n_all
        idx_dev = rng_cv.choice(idx_all, size=dev_size, replace=False)
        y_dev = np.asarray(ds.y, dtype=int)[idx_dev]
        folds_dev = np.zeros(dev_size, dtype=int)
        idx0 = np.where(y_dev == 0)[0]
        idx1 = np.where(y_dev == 1)[0]
        rng_cv.shuffle(idx0); rng_cv.shuffle(idx1)
        for f in range(1, K + 1):
            s0 = (f - 1) * len(idx0) // K; e0 = f * len(idx0) // K
            s1 = (f - 1) * len(idx1) // K; e1 = f * len(idx1) // K
            folds_dev[idx0[s0:e0]] = f
            folds_dev[idx1[s1:e1]] = f
        mask_val = np.zeros(n_all, dtype=bool)
        mask_tr = np.zeros(n_all, dtype=bool)
        mask_val[idx_dev[folds_dev == val_fold]] = True
        drop_fold = 1 + (val_fold % K)
        keep = (folds_dev != val_fold) & (folds_dev != drop_fold)
        mask_tr[idx_dev[keep]] = True
        tr = np.where(mask_tr)[0]
        va = np.where(mask_val)[0]

        dep_n = int(settings.get("deployment_size", 10000))
        seed_dep = int(settings.get("seed_deploy", int(settings["seed"]) + 1234))
        rng_dep = np.random.default_rng(seed_dep)
        pool = np.setdiff1d(idx_all, np.concatenate([tr, va])) if (tr.size + va.size) < n_all else idx_all
        if dep_n <= pool.size:
            idx_dep = rng_dep.choice(pool, size=dep_n, replace=False)
        else:
            need = dep_n - pool.size
            extra = rng_dep.choice(pool, size=need, replace=True)
            idx_dep = np.concatenate([pool, extra])

        def _subset(ds_obj, take):
            X = [ds_obj.X[i] for i in take]
            C = ds_obj.C[take]
            y = ds_obj.y[take]
            row_index_full = getattr(getattr(ds_obj, "_full", ds_obj), "meta", {}).get("row_index",
                                                                                       np.arange(len(ds_obj.X)))
            sub = ConceptDatasetSample(
                X=X, C=C, y=y,
                meta={"concepts": ds_obj.concepts, "classes": ds_obj.classes, "data_type": "text",
                      "df_indices": [int(row_index_full[i]) for i in take]}
            )
            for attr, val in ds_obj.__dict__.items():
                if attr.endswith("_generic_mask"):
                    setattr(sub, attr, np.asarray(val)[take])
            return sub

        ds.training = _subset(ds, tr)
        ds.validation = _subset(ds, va)
        ds.deployment = _subset(ds, idx_dep)
        ds.test = ds.deployment

    # --- redaction (masked-aware, multi-concept) ---
    rc = str(settings.get("redact_concepts", "") or "").strip().lower()
    rs = str(settings.get("redact_splits", "") or "").strip().lower()
    rm_only = bool(int(settings.get("redact_masked_only", 0) or 0))

    def _compose_pattern(concepts_to_redact: set[str]) -> re.Pattern:
        pats = []
        if "has_antennae" in concepts_to_redact:
            pats.append(r"(?i)\b(?:with\s+)?antenna(?:e|s)?\b|\bno\s+antennae\b")
        if "has_elbows" in concepts_to_redact:
            pats.append(r"(?i)\belbow(?:s)?\b")
        if "has_knees" in concepts_to_redact:
            pats.append(r"(?i)\bknee(?:s)?\b")
        if "hand_shape" in concepts_to_redact:
            pats.append(r"(?i)\bhand(?:s)?\b|\bmitts?\b|\bgrippers?\b|\bclaws?\b")
        if "head_shape" in concepts_to_redact:
            pats.append(r"(?i)\bhead(?:ed)?\b|\bboxy\b|\bdome-?like\b|\bround-?headed\b|\bsquare-?headed\b")
        if "body_shape" in concepts_to_redact:
            pats.append(r"(?i)\bbody\b|\bbarrel-?smooth\b|\bsharp-?cornered\b")
        if not pats:
            pats.append(r"(?!x)x")
        return re.compile("|".join(pats))

    def _redact_text_list(lst: list[str], pat: re.Pattern) -> list[str]:
        out = []
        for s in lst:
            z = re.sub(pat, " ", str(s))
            z = re.sub(r"\s{2,}", " ", z)
            z = re.sub(r"\s+([,.;:!?])", r"\1", z).strip()
            out.append(z)
        return out

    def _maybe_masked_indices(part, gen_target: str) -> np.ndarray:
        gm = getattr(part, f"{gen_target}_generic_mask", None)
        if gm is None and gen_target == "footmouth":
            gf = getattr(part, "foot_generic_mask", None)
            gm2 = getattr(part, "mouth_generic_mask", None)
            if gf is not None and gm2 is not None:
                gm = (np.asarray(gf) | np.asarray(gm2))
        if gm is None:
            return np.zeros(len(part.X), dtype=bool)
        return np.asarray(gm, dtype=bool)

    if rc and rs:
        concepts_to_redact = {t.strip() for t in rc.split(",") if t.strip()}
        targets = {t.strip() for t in rs.split(",") if t.strip()}
        pat = _compose_pattern(concepts_to_redact)
        _gen_target = str(settings.get("generic_what", "ears")).lower()

        def _apply(part):
            if not rm_only:
                part = ConceptDatasetSample(X=_redact_text_list(list(part.X), pat), C=part.C, y=part.y, meta=part.meta)
                return part
            m = _maybe_masked_indices(part, _gen_target)
            if not m.any():
                return part
            X = list(part.X)
            idx = np.where(m)[0].tolist()
            sub = [X[i] for i in idx]
            red = _redact_text_list(sub, pat)
            for j, i in enumerate(idx):
                X[i] = red[j]
            part = ConceptDatasetSample(X=X, C=part.C, y=part.y, meta=part.meta)
            return part

        apply_all = "masked" in targets and not ({"train","val","test"} & targets)
        chosen = {"train","val","test"} if apply_all else ({"train","val","test"} & targets)
        if "train" in chosen: ds.training = _apply(ds.training)
        if "val"   in chosen: ds.validation = _apply(ds.validation)
        if "test"  in chosen: ds.test = _apply(ds.test)
    # --- end redaction ---

    _gen_target = str(settings.get("generic_what", "ears")).lower()
    _key_near = f"generic_near_{_gen_target}_shape"

    leak = {}
    dist = {}
    for name, part in [("train", ds.training), ("val", ds.validation), ("test", ds.test)]:
        gm = getattr(part, f"{_gen_target}_generic_mask", None)
        if gm is None and _gen_target == "footmouth":
            gf = getattr(part, "foot_generic_mask", None)
            gm2 = getattr(part, "mouth_generic_mask", None)
            if gf is not None and gm2 is not None:
                gm = (np.asarray(gf) | np.asarray(gm2)).astype(bool)
            else:
                gm = gf if gf is not None else gm2
        if gm is None:
            gm = getattr(part, "ears_generic_mask", None)

        yv = np.asarray(part.y, dtype=int)
        if gm is None:
            leak[name] = {_key_near: "na"}
            dist[name] = {"overall": "na", "y1": "na", "y0": "na"}
            continue

        def _generic_leak(txt: str) -> bool:
            t = str(txt).lower()
            sents = re.split(r"[.!?;:]\s+", t)

            def _ears():
                part = re.compile(r"\bears?\b")
                shape = re.compile(r"\b(square|boxy|box|angular|cornered|right-angled|rectilinear|90-degree|triangle|triangular|tri-corner|three-angled|three-point|pointy|pointed|tapered|wedge|spearhead|spear-tip)\b")
                return any(part.search(s) and shape.search(s) for s in sents)

            def _foot():
                part = re.compile(r"\b(?:foot|feet)\b")
                shape = re.compile(r"\b(square|boxy|box|angular|cornered|right-angled|rectilinear|90-degree|triangle|triangular|tri-corner|three-angled|three-point|pointy|pointed|tapered|wedge|spearhead|spear-tip|flat|round|circle|circular|trapezoid|trapezoidal)\b")
                return any(part.search(s) and shape.search(s) for s in sents)

            def _mouth():
                part = re.compile(r"\bmouth\b")
                state = re.compile(r"\b(open|closed|shut|wide|narrow|smile|smiling|grin|frown|grimace)\b")
                return any(part.search(s) and state.search(s) for s in sents)

            if _gen_target == "ears":
                return _ears()
            if _gen_target == "foot":
                return _foot()
            if _gen_target == "mouth":
                return _mouth()
            if _gen_target == "footmouth":
                return _foot() or _mouth()
            return False

        leak[name] = {_key_near: int(sum(_generic_leak(t) for t, g in zip(part.X, gm) if g))}
        overall = float(gm.mean()) if gm.size else float("nan")
        y1 = float(gm[yv == 1].mean()) if (yv == 1).any() else float("nan")
        y0 = float(gm[yv == 0].mean()) if (yv == 0).any() else float("nan")
        dist[name] = {"overall": overall, "y1": y1, "y0": y0}

    t_train = float(settings.get("train_target_generic_frac", settings.get("generic_rate", 0.5))) if int(settings.get("train_balance_enable", 0)) == 1 else float(settings.get("generic_rate", 0.5))
    t_val = float(settings.get("generic_rate", 0.5))
    t_test = float(settings.get("generic_rate", 0.5))
    tol = float(settings.get("generic_tol", 0.02))
    targets = {"train": t_train, "val": t_val, "test": t_test}

    print(json.dumps({
        "split_sizes": {"train": int(ds.training.n), "val": int(ds.validation.n), "test": int(ds.test.n)},
        f"{_gen_target}_leak_counts_generic": leak,
        f"{_gen_target}_generic_rates": dist,
        "targets": {"train": t_train, "val": t_val, "test": t_test, "tol": tol},
        "split_files": {},
        "run_dir": str(out_dir)
    }, indent=2))

    # --- debug dump of texts by mask ---
    if int(settings.get("debug_dump", 0)) == 1:
        dbg_dir = Path(out_dir) / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        def _get_mask(part):
            gm = getattr(part, f"{_gen_target}_generic_mask", None)
            if gm is None and _gen_target == "footmouth":
                gf = getattr(part, "foot_generic_mask", None)
                gm2 = getattr(part, "mouth_generic_mask", None)
                gm = (np.asarray(gf) | np.asarray(gm2)).astype(bool) if (gf is not None and gm2 is not None) else (
                    gf if gf is not None else gm2)
            return np.asarray(gm, dtype=bool) if gm is not None else np.zeros(len(part.X), dtype=bool)

        def _dump_split(name, part):
            gm = _get_mask(part)
            rid_list = (part.meta or {}).get("df_indices", list(range(len(part.X))))
            for flag, tag in [(True, "masked"), (False, "unmasked")]:
                idx = [i for i, g in enumerate(gm) if bool(g) == flag]
                sample = idx[:200]
                with (dbg_dir / f"{name}_{tag}.txt").open("w", encoding="utf-8") as f:
                    for j in sample:
                        f.write(f"{rid_list[j]}\t{int(part.y[j])}\t{part.X[j]}\n")

        for nm, prt in [("train", ds.training), ("val", ds.validation), ("test", ds.test)]:
            _dump_split(nm, prt)
    # --- end debug dump ---

    if not bool(settings.get("skip_generic_leak_check", False)):
        if any(v.get(_key_near, 0) not in ("na", 0) for v in leak.values()):
            raise SystemExit(3)

    for name, vals in dist.items():
        if vals["overall"] != "na" and np.isfinite(vals["overall"]):
            if abs(vals["overall"] - targets[name]) > tol: raise SystemExit(4)
        for k in ("y1","y0"):
            if vals[k] != "na" and np.isfinite(vals[k]):
                if abs(vals[k] - targets[name]) > tol: raise SystemExit(4)

    yt = np.asarray(ds.training.y, dtype=int)
    yv = np.asarray(ds.validation.y, dtype=int)
    yte = np.asarray(ds.test.y, dtype=int)
    print("Split sizes →", {"train": int(ds.training.n), "val": int(ds.validation.n), "test": int(ds.test.n)})
    print("Label distribution (train):", {"glorp": int((yt == 1).sum()), "drent": int((yt == 0).sum()), "total": int(yt.size), "pos_frac": round((yt == 1).mean() if yt.size else 0.0, 4)})
    print("Label distribution (val):", {"glorp": int((yv == 1).sum()), "drent": int((yv == 0).sum()), "total": int(yv.size), "pos_frac": round((yv == 1).mean() if yv.size else 0.0, 4)})
    print("Label distribution (test):", {"glorp": int((yte == 1).sum()), "drent": int((yte == 0).sum()), "total": int(yte.size), "pos_frac": round((yte == 1).mean() if yte.size else 0.0, 4)})

    Xtr = ds.training.X
    ytr = ds.training.y.astype(int)
    Xva = ds.validation.X
    yva = ds.validation.y.astype(int)
    Xte = ds.test.X
    yte = ds.test.y.astype(int)

    acc, tok_or_proc, model = train_eval_text(Xtr, ytr, Xte, yte, model_id=model_id, epochs=int(settings["epochs"]), batch_size=int(settings["batch_size"]), lr=float(settings["lr"]), device=device)
    calibrator = None
    if str(settings.get("calibrate", "none")).lower() == "platt":
        calibrator = _fit_platt(Xva, yva, tok_or_proc, model, device)

    abstain_mode = str(settings.get("abstain", "none")).lower()
    target_sel = float(settings.get("tau_target", 0.99))
    tau_val = settings.get("tau", None)
    if abstain_mode == "conf" and (tau_val is None or str(tau_val).lower() == "none"):
        grid = np.linspace(0.5, 0.999, 201)
        best_tau = None
        best_cov = 0.0
        for t in grid:
            m = _eval_text_metrics(Xva, yva, tok_or_proc, model, device, calibrator=calibrator, abstain="conf", tau=float(t))
            sel_acc = m.get("selective_accuracy", float("nan"))
            cov = m.get("coverage", 0.0)
            if not np.isnan(sel_acc) and sel_acc >= target_sel:
                if best_tau is None or t < best_tau or (t == best_tau and cov > best_cov):
                    best_tau = float(t); best_cov = float(cov)
        tau_val = best_tau if best_tau is not None else 1.0
        settings["tau"] = float(tau_val)

    _train_metrics = _eval_text_metrics(Xtr, ytr, tok_or_proc, model, device, calibrator=calibrator, abstain=abstain_mode, tau=tau_val)
    _val_metrics = _eval_text_metrics(Xva, yva, tok_or_proc, model, device, calibrator=calibrator, abstain=abstain_mode, tau=tau_val)
    _test_metrics = _eval_text_metrics(Xte, yte, tok_or_proc, model, device, calibrator=calibrator, abstain=abstain_mode, tau=tau_val)

    if abstain_mode == "conf" and tau_val is not None:
        for _m in [_train_metrics, _val_metrics, _test_metrics]:
            _m["tau"] = float(tau_val)
            _m["tau_target"] = float(target_sel)

    _gen_target = str(settings.get("generic_what", "ears")).lower()
    gm = getattr(ds.test, f"{_gen_target}_generic_mask", None)
    if gm is None and _gen_target == "footmouth":
        gf = getattr(ds.test, "foot_generic_mask", None)
        gm2 = getattr(ds.test, "mouth_generic_mask", None)
        gm = (np.asarray(gf) | np.asarray(gm2)) if (gf is not None and gm2 is not None) else None

    masked_metrics = {}
    unmasked_metrics = {}
    if isinstance(gm, np.ndarray) and gm.size == len(Xte):
        m = gm.astype(bool)
        if m.any():
            masked_metrics = _eval_text_metrics([Xte[i] for i in range(len(Xte)) if m[i]],
                                                [int(yte[i]) for i in range(len(yte)) if m[i]],
                                                tok_or_proc, model, device,
                                                calibrator=calibrator, abstain=abstain_mode, tau=tau_val)
        if (~m).any():
            unmasked_metrics = _eval_text_metrics([Xte[i] for i in range(len(Xte)) if not m[i]],
                                                  [int(yte[i]) for i in range(len(yte)) if not m[i]],
                                                  tok_or_proc, model, device,
                                                  calibrator=calibrator, abstain=abstain_mode, tau=tau_val)

    print("Baseline test metrics:",
          {"accuracy": round(_test_metrics["accuracy"], 4),
           "balanced_acc": round(_test_metrics["balanced_acc"], 4),
           "ber": round(_test_metrics["ber"], 4),
           "f1": round(_test_metrics["f1"], 4),
           "roc_auc": (round(_test_metrics["roc_auc"], 4) if not np.isnan(_test_metrics["roc_auc"]) else "nan")},
          "masked", {"accuracy": round(masked_metrics.get("accuracy", float("nan")), 4)},
          "unmasked", {"accuracy": round(unmasked_metrics.get("accuracy", float("nan")), 4)})

else:
    imgs_dir = pkg_dir / "synthetic" / "helper" / "static" / "robot_images_small"
    meta = pd.read_csv(imgs_dir / "meta.csv")
    meta = meta.sample(frac=1.0, random_state=int(settings["seed"])).reset_index(drop=True)
    label_expr = settings["label_model_expr"] or "'glorp' if (min(int(row['ears_shape']=='square'), int(row['body_shape']=='square')) >= 1) else 'drent'"
    meta["label"] = compute_label(meta, label_expr, label_model_type=settings.get("label_model_type", "deterministic"), alpha=float(settings.get("label_model_alpha", 10.0)), bias=float(settings.get("label_model_bias", -0.2)), seed=int(settings.get("seed", 0)))
    _lbl_img = meta["label"].astype(str)
    print("Label distribution (images, full):", {"glorp": int((_lbl_img == "glorp").sum()), "drent": int((_lbl_img == "drent").sum()), "total": int(len(_lbl_img)), "pos_frac": round((_lbl_img == "glorp").mean(), 4)})
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
    print("Label distribution (train):", {"glorp": int((ytr == 1).sum()), "drent": int((ytr == 0).sum()), "total": int(ytr.size), "pos_frac": round((ytr == 1).mean() if ytr.size else 0.0, 4)})
    print("Label distribution (val):", {"glorp": int((yva == 1).sum()), "drent": int((yva == 0).sum()), "total": int(yva.size), "pos_frac": round((yva == 1).mean() if yva.size else 0.0, 4)})
    print("Label distribution (test):", {"glorp": int((yte == 1).sum()), "drent": int((yte == 0).sum()), "total": int(yte.size), "pos_frac": round((yte == 1).mean() if yte.size else 0.0, 4)})
    acc, tok_or_proc, model = train_eval_image(paths_tr, ytr, paths_te, yte, model_id=model_id, size=int(settings["image_size"]), epochs=int(settings["epochs"]), batch_size=int(settings["batch_size"]), lr=float(settings["lr"]), device=device)
    _test_metrics = _eval_image_metrics(paths_te, yte, tok_or_proc, model, device)
    print("Baseline test metrics:",
          {"accuracy": round(_test_metrics["accuracy"], 4), "balanced_acc": round(_test_metrics["balanced_acc"], 4),
           "ber": round(_test_metrics["ber"], 4), "f1": round(_test_metrics["f1"], 4),
           "roc_auc": (round(_test_metrics["roc_auc"], 4) if not np.isnan(_test_metrics["roc_auc"]) else "nan")})

    # Image baseline does not track generic masks; skip masked/unmasked breakdown here.

metrics = {
    "accuracy": float(acc),
    "balanced_acc": float(_test_metrics.get("balanced_acc", float("nan"))),
    "ber": float(_test_metrics.get("ber", float("nan"))),
    "f1": float(_test_metrics.get("f1", float("nan"))),
    "roc_auc": float(_test_metrics.get("roc_auc", float("nan"))),
    "seed": int(settings["seed"]),
    "seed_cv": int(settings["seed_cv"]),
    "seed_deploy": int(settings["seed_deploy"]),
    "modality": settings["modality"],
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

print(json.dumps({"run_dir": str(out_dir), "metrics_path": str(legacy_metrics), "metrics_named_path": str(named_metrics), "model_dir": str(model_dir)}, indent=2))

split_files = {
    "train": out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_metrics_train.json",
    "val":   out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_metrics_val.json",
    "test":  out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_seed{int(settings['seed'])}_metrics_test.json",
}
metrics_map = {"train": _train_metrics, "val": _val_metrics, "test": _test_metrics}
for name, path in split_files.items():
    path.write_text(json.dumps(metrics_map[name], indent=2))
