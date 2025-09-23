# scripts/robot_baseline.py  (hard, LLM-free caption path supported)

import argparse, json, time, random
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

from concept_benchmark.paths import results_dir, pkg_dir
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.synthetic.helper.textgen import create_synthetic_dataset as make_text_ds

# -------------------- settings --------------------
settings = {
    "modality": "text",
    "n": 5000,
    "seed": 1337,
    "out_dir": str(results_dir / "robot_baseline"),
    "label_model_expr": "",
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
    "samples_per_instance": 3,   # used as variants_per_row for hard JSONL path
    "draw": 0,
    "run_name": "",
    "templates_file": "",
    "template_difficulty": "hard",
}
# --------------------------------------------------

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def compute_label(df: pd.DataFrame, model_expr: str) -> pd.Series:
    SAFE_GLOBALS = {"__builtins__": None, "int": int, "str": str, "float": float, "bool": bool, "any": any, "all": all}
    def eval_one(sr):
        row = sr.to_dict()
        return eval(model_expr, SAFE_GLOBALS, {"row": row})
    return df.apply(eval_one, axis=1)

def enumerate_concepts(concepts, shuffle=True, seed=0):
    cols = list(concepts.keys())
    grids = [concepts[c] for c in cols]
    combos = list(product(*grids))
    df = pd.DataFrame(combos, columns=cols)
    if shuffle:
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df

# -------------------- Torch datasets --------------------
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
# --------------------------------------------------------

# -------------------- Hard JSONL corpus helpers --------------------
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
        "round_circle":"round mitts","wide_oval":"broad ovals","tall_oval":"long ovals",
        "edgy_square":"square claws","edgy_triangle":"triangular grippers","edgy_trapezoid":"trapezoid claws",
    }
    feet_nat_map = {
        "flat_4sided":"flat four-sided pads","flat_5sided":"pentagonal pads","flat_lshaped":"L-shaped feet",
        "pointy_3sided":"three-point feet","pointy_4sided":"four-point feet","pointy_6sided":"hex-point feet",
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
    # Accept UTF-8 BOM and either JSONL, a JSON array file, or plain-text corpus
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
        return [{"id": f"pt_{i:04d}", "when": {"any": True}, "text": s}
                for i, s in enumerate(plain_lines, 1)]
    raise ValueError(f"No valid JSON or plain-text lines found in {p}.")

def _nat_from_tokens(row: dict) -> dict:
    head_nat = {"square": "boxy", "round": "dome-like"}[str(row["head_shape"])]
    body_nat = {"square": "sharp-cornered", "round": "barrel-smooth"}[str(row["body_shape"])]
    ears_nat = {"square": "square", "triangle": "pointy"}[str(row["ears_shape"])]
    mouth_nat = {"closed": "shut", "open": "open"}[str(row["mouth_type"])]
    hands_nat_map = {
        "round_circle":"round mitts","wide_oval":"broad ovals","tall_oval":"long ovals",
        "edgy_square":"square claws","edgy_triangle":"triangular grippers","edgy_trapezoid":"trapezoid claws",
    }
    feet_nat_map = {
        "flat_4sided":"flat four-sided pads","flat_5sided":"pentagonal pads","flat_lshaped":"L-shaped feet",
        "pointy_3sided":"three-point feet","pointy_4sided":"four-point feet","pointy_6sided":"hex-point feet",
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

def _render_from_corpus(row: dict, corpus: list[dict], seed: int) -> str:
    # If corpus entries have conditions, you can add a filter here; for now just deterministic pick
    key = f'{seed}:{row["head_shape"]}:{row["body_shape"]}:{row["foot_shape"]}:{row["ears_shape"]}:{row["mouth_type"]}:{row["hand_shape"]}:{row["has_antennae"]}:{row["has_knees"]}:{row["has_elbows"]}'
    idx = abs(hash(key)) % len(corpus)
    txt = str(corpus[idx].get("text", ""))

    # 1) naturalized placeholders
    nat = _nat_from_tokens(row)
    for k, v in nat.items():
        ph = "{" + k + "}"
        if ph in txt:
            txt = txt.replace(ph, v)

    # 2) raw placeholders (for plain-text lines)
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

def _names_from_params(params) -> list[str]:
    names = []
    for k, vals in params["concepts"].items():
        for v in vals:
            names.append(f"{k}={v}")
    return names

def _onehot_for_row(row: dict, params, names: list[str]) -> np.ndarray:
    J = len(names); vec = np.zeros((J,), dtype=np.float32)
    pos = 0
    for k, vals in params["concepts"].items():
        for v in vals:
            if str(row[k]) == str(v):
                vec[pos] = 1.0
            pos += 1
    return vec

def _build_ds_from_corpus(catalog_df: pd.DataFrame, params, corpus_path: Path, variants_per_row: int, seed: int):
    corpus = _load_jsonl(corpus_path)
    names = _names_from_params(params)
    classes = [0, 1]
    X, C, y, row_index = [], [], [], []
    for i, sr in catalog_df.iterrows():
        row = {k: sr[k] for k in params["concepts"].keys()}
        for v in range(int(variants_per_row)):
            text = _render_from_corpus(row, corpus, seed + v)
            X.append(text)
            C.append(_onehot_for_row(row, params, names))
            y.append(1 if str(sr["label"]) == "glorp" else 0)
            row_index.append(i)
    X = [str(t) for t in X]
    C = np.stack(C, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=int)
    ds = ConceptDatasetSample(
        X=X, C=C, y=y, meta={"concepts": tuple(names), "classes": (0,1), "data_type": "text"}
    )
    setattr(ds, "_full", type("Full", (), {"meta": {"row_index": np.asarray(row_index, dtype=int)}}))
    return ds



def _render_from_corpus(row: dict, corpus: list[dict], seed: int) -> str:
    sig = _signals_from_row(row)
    cand = [it for it in corpus if _line_matches(sig, it.get("when", {}))]
    if not cand:
        cand = corpus  # fallback

    key = f'{seed}:{row["head_shape"]}:{row["body_shape"]}:{row["foot_shape"]}:{row["ears_shape"]}:{row["mouth_type"]}:{row["hand_shape"]}:{row["has_antennae"]}:{row["has_knees"]}:{row["has_elbows"]}'
    idx = abs(hash(key)) % len(cand)
    txt = str(cand[idx]["text"])

    # 1) Naturalized placeholders (HEAD_NAT, BODY_NAT, etc.)
    nat = _nat_from_tokens(row)
    for k, v in nat.items():
        if "{" + k + "}" in txt:
            txt = txt.replace("{" + k + "}", v)

    # 2) Raw placeholders (head_shape, body_shape, …) — for plain template lines
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

def build_text_ds_hard(catalog_df: pd.DataFrame, concepts: dict, corpus_path: Path, variants_per_row: int, seed: int) -> ConceptDatasetSample:
    corpus = _load_jsonl(corpus_path)
    names = _names_from_concepts(concepts)
    classes = [0, 1]
    X, C, y, row_index = [], [], [], []
    for i, sr in catalog_df.iterrows():
        row = {k: sr[k] for k in concepts.keys()}
        row["label"] = sr["label"]
        for v in range(int(variants_per_row)):
            text = _render_from_corpus(row, corpus, seed + v)
            X.append(text)
            C.append(_onehot_for_row(row, concepts, names))
            y.append(1 if str(sr["label"]) == "glorp" else 0)
            row_index.append(i)
    C = np.stack(C, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=int)
    # Make a ConceptDatasetSample so downstream stays identical
    ds = ConceptDatasetSample(
        X=X, C=C, y=y,
        meta={"concepts": tuple(names), "classes": tuple(classes), "data_type": "text"}
    )
    # stash row_index so split logic can do "by_robot" if needed elsewhere
    setattr(ds, "_full", type("Full", (), {"meta": {"row_index": np.asarray(row_index, dtype=int)}}))
    return ds
# --------------------------------------------------------------

def train_eval_text(X_tr, y_tr, X_te, y_te, model_id, epochs, batch_size, lr, device):
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

# -------------------- arg parse --------------------
p = argparse.ArgumentParser(add_help=False)
p.add_argument("--modality", choices=["text", "image"])
p.add_argument("--n", type=int)
p.add_argument("--seed", type=int)
p.add_argument("--out_dir")
p.add_argument("--label_model_expr")
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
p.add_argument("--draw", type=int)
p.add_argument("--run-name", dest="run_name", type=str)
p.add_argument("--templates-file", type=str)
p.add_argument("--template-difficulty", choices=["easy","medium","hard"])
args, _ = p.parse_known_args()
for k, v in vars(args).items():
    if v is not None:
        settings[k] = v if k != "draw" else bool(v)

# -------------------- main --------------------
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

if modality == "text":
    # concept space
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
    label_expr = settings["label_model_expr"] or "'glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')) - 2 >= 0) else 'drent'"
    catalog_df["label"] = compute_label(catalog_df, label_expr)

    # pick template source
    tpl_path = None
    if settings.get("templates_file"):
        tpl_path = Path(settings["templates_file"])
    else:
        # default files: Templates.txt (medium), Templates_simple.txt (easy), HardCorpus.jsonl (hard)
        if settings.get("template_difficulty","medium") == "hard":
            cand = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "HardCorpus.jsonl"
            tpl_path = cand if cand.is_file() else None
        if tpl_path is None:
            template_file_name = "Templates.txt" if settings.get("template_difficulty","medium") == "medium" else "Templates_simple.txt"
            tpl_path = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / template_file_name

    # build dataset (hard JSONL vs legacy templates)
    is_jsonl = str(tpl_path).lower().endswith(".jsonl")
    if is_jsonl:
        ds = build_text_ds_hard(
            catalog_df=catalog_df,
            concepts=concepts,
            corpus_path=tpl_path,
            variants_per_row=int(settings["samples_per_instance"]),
            seed=int(settings["seed"]),
        )
    else:
        with open(tpl_path, "r", encoding="utf-8-sig") as f:
            templates = [ln.strip() for ln in f if ln.strip()]
        # keep legacy textgen for non-hard paths
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

    # simple 75/15/10 split
    idx = np.arange(len(ds.X))
    rng = np.random.default_rng(int(settings["seed"]))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(0.75 * n)
    n_val = int(0.15 * n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]

    # subset helper (ConceptDatasetSample expected)
    def _subset(ds_obj, take):
        X = [ds_obj.X[i] for i in take]
        C = ds_obj.C[take]
        y = ds_obj.y[take]
        return ConceptDatasetSample(X=X, C=C, y=y, meta={"concepts": ds_obj.concepts, "classes": ds_obj.classes, "data_type": "text"})

    ds.training = _subset(ds, tr)
    ds.validation = _subset(ds, va)
    ds.test = _subset(ds, te)

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

else:
    imgs_dir = pkg_dir / "synthetic" / "helper" / "static" / "robot_images_small"
    meta = pd.read_csv(imgs_dir / "meta.csv")
    meta = meta.sample(frac=1.0, random_state=int(settings["seed"])).reset_index(drop=True)
    label_expr = settings["label_model_expr"] or "'glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')) - 2 >= 0) else 'drent'"
    meta["label"] = compute_label(meta, label_expr)
    n = min(int(settings["n"]), len(meta))
    meta = meta.iloc[:n].reset_index(drop=True)
    split = int(0.8 * n)
    tr = meta.iloc[:split]
    te = meta.iloc[split:]
    paths_tr = [imgs_dir / p for p in tr["path"]]
    ytr = tr["label"].map({"drent": 0, "glorp": 1}).astype(int).values
    paths_te = [imgs_dir / p for p in te["path"]]
    yte = te["label"].map({"drent": 0, "glorp": 1}).astype(int).values
    acc, tok_or_proc, model = train_eval_image(
        paths_tr, ytr, paths_te, yte,
        model_id=model_id,
        size=int(settings["image_size"]),
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        lr=float(settings["lr"]),
        device=device,
    )

# -------------------- output --------------------
metrics = {
    "accuracy": float(acc),
    "seed": int(settings["seed"]),
    "modality": modality,
    "model": model_id,
    "n": int(settings["n"]),
}

processor_or_tok = tok_or_proc
seed_tag = f"seed{int(settings['seed'])}"
out_dir = Path(settings["out_dir"]) / modality / (settings.get("run_name", "").strip() or time.strftime("%Y%m%d_%H%M%S"))
out_dir.mkdir(parents=True, exist_ok=True)

legacy_metrics = out_dir / f"baseline_metrics_{seed_tag}.json"
legacy_metrics.write_text(json.dumps(metrics, indent=2))

named_metrics = out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_{seed_tag}_metrics.json"
named_metrics.write_text(json.dumps(metrics, indent=2))

model_dir = out_dir / f"baseline_dnn_robots_{modality}_{model_tag}_{seed_tag}_model"
model_dir.mkdir(parents=True, exist_ok=True)
processor_or_tok.save_pretrained(model_dir)
model.save_pretrained(model_dir)

print(json.dumps({
    "run_dir": str(out_dir),
    "metrics_path": str(legacy_metrics),
    "metrics_named_path": str(named_metrics),
    "model_dir": str(model_dir),
}, indent=2))
