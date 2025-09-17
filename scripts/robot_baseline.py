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
from concept_benchmark.synthetic.robot import (
    create_robot_text_dataset as make_text_ds,
    create_synthetic_dataset as make_image_ds,
)
from concept_benchmark.data import ConceptDatasetSample

settings = {
    "modality": "text",
    "n": 5000,
    "seed": 0,
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
    "color_mode": "color",
    "samples_per_instance": 1,
    "draw": 0,
}

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def _csv_list(s):
    s = str(s).strip()
    return [t.strip() for t in s.split(",")] if s else []

def compute_label(df, model_expr):
    g = {"__builtins__": None, "int": int, "str": str, "float": float, "bool": bool, "any": any, "all": all}
    return df.apply(lambda r: eval(model_expr, g, {"row": r.to_dict()}), axis=1)

def _subset_sample(sample, keep_idx):
    keep_idx = np.asarray(keep_idx, dtype=int)
    X = [str(x) for x in np.array(sample.X, dtype=object)[keep_idx]]
    C = sample.C[keep_idx]
    y = sample.y[keep_idx]
    return ConceptDatasetSample(X=X, C=C, y=y, meta=sample.meta)

def _group_indices(names, key):
    return [i for i, n in enumerate(names) if n.startswith(key + "=")]

def _corr_equal_mask(sample, a, b):
    names = list(sample.concepts)
    ai = _group_indices(names, a)
    bi = _group_indices(names, b)
    T = sample.C.astype(int)
    aa = T[:, ai].argmax(1)
    bb = T[:, bi].argmax(1)
    return aa == bb

def _enforce_corr(sample, pair, frac_corr, seed):
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

class TextDS(Dataset):
    def __init__(self, X, y, tok, max_length=256):
        self.X = list(map(str, X))
        self.y = np.asarray(y, dtype=int)
        self.tok = tok
        self.max_length = max_length
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        enc = self.tok(self.X[i], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
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

def train_model(modality, train_X, train_y, test_X, test_y, settings, device):
    if modality == "text":
        tok = AutoTokenizer.from_pretrained(settings["text_model"])
        model = AutoModelForSequenceClassification.from_pretrained(settings["text_model"], num_labels=2).to(device)
        ds_tr = TextDS(train_X, train_y, tok)
        ds_te = TextDS(test_X, test_y, tok)
    else:
        proc = AutoImageProcessor.from_pretrained(settings["image_model"])
        model = AutoModelForImageClassification.from_pretrained(settings["image_model"], num_labels=2).to(device)
        ds_tr = ImageDS(train_X, train_y, proc)
        ds_te = ImageDS(test_X, test_y, proc)
    dl_tr = DataLoader(ds_tr, batch_size=settings["batch_size"], shuffle=True, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    for ep in range(settings["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb, labels=yb)
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
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
    return float(acc)

def main():
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
    args, _ = p.parse_known_args()
    for k, v in vars(args).items():
        if v is not None:
            settings[k] = v if k != "draw" else bool(v)
    set_seed(int(settings["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    out_dir = Path(settings["out_dir"]) / settings["modality"] / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if settings["modality"] == "text":
        tpl_path = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "Templates.txt"
        with open(tpl_path, "r", encoding="utf-8-sig") as f:
            templates = [ln.strip() for ln in f if ln.strip()]
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
        grids = [concepts[c] for c in cols]
        combos = list(product(*grids))
        catalog_df = pd.DataFrame(combos, columns=cols)
        model_expr = settings["label_model_expr"] or "('glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')))>=1 else 'drent')"
        catalog_df["label"] = compute_label(catalog_df, model_expr)
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
        ds.generate_cvindices(seed=int(settings["seed"]))
        ds.split("K05N01", fold_num_validation=4, fold_num_test=5)
        train = ds.training
        test = ds.test
        if settings["corr_pair"]:
            train = _enforce_corr(train, settings["corr_pair"], float(settings["train_corr"]), int(settings["seed"]))
            test = _enforce_corr(test, settings["corr_pair"], max(0.0, 1.0 - float(settings["test_break"])), int(settings["seed"]) + 1)
        acc = train_model(
            "text",
            train_X=train.X,
            train_y=train.y,
            test_X=test.X,
            test_y=test.y,
            settings=settings,
            device=device,
        )
    else:
        img_dir = out_dir / "images"
        params = {
            "samples_per_instance": int(settings["samples_per_instance"]),
            "draw": bool(settings["draw"]),
            "output_directory": str(img_dir),
            "concepts": {
                "head_shape": ["square", "round"],
                "body_shape": ["square", "round"],
                "has_knees": ["false", "true"],
                "has_elbows": ["false", "true"],
                "has_antennae": ["false", "true"],
                "ears_shape": ["square", "triangle"],
                "mouth_type": ["closed", "open"],
                "hand_shape": ["round_circle","round_oval","round_oval2","edgy_triangle","edgy_square","edgy_trapezoid"],
                "foot_shape": ["flat_4sided","flat_5sided","flat_lshaped","pointy_3sided","pointy_4sided","pointy_6sided"],
            },
            "spurious_features": ["has_elbows","hand_shape"],
            "model": settings["label_model_expr"] or "('glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')))>=1 else 'drent')",
            "model_type": "deterministic",
            "size": "large",
            "color_mode": settings["color_mode"],
            "train_concept_detector": False,
            "epochs": 1,
            "verbose": True,
        }
        img_dir.mkdir(parents=True, exist_ok=True)
        ds = make_image_ds(**params)
        ds.generate_cvindices(seed=int(settings["seed"]))
        ds.split("K05N01", fold_num_validation=4, fold_num_test=5)
        train = ds.training
        test = ds.test
        if settings["corr_pair"]:
            train = _enforce_corr(train, settings["corr_pair"], float(settings["train_corr"]), int(settings["seed"]))
            test = _enforce_corr(test, settings["corr_pair"], max(0.0, 1.0 - float(settings["test_break"])), int(settings["seed"]) + 1)
        acc = train_model(
            "image",
            train_X=train.X,
            train_y=train.y,
            test_X=test.X,
            test_y=test.y,
            settings=settings,
            device=device,
        )

    metrics = {"accuracy": float(acc)}
    Path(out_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"out_dir": str(out_dir), "metrics_path": str(Path(out_dir, "metrics.json"))}))
