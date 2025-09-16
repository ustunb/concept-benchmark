
from __future__ import annotations
import itertools, json, math, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from PIL import Image
from .robot_pil import draw_robot_image

try:
    from concept_benchmark.data import ConceptDataset, ConceptImageDatasetSample, TextConceptDataset
    HAVE_CB = True
except Exception:
    HAVE_CB = False

DEFAULT_CONCEPTS = {
    "head_shape": ["square", "round"],
    "body_shape": ["square", "round"],
    "has_knees": ["false", "true"],
    "has_elbows": ["false", "true"],
    "has_antennae": ["false", "true"],
    "ears_shape": ["square", "triangle"],
    "mouth_type": ["closed", "open"],
    "hand_shape": ["round_circle", "edgy_square"],
    "foot_shape": ["flat_4sided", "pointy_4sided"],
}

def _onehot_names(concepts: Dict[str, Sequence[str]]) -> List[str]:
    names = []
    for k, vals in concepts.items():
        if set(map(str.lower, vals)) == {"true","false"}:
            names.append(k)
        elif len(vals) == 2:
            names.append(f"{k}={vals[0]}")
        else:
            for v in vals:
                names.append(f"{k}={v}")
    return names

def _encode_row(row: Dict[str,str], concepts: Dict[str, Sequence[str]]) -> np.ndarray:
    bits = []
    for k, vals in concepts.items():
        v = str(row[k])
        if set(map(str.lower, vals)) == {"true","false"}:
            bits.append(1 if v.lower()=="true" else 0)
        elif len(vals) == 2:
            bits.append(1 if v == vals[0] else 0)
        else:
            for vv in vals:
                bits.append(1 if v == vv else 0)
    return np.asarray(bits, dtype=np.int64)

def _safe_eval_model(model: str, row: Dict[str,str]) -> str:
    loc = {"row": row, "int": int, "min": min, "max": max, "abs": abs}
    return eval(model, {"__builtins__": {}}, loc)

def _choose_colors(rng: np.random.Generator) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    def randc(): return (int(rng.integers(80,220)), int(rng.integers(80,220)), int(rng.integers(80,220)))
    return randc(), randc()

def _make_text(row: Dict[str,str], mention: Iterable[str]) -> str:
    parts = []
    if "head_shape" in mention: parts.append(f"Head is {row['head_shape']}.")
    if "body_shape" in mention: parts.append(f"Body is {row['body_shape']}.")
    if "has_elbows" in mention: parts.append(("Has elbows." if str(row['has_elbows']).lower()=='true' else "No elbows."))
    if "has_knees" in mention: parts.append(("Has knees." if str(row['has_knees']).lower()=='true' else "No knees."))
    if "has_antennae" in mention: parts.append(("Has antennae." if str(row['has_antennae']).lower()=='true' else "No antennae."))
    if "ears_shape" in mention: parts.append(f"Ears are {row['ears_shape']}.")
    if "mouth_type" in mention: parts.append(f"Mouth is {row['mouth_type']}.")
    if "hand_shape" in mention: parts.append(f"Hands are {row['hand_shape']}.")
    if "foot_shape" in mention: parts.append(f"Feet are {row['foot_shape']}.")
    return " ".join(parts).strip()

def _occludable():
    return {"head_shape","body_shape","has_elbows","has_knees","has_antennae","ears_shape","mouth_type","hand_shape","foot_shape"}

@dataclass
class MultiModalOutput:
    image_csv: Path
    text_csv: Path
    pairs_csv: Path
    image_dir: Path
    meta_json: Path

def create_multimodal_robot_dataset(mode: str, *, n: int = 200, concepts: Optional[Dict[str, Sequence[str]]] = None, seed: int = 0, out_dir: os.PathLike | str = "results/robots_mm", image_size: int = 256, color_mode: str = "color", p_overlap: float = 0.3, missing_rate: float = 0.2, model: Optional[str] = None) -> MultiModalOutput:
    rng = np.random.default_rng(seed)
    concepts = concepts or DEFAULT_CONCEPTS
    concept_names = _onehot_names(concepts)
    classes = ["drent","glorp"]
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        cfg = {}
        for k, vals in concepts.items():
            cfg[k] = str(vals[int(rng.integers(0, len(vals)))])
        y = None
        if model:
            y = _safe_eval_model(model, cfg)
        else:
            y = "glorp" if (int(cfg.get("body_shape")=="square") + int(cfg.get("foot_shape","").startswith("pointy")) - 1 >= 0) else "drent"
        left_color, right_color = _choose_colors(rng)
        if color_mode == "greyscale":
            to_gray = lambda c: (int(0.3*c[0]+0.59*c[1]+0.11*c[2]),)*3
            left_color, right_color = to_gray(left_color), to_gray(right_color)
        rows.append((i, cfg, y, left_color, right_color))
    occl_ok = _occludable()
    image_records = []
    text_records = []
    pair_records = []
    C_true_rows = []
    for i, cfg, y, cL, cR in rows:
        truth = _encode_row(cfg, concepts)
        C_true_rows.append(truth)
        cname = f"r{i:05d}.png"
        if mode not in {"complete_both","complete_union","incomplete_union"}:
            raise ValueError("mode must be one of: complete_both, complete_union, incomplete_union")
        text_mention = set(concepts.keys())
        img_hide = set()
        if mode == "complete_both":
            text_mention = set(concepts.keys())
            img_hide = set()
        elif mode == "complete_union":
            all_feats = list(concepts.keys())
            rng.shuffle(all_feats)
            k_text = int(math.ceil(len(all_feats)/2.0))
            text_mention = set(all_feats[:k_text])
            img_hide = set(k for k in all_feats[k_text:] if k in occl_ok)
            if p_overlap > 0:
                overlap = set(all_feats[:int(p_overlap*len(all_feats))])
                img_hide = set(k for k in img_hide if k not in overlap)
        else:
            all_feats = list(concepts.keys())
            miss = set(rng.choice(all_feats, size=max(1,int(missing_rate*len(all_feats))), replace=False).tolist())
            text_mention = set(k for k in all_feats if k not in miss)
            img_hide = set(k for k in miss if k in occl_ok)
        img, bboxes = draw_robot_image(cfg, size=image_size, occlude=img_hide, left_color=cL, right_color=cR)
        img_path = img_dir / cname
        img.save(img_path)
        txt = _make_text(cfg, text_mention)
        image_records.append({"id": i, "path": str(img_path.name), "y": classes.index(y), "text": txt})
        text_records.append({"id": i, "text": txt, "y": classes.index(y)})
        pair_records.append({
            "id": i,
            "image_path": str(img_path.name),
            "text": txt,
            "y": classes.index(y),
            "mask_img": json.dumps({k: (k not in img_hide) for k in concepts.keys()}),
            "mask_text": json.dumps({k: (k in text_mention) for k in concepts.keys()}),
            "C_true": json.dumps(truth.astype(int).tolist()),
        })
    df_img = pd.DataFrame(image_records)
    df_txt = pd.DataFrame(text_records)
    df_pairs = pd.DataFrame(pair_records)
    image_csv = out_dir / "images.csv"
    text_csv = out_dir / "texts.csv"
    pairs_csv = out_dir / "pairs.csv"
    df_img.to_csv(image_csv, index=False)
    df_txt.to_csv(text_csv, index=False)
    df_pairs.to_csv(pairs_csv, index=False)
    meta = {
        "classes": classes,
        "concepts_spec": concepts,
        "concept_names": _onehot_names(concepts),
        "mode": mode,
        "n": len(rows),
        "image_dir": str(img_dir),
        "image_csv": str(image_csv),
        "text_csv": str(text_csv),
        "pairs_csv": str(pairs_csv),
    }
    meta_json = out_dir / "meta.json"
    meta_json.write_text(json.dumps(meta, indent=2))
    if HAVE_CB:
        C_true = np.vstack(C_true_rows).astype(int)
        X_img = df_img["path"].values.tolist()
        X_txt = df_txt["text"].values.tolist()
        y = df_img["y"].values.astype(int)
        meta_img = {"classes": classes, "concepts": meta["concept_names"], "data_type": "image"}
        meta_txt = {"classes": classes, "concepts": meta["concept_names"], "data_type": "text"}
        if not isinstance(X_img, np.ndarray):
            X_img = np.array(X_img, dtype=object)
        if not isinstance(X_txt, np.ndarray):
            X_txt = np.array(X_txt, dtype=object)
        ds_img = ConceptDataset(X=X_img, C=C_true, y=y, meta=meta_img)
        ds_txt = TextConceptDataset(X=X_txt, C=C_true, y=y, meta=meta_txt)
        ds_img.training.base_dir = Path(meta["image_dir"])
        ds = {"image": ds_img, "text": ds_txt}
        (out_dir / "datasets.npz").write_bytes(b"")
    return MultiModalOutput(image_csv=image_csv, text_csv=text_csv, pairs_csv=pairs_csv, image_dir=img_dir, meta_json=meta_json)

def load_concept_datasets(meta_json_path: os.PathLike | str):
    if not HAVE_CB:
        raise RuntimeError("concept_benchmark.data not available in this environment")
    m = json.loads(Path(meta_json_path).read_text())
    import pandas as pd, numpy as np
    df_img = pd.read_csv(m["image_csv"])
    df_txt = pd.read_csv(m["text_csv"])
    pairs = pd.read_csv(m["pairs_csv"])
    C = np.vstack(pairs["C_true"].apply(json.loads).tolist()).astype(int)
    X_img = df_img["path"].tolist()
    X_txt = df_txt["text"].tolist()
    y = df_img["y"].values.astype(int)
    classes = m["classes"] if "classes" in m else ["drent","glorp"]
    meta_img = {"classes": classes, "concepts": m["concept_names"], "data_type": "image"}
    meta_txt = {"classes": classes, "concepts": m["concept_names"], "data_type": "text"}
    ds_img = ConceptDataset(X=X_img, C=np.zeros((len(X_img), len(m["concept_names"])), dtype=int), y=y, meta=meta_img)
    ds_txt = ConceptDataset(X=X_txt, C=np.zeros((len(X_txt), len(m["concept_names"])), dtype=int), y=y, meta=meta_txt)
    ds_img.training.base_dir = Path(m["image_dir"])
    return ds_img, ds_txt
