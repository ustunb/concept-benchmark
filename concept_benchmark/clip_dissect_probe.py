# concept_benchmark/clip_dissect_probe.py
from __future__ import annotations
import shutil, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def _resolve_items(items, base_dir):
    from pathlib import Path
    base = Path(base_dir) if base_dir else Path(".")
    out = []
    for s in items:
        p = Path(str(s))
        q = p if p.is_absolute() else base / p
        out.append(str(q.resolve()))
    return out

def _to_df_indices(ds):
    idxs = ds.meta.get("df_indices", None)
    if idxs is None:
        return np.arange(ds.n)
    return np.asarray(idxs, dtype=int)

def _subset_indices_by_query(full_catalog: pd.DataFrame, idxs: np.ndarray, query: str) -> np.ndarray:
    sub = full_catalog.iloc[idxs]
    sel = sub.query(query, engine="python").index.values
    pos = np.nonzero(np.isin(idxs, sel))[0]
    mask = np.zeros_like(idxs, dtype=bool)
    mask[pos] = True
    return mask

def _choose_by_class(ds, counts: Dict[str,int]) -> np.ndarray:
    y = np.asarray(ds.y, dtype=int)
    classes = list(map(str, ds.meta.get("classes", [])))
    mask = np.zeros(ds.n, dtype=bool)
    rng = np.random.default_rng(int(ds.meta.get("seed", 0)))
    for cname, k in counts.items():
        if cname.isdigit():
            cidx = int(cname)
        else:
            try:
                cidx = classes.index(cname)
            except Exception:
                continue
        where = np.where(y == cidx)[0]
        rng.shuffle(where)
        where = where[: int(k)]
        mask[where] = True
    return mask

def _collect_stats(ds, sel_mask: np.ndarray) -> Dict:
    y = np.asarray(ds.y, dtype=int)[sel_mask]
    classes = list(map(str, ds.meta.get("classes", [])))
    class_counts = {classes[i]: int((y == i).sum()) for i in range(len(classes))}
    C = np.asarray(ds.C, dtype=int)[sel_mask]
    concepts = list(map(str, ds.concepts))
    subconcepts = {concepts[i]: int(C[:,i].sum()) for i in range(len(concepts))}
    per_class = {}
    for i, cname in enumerate(classes):
        m = (y == i)
        if m.any():
            per_class[cname] = {concepts[j]: int(C[m, j].sum()) for j in range(len(concepts))}
        else:
            per_class[cname] = {concepts[j]: 0 for j in range(len(concepts))}
    return {"classes": class_counts, "subconcept_counts": subconcepts, "by_class_subconcept_counts": per_class}

def export_probe_dataset(train, valid, test, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    select = str(cfg.get("probe_select", "train")).lower()
    parts = []
    if select in {"train", "training"}:
        parts = [("train", train)]
    elif select in {"test", "testing"}:
        parts = [("test", test)]
    elif select in {"trainval", "train+valid", "train_valid"}:
        parts = [("train", train), ("valid", valid)]
    elif select in {"all", "trainvaltest", "train+valid+test"}:
        parts = [("train", train), ("valid", valid), ("test", test)]
    else:
        parts = [("train", train)]

    full_catalog = train.meta.get("catalog_df", None)
    export_rows = []

    for split_name, ds in parts:
        base_dir = getattr(ds, "base_dir", None) or ds.meta.get("image_root") or ds.meta.get("root")
        X = np.asarray(ds.X, dtype=object)
        sel_mask = np.ones(ds.n, dtype=bool)

        subset_cfg = (cfg.get("probe_subset", {}) or {})
        where = str(subset_cfg.get("where", "") or "").strip()
        if where and isinstance(full_catalog, pd.DataFrame):
            df_idxs = _to_df_indices(ds)
            sel_mask = _subset_indices_by_query(full_catalog, df_idxs, where)

        per_class = subset_cfg.get("per_class", None)
        if isinstance(per_class, dict) and per_class:
            sel_mask &= _choose_by_class(ds, per_class)

        limit = int(subset_cfg.get("limit", 0) or 0)
        if limit > 0 and sel_mask.sum() > limit:
            rng = np.random.default_rng(int(cfg.get("seed", 0)))
            idxs = np.where(sel_mask)[0]
            rng.shuffle(idxs)
            keep = np.zeros_like(sel_mask)
            keep[idxs[:limit]] = True
            sel_mask = keep

        stats = _collect_stats(ds, sel_mask)
        (out_dir / f"probe_stats_{split_name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

        items = []
        for i in np.where(sel_mask)[0]:
            path = _resolve_items([X[i]], base_dir)[0]
            y = int(ds.y[i])
            cname = str(ds.meta.get("classes", [y])[y] if ds.meta.get("classes") else y)
            dst = images_dir / str(cname) / f"{split_name}-{i:06d}{Path(path).suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not Path(path).is_file():
                raise SystemExit(f"probe export: missing source image: {path}")
            shutil.copy2(path, dst)
            items.append({"rel_path": dst.relative_to(out_dir).as_posix(), "class": cname, "origin": path})

        export_rows.extend(items)

    import csv
    index_path = out_dir / "probe_index.csv"
    if not export_rows:
        raise SystemExit("probe export: no images were copied; check probe_select/probe_subset and source paths.")
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rel_path", "class", "origin"])
        writer.writeheader()
        for it in export_rows:
            writer.writerow(it)

    summary = {"n_images": len(export_rows), "index_csv": str(index_path), "images_dir": str(images_dir)}

    (out_dir / "probe_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
