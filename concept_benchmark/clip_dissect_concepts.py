# concept_benchmark/clip_dissect_concepts.py
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Iterable, List, Tuple

def _pick_text(row: dict) -> str:
    for k in ("description", "desc", "label", "concept", "text"):
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""

def _pick_score(row: dict) -> float:
    for k in ("clip_similarity", "similarity", "score", "conf", "confidence"):
        try:
            return float(row.get(k))
        except Exception:
            continue
    return 0.0

def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in items:
        k = t.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t.strip())
    return out

def extract_concepts_from_descriptions_csv(descriptions_csv: str, out_jsonl: str | Path, n_concepts: int = 12) -> Path:
    p = Path(descriptions_csv)
    if not p.exists():
        raise SystemExit(f"CLIP-Dissect descriptions.csv not found: {p}")
    rows: List[Tuple[str, float]] = []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = _pick_text(row)
            if not t:
                continue
            rows.append((t, _pick_score(row)))
    if not rows:
        raise SystemExit(f"No usable concept texts in: {p}")
    rows.sort(key=lambda x: x[1], reverse=True)
    texts = [t for t,_ in rows]
    texts = _dedupe_keep_order(texts)
    texts = [t[:30] for t in texts if t.strip()]
    texts = texts[:int(n_concepts)]
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    return out
