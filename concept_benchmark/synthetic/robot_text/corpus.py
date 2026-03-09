"""JSONL corpus loading and text rendering for robot descriptions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def load_jsonl(p: Path) -> list[dict]:
    """Load a JSONL, JSON array, or plain-text corpus file."""
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"HardCorpus file is empty: {p}")

    # JSON array
    if text.startswith("["):
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError("Top-level JSON is not a list")
        return arr

    items: list[dict] = []
    plain_lines: list[str] = []

    for ln in text.splitlines():
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
        return [
            {"id": f"pt_{i:04d}", "when": {"any": True}, "text": s}
            for i, s in enumerate(plain_lines, 1)
        ]
    raise ValueError(f"No valid JSON or plain-text lines found in {p}.")


def signals_from_row(row: dict) -> dict:
    """Extract matching signals from a concept row for corpus filtering."""
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


def line_matches(sig: dict, cond: dict) -> bool:
    """Check if a signal dict matches a corpus entry's condition dict."""
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


def nat_from_tokens(row: dict, seed: int) -> dict:
    """Generate natural language tokens for a robot row using SHA-256 synonym selection."""
    fp = (
        f"{row['head_shape']}|{row['body_shape']}|{row['foot_shape']}|"
        f"{row['ears_shape']}|{row['mouth_type']}|{row['hand_shape']}|"
        f"{row['has_antennae']}|{row['has_knees']}|{row['has_elbows']}"
    )

    def pick(opts, key):
        h = hashlib.sha256(f"{seed}:{fp}:{key}".encode()).hexdigest()
        return opts[int(h, 16) % len(opts)]

    head_map = {
        "square": ["square", "square-shaped", "boxy", "right-angled", "angular"],
        "round": ["round", "rounded", "dome-shaped", "curved", "circular"],
    }
    body_map = {
        "square": ["square", "square-bodied", "boxy", "right-angled", "angular"],
        "round": ["rounded", "barrel-shaped", "curved", "tubular", "cylindrical"],
    }
    ears_map = {
        "square": ["square", "square-cut", "right-angled", "box-form", "angular"],
        "triangle": ["triangular", "three-angled", "pointed", "tri-corner", "tapered"],
    }
    mouth_map = {"closed": ["closed"], "open": ["open"]}
    hands_map = {
        "round_circle": ["round mitts", "round hands", "circular mitts", "circular hands", "rounded mitts"],
        "wide_oval": ["wide ovals", "broad ovals", "oval hands", "broad-oval hands", "wide-oval hands"],
        "tall_oval": ["tall ovals", "long ovals", "elongated ovals", "oval grips", "oval mitts"],
        "edgy_square": ["square-edged grippers", "square claws", "right-angled grippers", "angular grippers", "square clamps"],
        "edgy_triangle": ["triangular grippers", "pointed grippers", "three-angled grippers", "tri-point grippers", "tapered grippers"],
        "edgy_trapezoid": ["trapezoid grippers", "trapezoidal grippers", "trapezoid claws", "angled trapezoids", "trapezoid clamps"],
    }
    feet_map = {
        "flat_4sided": ["flat four-sided pads", "flat four-sided feet", "flat quad pads", "flat quad feet", "flat square pads"],
        "flat_5sided": ["flat five-sided pads", "flat pentagonal pads", "flat five-sided feet", "flat pentagon pads", "flat pentagon feet"],
        "flat_lshaped": ["L-shaped feet", "L-shaped pads", "ell-shaped feet", "ell-shaped pads", "right-angle feet"],
        "pointy_3sided": ["three-point feet", "triangular points", "tri-point feet", "three-tipped feet", "tri-tipped feet"],
        "pointy_4sided": ["four-point feet", "quad-point feet", "four-tipped feet", "quad-tipped feet", "pointed four-sided feet"],
        "pointy_6sided": ["six-point feet", "hex-point feet", "six-tipped feet", "hex-tipped feet", "pointed hex feet"],
    }

    return {
        "HEAD_NAT": pick(head_map[str(row["head_shape"])], "HEAD"),
        "BODY_NAT": pick(body_map[str(row["body_shape"])], "BODY"),
        "EARS_NAT": pick(ears_map[str(row["ears_shape"])], "EARS"),
        "MOUTH_NAT": pick(mouth_map[str(row["mouth_type"])], "MOUTH"),
        "HANDS_NAT": pick(hands_map[str(row["hand_shape"])], "HANDS"),
        "FEET_NAT": pick(feet_map[str(row["foot_shape"])], "FEET"),
        "ANT_NAT": "has antennae" if str(row["has_antennae"]).lower() == "true" else "no antennae",
        "KNEES_NAT": "has knees" if str(row["has_knees"]).lower() == "true" else "no knees",
        "ELBOWS_NAT": "has elbows" if str(row["has_elbows"]).lower() == "true" else "no elbows",
    }


def render_from_corpus(row: dict, corpus: list[dict], seed: int) -> str:
    """Render a text description for a robot row from a JSONL corpus."""
    try:
        sig = signals_from_row(row)
        cand = [it for it in corpus if line_matches(sig, it.get("when", {}))]
        if not cand:
            cand = corpus
    except Exception:
        cand = corpus

    key = (
        f'{seed}:{row["head_shape"]}:{row["body_shape"]}:{row["foot_shape"]}:'
        f'{row["ears_shape"]}:{row["mouth_type"]}:{row["hand_shape"]}:'
        f'{row["has_antennae"]}:{row["has_knees"]}:{row["has_elbows"]}'
    )
    idx = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(cand)
    txt = str(cand[idx].get("text", ""))

    # Replace natural language tokens
    nat = nat_from_tokens(row, seed)
    for k, v in nat.items():
        ph = "{" + k + "}"
        if ph in txt:
            txt = txt.replace(ph, v)

    # Replace raw attribute placeholders
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


def core_vector_from_row(row: dict) -> np.ndarray:
    """Convert a concept row to a 9-element binary vector."""
    def b(x):
        return str(x).lower()

    return np.array([
        1.0 if str(row["head_shape"]) == "square" else 0.0,
        1.0 if str(row["body_shape"]) == "square" else 0.0,
        1.0 if b(row["has_knees"]) == "true" else 0.0,
        1.0 if b(row["has_elbows"]) == "true" else 0.0,
        1.0 if str(row["foot_shape"]).startswith("pointy_") else 0.0,
        1.0 if b(row["has_antennae"]) == "true" else 0.0,
        1.0 if str(row["ears_shape"]) == "triangle" else 0.0,
        1.0 if str(row["mouth_type"]) == "open" else 0.0,
        1.0 if str(row["hand_shape"]).startswith("edgy_") else 0.0,
    ], dtype=np.float32)
