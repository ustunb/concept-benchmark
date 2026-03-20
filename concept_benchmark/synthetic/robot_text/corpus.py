"""JSONL corpus loading and text rendering for robot descriptions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from concept_benchmark.synthetic.robot_text.catalog import CORE_CONCEPT_NAMES

if TYPE_CHECKING:
    from concept_benchmark.config import RobotBenchmarkConfig


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
        "round_circle": [
            "round mitts",
            "round hands",
            "circular mitts",
            "circular hands",
            "rounded mitts",
        ],
        "round_oval": [
            "wide ovals",
            "broad ovals",
            "oval hands",
            "broad-oval hands",
            "wide-oval hands",
        ],
        "round_oval2": [
            "tall ovals",
            "long ovals",
            "elongated ovals",
            "oval grips",
            "oval mitts",
        ],
        "edgy_triangle": [
            "triangular grippers",
            "pointed grippers",
            "three-angled grippers",
            "tri-point grippers",
            "tapered grippers",
        ],
        "edgy_square": [
            "square-edged grippers",
            "square claws",
            "right-angled grippers",
            "angular grippers",
            "square clamps",
        ],
        "edgy_trapezoid": [
            "trapezoid grippers",
            "trapezoidal grippers",
            "trapezoid claws",
            "angled trapezoids",
            "trapezoid clamps",
        ],
    }
    feet_map = {
        "flat_trapezoid": [
            "flat trapezoid pads",
            "flat trapezoid feet",
            "trapezoidal pads",
            "trapezoidal feet",
            "flat angled pads",
        ],
        "flat_rounded": [
            "flat rounded pads",
            "flat rounded feet",
            "rounded flat pads",
            "smooth flat feet",
            "curved flat pads",
        ],
        "flat_square": [
            "flat square pads",
            "flat square feet",
            "flat quad pads",
            "flat quad feet",
            "flat four-sided pads",
        ],
        "flat_5sided": [
            "flat five-sided pads",
            "flat pentagonal pads",
            "flat five-sided feet",
            "flat pentagon pads",
            "flat pentagon feet",
        ],
        "flat_lshaped": [
            "L-shaped feet",
            "L-shaped pads",
            "ell-shaped feet",
            "ell-shaped pads",
            "right-angle feet",
        ],
        "pointy_trapezoid": [
            "pointed trapezoid feet",
            "pointy trapezoid pads",
            "tapered trapezoid feet",
            "sharp trapezoid pads",
            "angled point feet",
        ],
        "pointy_rounded": [
            "pointed rounded feet",
            "pointy rounded pads",
            "rounded point feet",
            "curved point pads",
            "round-tipped feet",
        ],
        "pointy_square": [
            "pointed square feet",
            "pointy square pads",
            "square point feet",
            "angular point pads",
            "square-tipped feet",
        ],
        "pointy_3sided": [
            "three-point feet",
            "triangular points",
            "tri-point feet",
            "three-tipped feet",
            "tri-tipped feet",
        ],
        "pointy_4sided": [
            "four-point feet",
            "quad-point feet",
            "four-tipped feet",
            "quad-tipped feet",
            "pointed four-sided feet",
        ],
    }

    return {
        "HEAD_NAT": pick(head_map[str(row["head_shape"])], "HEAD"),
        "BODY_NAT": pick(body_map[str(row["body_shape"])], "BODY"),
        "EARS_NAT": pick(ears_map[str(row["ears_shape"])], "EARS"),
        "MOUTH_NAT": pick(mouth_map[str(row["mouth_type"])], "MOUTH"),
        "HANDS_NAT": pick(hands_map[str(row["hand_shape"])], "HANDS"),
        "FEET_NAT": pick(feet_map[str(row["foot_shape"])], "FEET"),
        "ANT_NAT": "has antennae"
        if str(row["has_antennae"]).lower() == "true"
        else "no antennae",
        "KNEES_NAT": "has knees"
        if str(row["has_knees"]).lower() == "true"
        else "no knees",
        "ELBOWS_NAT": "has elbows"
        if str(row["has_elbows"]).lower() == "true"
        else "no elbows",
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
        f"{seed}:{row['head_shape']}:{row['body_shape']}:{row['foot_shape']}:"
        f"{row['ears_shape']}:{row['mouth_type']}:{row['hand_shape']}:"
        f"{row['has_antennae']}:{row['has_knees']}:{row['has_elbows']}"
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
    """Convert a concept row to a 9-element binary vector (original fixed layout)."""
    return concept_vector_from_row(row, CORE_CONCEPT_NAMES)


# ── Dynamic concept vector builder ────────────────────────────────────

# Maps each feature to (binary concept name, test function).
# The test function takes a row value and returns 1.0 or 0.0.
_BINARY_CONCEPT_MAP: dict[str, tuple[str, Callable]] = {
    "head_shape": ("head_is_square", lambda v: 1.0 if str(v) == "square" else 0.0),
    "body_shape": ("body_is_square", lambda v: 1.0 if str(v) == "square" else 0.0),
    "has_knees": ("has_knees", lambda v: 1.0 if str(v).lower() == "true" else 0.0),
    "has_elbows": ("has_elbows", lambda v: 1.0 if str(v).lower() == "true" else 0.0),
    "foot_shape": (
        "foot_is_pointy",
        lambda v: 1.0 if str(v).startswith("pointy_") else 0.0,
    ),
    "has_antennae": (
        "has_antennae",
        lambda v: 1.0 if str(v).lower() == "true" else 0.0,
    ),
    "ears_shape": ("ears_is_triangle", lambda v: 1.0 if str(v) == "triangle" else 0.0),
    "mouth_type": ("mouth_is_open", lambda v: 1.0 if str(v) == "open" else 0.0),
    "hand_shape": (
        "hands_are_pointy",
        lambda v: 1.0 if str(v).startswith("edgy_") else 0.0,
    ),
}


def compute_text_concept_names(
    concepts: dict[str, list],
    expand: list[str] | None = None,
) -> list[str]:
    """Build concept column names based on which features are expanded.

    Args:
        concepts: Feature name → list of values (e.g. ROBOT_CONCEPTS).
        expand: Which features to expand into one-hot subtypes
            (e.g. ``["foot_shape"]``). If None, all features are
            collapsed to binary.

    Returns:
        List of concept names in a stable order.
    """
    expanded_features = set(expand or [])

    names: list[str] = []
    for feat in concepts:
        if feat in expanded_features:
            # One-hot: one column per subtype value
            for val in concepts[feat]:
                names.append(f"{feat}_{val}")
        elif feat in _BINARY_CONCEPT_MAP:
            names.append(_BINARY_CONCEPT_MAP[feat][0])
    return names


def concept_vector_from_row(
    row: dict,
    concept_names: list[str],
) -> np.ndarray:
    """Build a binary concept vector for the given concept names.

    Handles both collapsed features (binary test) and expanded subtypes
    (exact match on ``feat_val``).
    """
    vec = np.zeros(len(concept_names), dtype=np.float32)
    for i, name in enumerate(concept_names):
        # Check if it's a collapsed binary concept
        found = False
        for feat, (cname, test_fn) in _BINARY_CONCEPT_MAP.items():
            if name == cname:
                vec[i] = test_fn(row[feat])
                found = True
                break
        if found:
            continue
        # Must be an expanded subtype: name = "{feature}_{value}"
        # Find the feature by checking which feature name is a prefix
        for feat in row:
            prefix = f"{feat}_"
            if name.startswith(prefix):
                val = name[len(prefix) :]
                vec[i] = 1.0 if str(row[feat]) == val else 0.0
                break
    return vec


# ── Corpus path resolution ────────────────────────────────────────────


def get_corpus_path(config: RobotBenchmarkConfig) -> Path:
    """Resolve the corpus path based on difficulty setting."""
    from concept_benchmark.paths import package_dir

    if config.template_complexity == "high":
        p = (
            package_dir
            / "synthetic"
            / "helper"
            / "static"
            / "text_templates"
            / "hard_corpus.jsonl"
        )
        if p.is_file():
            return p
    name = (
        "templates.txt"
        if config.template_complexity == "medium"
        else "templates_simple.txt"
    )
    return package_dir / "synthetic" / "helper" / "static" / "text_templates" / name
