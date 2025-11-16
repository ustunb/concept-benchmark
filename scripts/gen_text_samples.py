from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd


DEFAULT_CONCEPTS = {
    "head_shape": ["square", "round"],
    "body_shape": ["square", "round"],
    "has_knees": ["false", "true"],
    "has_elbows": ["false", "true"],
    "foot_shape": [
        "flat_4sided",
        "flat_5sided",
        "flat_lshaped",
        "pointy_3sided",
        "pointy_4sided",
        "pointy_6sided",
    ],
    "has_antennae": ["false", "true"],
    "ears_shape": ["square", "triangle"],
    "mouth_type": ["closed", "open"],
    "hand_shape": [
        "round_circle",
        "wide_oval",
        "tall_oval",
        "edgy_square",
        "edgy_triangle",
        "edgy_trapezoid",
    ],
}


DEFAULT_LABEL_EXPR = (
    "'glorp' if (str(row['mouth_type']) == 'closed' or str(row['foot_shape']).startswith('pointy_')) else 'drent'"
)


settings: Dict[str, Any] = {
    "concepts_spec": "",
    "difficulty": "easy",
    "mode": "template",
    "samples_per_instance": 3,
    "use_llm": 0,
    "llm_provider": "openai",
    "llm_model": "gpt-4.1-mini",
    "llm_system_prompt": "",
    "llm_user_prompt": "",
    "wrinkles": "",
    "spurious_concepts": "",
    "subconcepts": "",
    "correlations": "",
    "shuffle_concepts": 0,
    "seed": 1337,
    "output_prefix": "robot_text_dataset",
    "mask_concepts": "",
    "omit_concepts": "",
    "mask_tolerance": 0.05,
    "max_instances": 0,
    "audit_every": 0,
    "audit_max_samples": 0,
    "audit_model": "",
    "audit_system_prompt": "",
    "audit_user_prompt": "",
    "audit_output": "",
}


@dataclass
class CorrelationSpec:
    concept: str
    value: str
    label: str
    prob: float


@dataclass
class TextConceptDataset:
    texts: List[str]
    C: np.ndarray
    y: np.ndarray
    concept_names: List[str]
    label_names: List[str]
    meta: Dict[str, Any]


def _parse_concepts_spec(path: str | None) -> Dict[str, List[str]]:
    if not path:
        return {k: list(v) for k, v in DEFAULT_CONCEPTS.items()}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Concept spec file not found: {path}")
    if p.suffix.lower() in {".json", ".jsonl"}:
        data = json.loads(p.read_text())
        if "concepts" in data:
            raw = data["concepts"]
        else:
            raw = data
        return {k: [str(x) for x in v] for k, v in raw.items()}
    if p.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(p)
        out: Dict[str, List[str]] = {}
        for col in df.columns:
            out[col] = sorted(df[col].dropna().astype(str).unique().tolist())
        return out
    raise SystemExit(f"Unsupported concept spec format: {path}")


def _parse_list_arg(s: str | None) -> List[str]:
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _parse_correlations(s: str | None) -> List[CorrelationSpec]:
    out: List[CorrelationSpec] = []
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lhs, label, prob_str = part.split(":")
            concept, value = lhs.split("=", 1)
            out.append(
                CorrelationSpec(
                    concept=concept.strip(),
                    value=value.strip(),
                    label=label.strip(),
                    prob=float(prob_str),
                )
            )
        except Exception:
            raise SystemExit(f"Bad correlation spec: {part}")
    return out


def _parse_mask_spec(parts: List[str]) -> Dict[str, float]:
    spec: Dict[str, float] = {}
    for raw in parts:
        s = raw.strip()
        if not s:
            continue
        if ":" in s:
            name, frac_str = s.split(":", 1)
            name = name.strip()
            try:
                frac = float(frac_str)
            except Exception:
                raise SystemExit(f"Bad mask fraction in '{s}'")
            if not (0.0 <= frac <= 1.0):
                raise SystemExit(f"Mask fraction for '{name}' must be in [0, 1], got {frac}")
            spec[name] = frac
        else:
            spec[s] = 1.0
    return spec


def _build_catalog(concepts: Dict[str, Sequence[str]]) -> pd.DataFrame:
    cols = list(concepts.keys())
    rows = [dict(zip(cols, vals)) for vals in product(*[concepts[c] for c in cols])]
    return pd.DataFrame(rows, columns=cols)


def _apply_label_expr(df: pd.DataFrame, expr: str) -> pd.Series:
    labels: List[str] = []
    safe_globals = {
        "__builtins__": {},
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }
    for _, row in df.iterrows():
        val = eval(expr, safe_globals, {"row": row})
        labels.append(str(val))
    return pd.Series(labels, index=df.index)


def _apply_correlations(df: pd.DataFrame, corr_specs: List[CorrelationSpec], label_col: str) -> pd.DataFrame:
    if not corr_specs:
        return df
    df = df.copy()
    rng = np.random.default_rng(0)
    labels = df[label_col].astype(str).to_numpy()
    for spec in corr_specs:
        if spec.concept not in df.columns:
            continue
        mask = (df[spec.concept].astype(str) == spec.value) & (labels != spec.label)
        if not mask.any():
            continue
        flip = rng.random(size=mask.sum()) < spec.prob
        labels_sub = labels[mask]
        labels_sub[flip] = spec.label
        labels[mask] = labels_sub
    df[label_col] = labels
    return df


def _canonicalize_bools(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def canon(x: Any) -> str:
        sx = str(x).strip().lower()
        if sx in {"1", "true", "t", "yes", "y"}:
            return "true"
        if sx in {"0", "false", "f", "no", "n"}:
            return "false"
        return sx

    for col in ["has_antennae", "has_knees", "has_elbows"]:
        if col in df.columns:
            df[col] = df[col].map(canon)
    return df


def _bool_from_str(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _describe_head(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "head_shape" in omit or "head_shape" not in row:
        return ""
    if "head_shape" in mask:
        return "a head"
    val = str(row["head_shape"])
    return f"a {val} head"


def _describe_body(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "body_shape" in omit or "body_shape" not in row:
        return ""
    if "body_shape" in mask:
        return "a body"
    val = str(row["body_shape"])
    return f"a {val} body"


def _describe_knees(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "has_knees" in omit or "has_knees" not in row:
        return ""
    if "has_knees" in mask:
        return "legs"
    has = _bool_from_str(row["has_knees"])
    return "bendy knees" if has else "rigid legs with no knees"


def _describe_elbows(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "has_elbows" in omit or "has_elbows" not in row:
        return ""
    if "has_elbows" in mask:
        return "arms"
    has = _bool_from_str(row["has_elbows"])
    return "flexible elbows" if has else "straight, stiff arms"


def _describe_foot(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "foot_shape" in omit or "foot_shape" not in row:
        return ""
    if "foot_shape" in mask:
        return "feet"
    val = str(row["foot_shape"])
    if val.startswith("flat_"):
        return "flat feet"
    if val.startswith("pointy_"):
        return "pointy feet"
    return "feet"


def _describe_antennae(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "has_antennae" in omit or "has_antennae" not in row:
        return ""
    if "has_antennae" in mask:
        return "a feature on top of its head"
    has = _bool_from_str(row["has_antennae"])
    return "a pair of antennae" if has else "no antennae"


def _describe_ears(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "ears_shape" in omit or "ears_shape" not in row:
        return ""
    if "ears_shape" in mask:
        return "ears"
    val = str(row["ears_shape"])
    if val == "square":
        return "square ears"
    if val == "triangle":
        return "triangular ears"
    return "ears"


def _describe_mouth(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "mouth_type" in omit or "mouth_type" not in row:
        return ""
    if "mouth_type" in mask:
        return "a mouth"
    val = str(row["mouth_type"])
    if val == "open":
        return "an open mouth"
    if val == "closed":
        return "a closed mouth"
    return "a mouth"


def _describe_hands(row: Dict[str, str], mask: Set[str], omit: Set[str]) -> str:
    if "hand_shape" in omit or "hand_shape" not in row:
        return ""
    if "hand_shape" in mask:
        return "hands"
    val = str(row["hand_shape"])
    if val.startswith("round_") or val.endswith("_oval"):
        return "rounded hands"
    if val.startswith("edgy_"):
        return "sharp, edgy hands"
    return "hands"


def _generate_template_sentence(
    row: Dict[str, str],
    difficulty: str,
    mask: Set[str],
    omit: Set[str],
    rng: np.random.Generator,
    wrinkles: str | None = None,
) -> str:
    head = _describe_head(row, mask, omit)
    body = _describe_body(row, mask, omit)
    knees = _describe_knees(row, mask, omit)
    elbows = _describe_elbows(row, mask, omit)
    hands = _describe_hands(row, mask, omit)
    foot = _describe_foot(row, mask, omit)
    antennae = _describe_antennae(row, mask, omit)
    ears = _describe_ears(row, mask, omit)
    mouth = _describe_mouth(row, mask, omit)

    shape_bits = [x for x in [head, body] if x]
    limb_bits = [x for x in [knees, elbows, hands] if x]
    face_bits = [x for x in [antennae, ears, mouth] if x]
    foot_bits = [x for x in [foot] if x]

    sentences: List[str] = []

    if shape_bits:
        sentences.append("This robot has " + " and ".join(shape_bits) + ".")
    if limb_bits:
        sentences.append("It has " + " and ".join(limb_bits) + ".")
    if foot_bits:
        sentences.append("Its " + " and ".join(foot_bits) + ".")
    if face_bits:
        sentences.append("On its face, it has " + " and ".join(face_bits) + ".")

    if not sentences:
        sentences.append("This is a simple robot.")

    if difficulty == "medium":
        if len(sentences) > 1:
            idx = list(range(len(sentences)))
            rng.shuffle(idx)
            sentences = [sentences[i] for i in idx]
    elif difficulty == "hard":
        rng.shuffle(sentences)
        if rng.random() < 0.5 and len(sentences) >= 2:
            first = sentences[0][:-1]
            second = sentences[1].lstrip()
            sentences = [first + ", and " + second[0].lower() + second[1:]]

    text = " ".join(sentences)
    text = " ".join(text.split())
    if wrinkles:
        text = text + " " + str(wrinkles).strip()
    return text.strip()


def _call_openai_chat(model: str, system_prompt: str, user_prompt: str, seed: int | None = None) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError("openai package is required for LLM-based generation/audit") from e
    client = OpenAI()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if seed is not None:
        kwargs["seed"] = int(seed)
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def _build_llm_generation_prompt(
    row: Dict[str, str],
    label: str,
    mask: Set[str],
    omit: Set[str],
    spurious: List[str],
    base_prompt: str | None,
    wrinkles: str | None,
) -> tuple[str, str]:
    system = base_prompt or "You describe simple robots from attribute lists."
    payload = {
        "label": label,
        "attributes": row,
        "masked": sorted(mask),
        "omitted": sorted(omit),
        "spurious": sorted(spurious),
    }
    user_parts = [
        "You are given attributes of a robot as JSON.",
        "Write a single English sentence describing the robot.",
        "Use natural language, not attribute names.",
        "For attributes in 'masked', describe them only generically without revealing their exact value.",
        "For attributes in 'omitted', do not include any information about them.",
        "All other attributes should be reflected in the sentence.",
        "Do not add extra attributes that are not implied by the JSON.",
        "",
        "JSON:",
        json.dumps(payload, sort_keys=True),
    ]
    if wrinkles:
        user_parts.append("")
        user_parts.append("Extra style instructions:")
        user_parts.append(str(wrinkles).strip())
    user = "\n".join(user_parts)
    return system, user


def _generate_llm_sentence(
    row: Dict[str, str],
    label: str,
    difficulty: str,
    mask: Set[str],
    omit: Set[str],
    spurious: List[str],
    llm_model: str,
    llm_system_prompt: str | None,
    llm_user_prompt: str | None,
    wrinkles: str | None,
    seed: int | None = None,
) -> str:
    base_system, default_user = _build_llm_generation_prompt(row, label, mask, omit, spurious, llm_system_prompt, wrinkles)
    if llm_user_prompt:
        user = llm_user_prompt.format(
            text="",
            concepts=json.dumps(row, sort_keys=True),
            label=label,
            masked=", ".join(sorted(mask)) or "none",
            omitted=", ".join(sorted(omit)) or "none",
            difficulty=difficulty,
        )
    else:
        user = default_user
    return _call_openai_chat(llm_model, base_system, user, seed)


def generate_text_dataset(
    *,
    concepts_spec: str | None,
    difficulty: str,
    mode: str,
    samples_per_instance: int,
    use_llm: bool,
    llm_provider: str | None,
    llm_model: str | None,
    llm_system_prompt: str | None,
    llm_user_prompt: str | None,
    wrinkles: str | None,
    spurious_concepts: List[str],
    subconcepts: List[str],
    correlations: List[CorrelationSpec],
    shuffle_concepts: bool,
    seed: int,
    mask_concepts: List[str],
    omit_concepts: List[str],
    mask_tolerance: float,
    max_instances: int | None = None,
) -> TextConceptDataset:
    if mask_tolerance < 0.0:
        raise SystemExit(f"mask_tolerance must be >= 0, got {mask_tolerance}")

    concepts = _parse_concepts_spec(concepts_spec)
    df = _build_catalog(concepts)
    df = _canonicalize_bools(df)
    df["label"] = _apply_label_expr(df, DEFAULT_LABEL_EXPR)
    if correlations:
        df = _apply_correlations(df, correlations, "label")

    rng = np.random.default_rng(seed)
    if max_instances is not None and max_instances > 0 and len(df) > max_instances:
        idx = rng.choice(len(df), size=int(max_instances), replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    concept_names = list(concepts.keys())
    concept_values: Dict[str, List[str]] = {k: sorted([str(x) for x in v]) for k, v in concepts.items()}
    value_to_index: Dict[str, Dict[str, int]] = {
        k: {v: i for i, v in enumerate(vals)} for k, vals in concept_values.items()
    }

    label_names = ["drent", "glorp"]
    label_to_index = {name: i for i, name in enumerate(label_names)}
    if not set(df["label"].unique()).issubset(set(label_names)):
        raise SystemExit(f"Unexpected labels in data: {sorted(df['label'].unique())}")

    mask_spec = _parse_mask_spec(mask_concepts)
    for name in mask_spec.keys():
        if name not in concept_names:
            raise SystemExit(f"Requested mask for unknown concept '{name}'")

    omit_set = set(omit_concepts)
    texts: List[str] = []
    C_rows: List[List[int]] = []
    y_list: List[int] = []

    use_llm_flag = use_llm or (mode == "llm")
    if use_llm_flag:
        if llm_provider and llm_provider.lower() != "openai":
            raise SystemExit("Only 'openai' provider is supported for LLM-based generation in this script.")
        if not llm_model:
            raise SystemExit("llm_model must be set when using LLM-based generation.")

    total_instances = len(df)
    total_samples = total_instances * int(samples_per_instance)

    if total_samples == 0:
        if any(frac > mask_tolerance for frac in mask_spec.values()):
            raise SystemExit("No samples generated; cannot satisfy non-zero mask fractions.")
        mask_flags = np.zeros((0, len(concept_names)), dtype=bool)
    else:
        mask_flags = np.zeros((total_samples, len(concept_names)), dtype=bool)
        for cname, frac in mask_spec.items():
            if frac <= 0.0:
                continue
            target_real = float(frac) * float(total_samples)
            floor_c = int(np.floor(target_real))
            ceil_c = int(np.ceil(target_real))
            candidates = [floor_c, ceil_c]
            best_c = min(candidates, key=lambda c: abs((c / float(total_samples)) - float(frac)))
            best_ratio = best_c / float(total_samples)
            if abs(best_ratio - float(frac)) > float(mask_tolerance):
                raise SystemExit(
                    f"Mask fraction {frac} for concept '{cname}' not attainable with tolerance {mask_tolerance}; "
                    f"closest achievable ratio is {best_ratio:.4f} with {best_c} / {total_samples} samples."
                )
            if best_c > 0:
                indices = rng.choice(total_samples, size=best_c, replace=False)
                j = concept_names.index(cname)
                mask_flags[indices, j] = True

    sample_idx = 0
    for i, sr in df.iterrows():
        row_dict: Dict[str, str] = {k: str(sr[k]) for k in concept_names}
        label_str = str(sr["label"])
        label_idx = label_to_index[label_str]
        base_vec = [value_to_index[name][row_dict[name]] for name in concept_names]
        for v in range(int(samples_per_instance)):
            if total_samples > 0:
                masked_now = {
                    concept_names[j]
                    for j in range(len(concept_names))
                    if bool(mask_flags[sample_idx, j])
                }
            else:
                masked_now = set()
            if use_llm_flag:
                text = _generate_llm_sentence(
                    row=row_dict,
                    label=label_str,
                    difficulty=difficulty,
                    mask=masked_now,
                    omit=omit_set,
                    spurious=spurious_concepts,
                    llm_model=llm_model,
                    llm_system_prompt=llm_system_prompt,
                    llm_user_prompt=llm_user_prompt,
                    wrinkles=wrinkles,
                    seed=seed + i * samples_per_instance + v,
                )
            else:
                text = _generate_template_sentence(
                    row=row_dict,
                    difficulty=difficulty,
                    mask=masked_now,
                    omit=omit_set,
                    rng=rng,
                    wrinkles=wrinkles,
                )
            texts.append(text)
            C_rows.append(list(base_vec))
            y_list.append(label_idx)
            sample_idx += 1

    C = np.asarray(C_rows, dtype=np.int64)
    y = np.asarray(y_list, dtype=np.int64)

    if shuffle_concepts and len(concept_names) > 1:
        perm = rng.permutation(len(concept_names))
        C = C[:, perm]
        concept_names = [concept_names[i] for i in perm]
        concept_values = {name: concept_values[name] for name in concept_names}
        mask_flags = mask_flags[:, perm] if mask_flags.shape[0] > 0 else mask_flags

    meta: Dict[str, Any] = {
        "concepts": concept_names,
        "concept_values": concept_values,
        "label_names": label_names,
        "difficulty": difficulty,
        "mode": "llm" if use_llm_flag else "template",
        "mask_concepts": sorted(mask_spec.keys()),
        "mask_spec": {k: float(v) for k, v in mask_spec.items()},
        "mask_tolerance": float(mask_tolerance),
        "mask_flags": mask_flags,
        "omit_concepts": sorted(omit_set),
        "spurious_concepts": list(spurious_concepts),
        "subconcepts": list(subconcepts),
        "correlations": [
            {"concept": c.concept, "value": c.value, "label": c.label, "prob": c.prob}
            for c in correlations
        ],
        "seed": int(seed),
    }

    return TextConceptDataset(
        texts=texts,
        C=C,
        y=y,
        concept_names=concept_names,
        label_names=label_names,
        meta=meta,
    )


def dataset_to_dataframe(ds: TextConceptDataset) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "text": [str(t) for t in ds.texts],
            "label": [ds.label_names[int(i)] for i in ds.y],
        }
    )
    concept_values = ds.meta.get("concept_values", {})
    for j, name in enumerate(ds.concept_names):
        vals = concept_values.get(name)
        if vals is None:
            df[name] = ds.C[:, j]
        else:
            df[name] = [vals[int(ix)] for ix in ds.C[:, j]]
    return df


def _build_default_audit_system_prompt() -> str:
    return (
        "You are an automated checker for robot descriptions. "
        "You are given a description sentence and the true concept values of the robot. "
        "Check that the description matches the attributes (except for masked/omitted ones) "
        "and that the sentence is grammatical and coherent. "
        "Respond with a single JSON object of the form "
        "{\\\"pass\\\": true/false, \\\"reason\\\": \\\"<short explanation>\\\"}. "
        "Do not output anything else."
    )


def _build_default_audit_user_prompt(
    text: str,
    concepts: Dict[str, str],
    label: str,
    masked: Set[str],
    omitted: Set[str],
) -> str:
    payload = {
        "text": text,
        "label": label,
        "concepts": concepts,
        "masked": sorted(masked),
        "omitted": sorted(omitted),
    }
    lines = [
        "Given the following JSON describing a robot and its generated description, decide if the description is acceptable.",
        "Rules:",
        "- Concepts in 'masked' may only be mentioned generically.",
        "- Concepts in 'omitted' may be missing from the text.",
        "- All other concepts should be reflected correctly.",
        "",
        "JSON:",
        json.dumps(payload, sort_keys=True),
    ]
    return "\n".join(lines)


def _render_audit_user_prompt(
    template: str | None,
    text: str,
    concepts: Dict[str, str],
    label: str,
    masked: Set[str],
    omitted: Set[str],
) -> str:
    if not template:
        return _build_default_audit_user_prompt(text, concepts, label, masked, omitted)
    return template.format(
        text=text,
        concepts=json.dumps(concepts, sort_keys=True),
        label=label,
        masked=", ".join(sorted(masked)) or "none",
        omitted=", ".join(sorted(omitted)) or "none",
    )


def run_audit(
    ds: TextConceptDataset,
    *,
    every: int,
    max_samples: int,
    model: str,
    system_prompt: str | None,
    user_prompt: str | None,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    if every <= 0:
        return {"audited": 0, "failures": 0, "failure_rate": 0.0}
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("Audit requested but openai package is not installed; skipping audit.")
        return {"audited": 0, "failures": 0, "failure_rate": 0.0}

    client = OpenAI()
    concept_values = ds.meta.get("concept_values", {})
    label_names = ds.label_names
    concept_names = ds.concept_names
    mask_flags = ds.meta.get("mask_flags", None)
    mask_spec_meta = ds.meta.get("mask_spec", {})
    mask_names_meta = ds.meta.get("mask_concepts", list(mask_spec_meta.keys()))
    omit_names_meta = ds.meta.get("omit_concepts", [])

    audited = 0
    failures = 0
    records: List[Dict[str, Any]] = []

    n = len(ds.texts)
    for i in range(n):
        if i % every != 0:
            continue
        if max_samples > 0 and audited >= max_samples:
            break

        text = str(ds.texts[i])
        label_idx = int(ds.y[i])
        label = label_names[label_idx]
        concepts: Dict[str, str] = {}
        for j, name in enumerate(concept_names):
            vals = concept_values.get(name)
            if vals is None:
                concepts[name] = str(ds.C[i, j])
            else:
                concepts[name] = str(vals[int(ds.C[i, j])])

        if mask_flags is not None and mask_flags.shape[0] == n:
            masked = {
                concept_names[j]
                for j in range(len(concept_names))
                if bool(mask_flags[i, j])
            }
        else:
            masked = set(mask_names_meta)

        omitted = set(omit_names_meta)

        sys_prompt = system_prompt or _build_default_audit_system_prompt()
        usr_prompt = _render_audit_user_prompt(user_prompt, text, concepts, label, masked, omitted)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        ok = False
        reason = ""
        try:
            obj = json.loads(raw)
            ok = bool(obj.get("pass", False))
            reason = str(obj.get("reason", ""))
        except Exception:
            ok = False
            reason = f"Bad JSON from model: {raw[:200]}"
        audited += 1
        if not ok:
            failures += 1
        records.append(
            {
                "index": i,
                "text": text,
                "label": label,
                "concepts": concepts,
                "pass": bool(ok),
                "reason": reason,
            }
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    failure_rate = float(failures) / float(audited) if audited > 0 else 0.0
    print(f"Audit: {failures}/{audited} failed ({failure_rate * 100.0:.1f}%).")
    return {"audited": audited, "failures": failures, "failure_rate": failure_rate}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate robot text samples")
    ap.add_argument("--concepts-spec", type=str, default=settings["concepts_spec"])
    ap.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard"], default=settings["difficulty"])
    ap.add_argument("--mode", type=str, choices=["template", "llm"], default=settings["mode"])
    ap.add_argument("--samples-per-instance", type=int, default=settings["samples_per_instance"])
    ap.add_argument("--use-llm", type=int, default=settings["use_llm"])
    ap.add_argument("--llm-provider", type=str, default=settings["llm_provider"])
    ap.add_argument("--llm-model", type=str, default=settings["llm_model"])
    ap.add_argument("--llm-system-prompt", type=str, default=settings["llm_system_prompt"])
    ap.add_argument("--llm-user-prompt", type=str, default=settings["llm_user_prompt"])
    ap.add_argument("--wrinkles", type=str, default=settings["wrinkles"])
    ap.add_argument("--spurious-concepts", type=str, default=settings["spurious_concepts"])
    ap.add_argument("--subconcepts", type=str, default=settings["subconcepts"])
    ap.add_argument("--correlations", type=str, default=settings["correlations"])
    ap.add_argument("--shuffle-concepts", type=int, default=settings["shuffle_concepts"])
    ap.add_argument("--seed", type=int, default=settings["seed"])
    ap.add_argument("--output-prefix", type=str, default=settings["output_prefix"])
    ap.add_argument("--mask-concepts", type=str, default=settings["mask_concepts"])
    ap.add_argument("--omit-concepts", type=str, default=settings["omit_concepts"])
    ap.add_argument("--mask-tolerance", type=float, default=settings["mask_tolerance"])
    ap.add_argument("--max-instances", type=int, default=settings["max_instances"])
    ap.add_argument("--audit-every", type=int, default=settings["audit_every"])
    ap.add_argument("--audit-max-samples", type=int, default=settings["audit_max_samples"])
    ap.add_argument("--audit-model", type=str, default=settings["audit_model"])
    ap.add_argument("--audit-system-prompt", type=str, default=settings["audit_system_prompt"])
    ap.add_argument("--audit-user-prompt", type=str, default=settings["audit_user_prompt"])
    ap.add_argument("--audit-output", type=str, default=settings["audit_output"])
    return ap


def run_from_settings(cfg: Dict[str, Any]) -> None:
    spurious = _parse_list_arg(str(cfg["spurious_concepts"]))
    subconcepts = _parse_list_arg(str(cfg["subconcepts"]))
    corr_specs = _parse_correlations(str(cfg["correlations"]))
    mask_concepts = _parse_list_arg(str(cfg["mask_concepts"]))
    omit_concepts = _parse_list_arg(str(cfg["omit_concepts"]))
    mask_tolerance = float(cfg["mask_tolerance"])
    max_instances = int(cfg["max_instances"]) if int(cfg["max_instances"]) > 0 else None

    ds = generate_text_dataset(
        concepts_spec=str(cfg["concepts_spec"]) or None,
        difficulty=str(cfg["difficulty"]),
        mode=str(cfg["mode"]),
        samples_per_instance=int(cfg["samples_per_instance"]),
        use_llm=bool(int(cfg["use_llm"])),
        llm_provider=str(cfg["llm_provider"]) or None,
        llm_model=str(cfg["llm_model"]) or None,
        llm_system_prompt=str(cfg["llm_system_prompt"]) or None,
        llm_user_prompt=str(cfg["llm_user_prompt"]) or None,
        wrinkles=str(cfg["wrinkles"]) or None,
        spurious_concepts=spurious,
        subconcepts=subconcepts,
        correlations=corr_specs,
        shuffle_concepts=bool(int(cfg["shuffle_concepts"])),
        seed=int(cfg["seed"]),
        mask_concepts=mask_concepts,
        omit_concepts=omit_concepts,
        mask_tolerance=mask_tolerance,
        max_instances=max_instances,
    )

    cfg_for_hash = {
        "concepts_spec": cfg["concepts_spec"],
        "difficulty": cfg["difficulty"],
        "mode": cfg["mode"],
        "samples_per_instance": int(cfg["samples_per_instance"]),
        "use_llm": bool(int(cfg["use_llm"])),
        "llm_provider": cfg["llm_provider"],
        "llm_model": cfg["llm_model"],
        "wrinkles": cfg["wrinkles"],
        "spurious_concepts": spurious,
        "subconcepts": subconcepts,
        "correlations": cfg["correlations"],
        "shuffle_concepts": bool(int(cfg["shuffle_concepts"])),
        "seed": int(cfg["seed"]),
        "mask_concepts": mask_concepts,
        "omit_concepts": omit_concepts,
        "mask_tolerance": mask_tolerance,
        "max_instances": max_instances,
    }
    suffix = hashlib.blake2b(json.dumps(cfg_for_hash, sort_keys=True).encode(), digest_size=8).hexdigest()
    out_dir = Path("results") / "robot_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{cfg['output_prefix']}_{suffix}.pkl"
    csv_path = out_dir / f"{cfg['output_prefix']}_{suffix}.csv"

    import pickle

    with pkl_path.open("wb") as f:
        pickle.dump(ds, f)

    df = dataset_to_dataframe(ds)
    df.to_csv(csv_path, index=False)

    print("Saved dataset:", pkl_path)
    print("Saved text+concepts CSV:", csv_path)

    audit_every = int(cfg["audit_every"])
    if audit_every > 0:
        audit_model = str(cfg["audit_model"]) or str(cfg["llm_model"])
        if not audit_model:
            raise SystemExit("Audit requested but no audit-model or llm-model specified.")
        audit_output = str(cfg["audit_output"]) or str(out_dir / f"{cfg['output_prefix']}_{suffix}_audit.jsonl")
        run_audit(
            ds,
            every=audit_every,
            max_samples=int(cfg["audit_max_samples"]),
            model=audit_model,
            system_prompt=str(cfg["audit_system_prompt"]) or None,
            user_prompt=str(cfg["audit_user_prompt"]) or None,
            output_path=Path(audit_output),
        )


def cli(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args(argv)
    cfg = dict(settings)
    cfg.update(vars(args))
    run_from_settings(cfg)
