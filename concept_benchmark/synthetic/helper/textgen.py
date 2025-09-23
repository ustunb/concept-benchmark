from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
from typing import Sequence, Iterable
from concept_benchmark.data import ConceptDataset

_MOD_RE = re.compile(r"\{([a-zA-Z0-9_]+)~(not|syn)\}")

def _rewrite_modifiers(tpl: str) -> str:
    return _MOD_RE.sub(lambda m: "{" + m.group(1) + "_" + m.group(2) + "}", tpl)

def _clean_unknown(text: str) -> str:
    parts = re.split(r'([,;.:])', text)
    kept = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if "unknown" in part.lower():
                continue
            kept.append(part.strip())
        else:
            if kept and kept[-1]:
                kept[-1] = kept[-1].rstrip()
                kept.append(part)
    s = " ".join(kept)
    s = s.replace(";", ".")
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.:])", r"\1", s)
    s = re.sub(r"([,.:])(\w)", r"\1 \2", s)
    s = re.sub(r"\s*\.(\s*\.)+", ".", s)
    return s.strip()

def _polish_text(s: str) -> str:
    s = re.sub(r"\bhas with (\w+)\s+\1\b", r"has \1", s, flags=re.I)
    s = re.sub(r"\bhas no (\w+)\s+\1\b", r"has no \1", s, flags=re.I)
    s = re.sub(r"\b([Ee]lbows|[Kk]nees|[Aa]ntennae)\s+with\s+\1\b", r"With \1", s)
    s = re.sub(r"\b([Ee]lbows|[Kk]nees|[Aa]ntennae)\s+no\s+\1\b", r"No \1", s)

    s = re.sub(r"\b[Nn]ot\s+([a-z0-9 _-]+)\s+at the head\b", r"The head isn’t \1", s)
    s = re.sub(r"\b[Nn]ot\s+([a-z0-9 _-]+)\s+for the body\b", r"The body isn’t \1", s)

    s = re.sub(r"\b[Ee]ars (are|:)?\s*triangle\b", r"Ears are triangular", s)
    s = re.sub(r"\bhand[s]?\s+edgy\s+square\b", r"hands are square-edged", s, flags=re.I)
    s = re.sub(r"\b[Ff]eet\s+pointy\s+4sided\b", r"feet are pointed four-sided", s)

    s = re.sub(r"\b[Cc]olor\s*[:—–-]?\s+(?!is|:)([a-z]+)\b", r"color is \1", s)
    s = re.sub(r"\b[Mm]outh\s*[:—–-]?\s*(open|closed)\b", r"mouth is \1", s, flags=re.I)
    s = re.sub(r"\b[Cc]olor\s+is\s+([a-z]+)/([a-z]+)\b", r"color is \1 and \2", s)

    def _fix_colon_has_no(m):
        item = m.group(1).lower()
        aux = m.group(2).lower()
        return f"{'Has' if aux=='has' else 'No'} {item}."
    s = re.sub(r"\b(Knees|Elbows|Antennae):\s*(has|no)\b\.?", _fix_colon_has_no, s, flags=re.I)
    s = re.sub(r"\b(Knees|Elbows|Antennae):\s*has\s+(knees|elbows|antennae)\b\.?", lambda m: f"Has {m.group(2)}.", s, flags=re.I)
    s = re.sub(r"\b(Knees|Elbows|Antennae):\s*no\s+(knees|elbows|antennae)\b\.?",  lambda m: f"No {m.group(2)}.",  s, flags=re.I)

    s = re.sub(r"\b(Elbows|Knees|Antennae)\s+no\s+\1\b",  lambda m: f"No {m.group(1).lower()}",  s, flags=re.I)
    s = re.sub(r"\b(Elbows|Knees|Antennae)\s+has\s+\1\b", lambda m: f"Has {m.group(1).lower()}", s, flags=re.I)
    s = re.sub(r"\b([Ee]lbows|[Kk]nees|[Aa]ntennae)\s+are\s+(has|no)\s+\1\b",
               lambda m: ("Has" if m.group(2).lower()=="has" else "No") + " " + m.group(1).lower(), s)

    s = re.sub(r"\b[Aa]ntennae\s*[:—–-]?\s*no\s+antennae\b\.?", "No antennae", s)
    s = re.sub(r"\b[Aa]ntennae\s*[:—–-]?\s*has\s+antennae\b\.?", "Has antennae", s)

    s = re.sub(r"\b(joints|hinges|articulation|limbs)\s+(has|no)\s+elbows/(has|no)\s+knees\b\.?",
               lambda m: f"{'Has' if m.group(2).lower()=='has' else 'No'} elbows; "
                         f"{'has' if m.group(3).lower()=='has' else 'no'} knees", s, flags=re.I)
    s = re.sub(r"\bno elbows/no knees\b", "no elbows and no knees", s, flags=re.I)
    s = re.sub(r"\bhas elbows/has knees\b", "has elbows and knees", s, flags=re.I)
    s = re.sub(r"\bhas elbows/no knees\b", "has elbows; no knees", s, flags=re.I)
    s = re.sub(r"\bno elbows/has knees\b", "no elbows; has knees", s, flags=re.I)

    s = re.sub(r"\bthen[- ]sharp[- ]cornered\b", "sharp-cornered", s, flags=re.I)
    s = re.sub(r"\bsharp\s+cornered\b", "sharp-cornered", s, flags=re.I)
    s = re.sub(r"\bboxy\s+(head|body)\b", r"square \1", s, flags=re.I)
    s = re.sub(r"\b(head|body)\s+boxy\b", r"\1 square", s, flags=re.I)

    s = re.sub(r"\b(is|are|was|were)\s+not\s+not\s+([a-z0-9 _-]+)", lambda m: f"{m.group(1)} {m.group(2)}", s, flags=re.I)
    s = re.sub(r"\b(is|are|was|were)n['’]t not\s+([a-z0-9 _-]+)", lambda m: f"{m.group(1)} {m.group(2)}", s, flags=re.I)
    s = re.sub(r"\bnot\s+not\s+([a-z0-9 _-]+)", r"\1", s, flags=re.I)

    s = re.sub(r"\bhas\s+has\s+(\w+)\s+\1\b", r"has \1", s, flags=re.I)
    s = re.sub(r"\bdoes\s+have\s+has\s+(\w+)\s+\1\b", r"does have \1", s, flags=re.I)
    s = re.sub(r"\b(has|does\s+have)\s+(has|no)\s+(\w+)\s+\3\b",
               lambda m: f"{m.group(1)} {'no ' if m.group(2).lower()=='no' else ''}{m.group(3)}", s, flags=re.I)

    sentences = [w.strip() for w in re.split(r"(?<=[.?!])\s+", s) if w.strip()]
    sentences = [sent[0].upper() + sent[1:] if sent and sent[0].islower() else sent for sent in sentences]
    s = " ".join(sentences)
    s = re.sub(r"\s+([,.:])", r"\1", s)
    s = re.sub(r"([,.:])(\w)", r"\1 \2", s)
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s, flags=re.I)
    s = re.sub(r"\b(Knees|Elbows|Antennae)\.\s+\1\.", r"\1.", s)
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."
    s = re.sub(r"([,.:;!?])\s*([,.:;!?])+", r"\2", s)

    s = re.sub(r"\bNo knees\.\s+Knees\.", "No knees.", s, flags=re.I)
    s = re.sub(r"\bKnees\.\s+No knees\.", "No knees.", s, flags=re.I)
    s = re.sub(r"\bNo elbows\.\s+Elbows\.", "No elbows.", s, flags=re.I)
    s = re.sub(r"\bElbows\.\s+No elbows\.", "No elbows.", s, flags=re.I)
    s = re.sub(r"\bNo antennae\.\s+Antennae\.", "No antennae.", s, flags=re.I)
    s = re.sub(r"\bAntennae\.\s+No antennae\.", "No antennae.", s, flags=re.I)
    s = re.sub(r"(?<!No )\b(Knees|Elbows|Antennae)\.(\s+|$)", "", s, flags=re.I)

    s = re.sub(r"\bKnees are no knees\b\.?", "No knees.", s, flags=re.I)
    s = re.sub(r"\bElbows are no elbows\b\.?", "No elbows.", s, flags=re.I)
    s = re.sub(r"\bAntennae are no antennae\b\.?", "No antennae.", s, flags=re.I)

    s = re.sub(
        r"\b(?:elbows\s*/\s*knees|knees\s*/\s*elbows)\s+(no|has)\s+elbows\s*/\s+(no|has)\s+knees\b",
        lambda m: f"{m.group(1).lower()} elbows; {m.group(2).lower()} knees",
        s, flags=re.I)
    s = re.sub(
        r"\bantennae\s+(?:is|are|read|reads)\s+(no|has)\s+antennae\b",
        lambda m: "No antennae" if m.group(1).lower() == "no" else "Has antennae",
        s, flags=re.I)
    return s


def _synonym(name: str, val: str) -> str:
    v = str(val).replace("_", " ").lower()
    if name == "hand_shape":
        v = re.sub(r"\bround\s+", "", v)
        v = re.sub(r"\bedgy\s+", "", v)
        if v in {"oval2", "oval 2"}:
            return "tall oval"
        if v == "wide oval":
            return "wide oval"
        return v
    if name == "foot_shape":
        num = {"3": "three", "4": "four", "5": "five", "6": "six"}
        v = v.replace("pointy", "pointed")
        v = v.replace("lshaped", "l-shaped").replace("l shaped", "l-shaped")
        v = re.sub(r"\bflat\s*(\d)\s*sided\b", lambda m: f"{num.get(m.group(1), '?')}-sided, not pointy", v)
        v = re.sub(r"\bpointed?\s*(\d)\s*sided\b", lambda m: f"{num.get(m.group(1), '?')}-sided, pointed", v)
        v = re.sub(r"\b(\d)\s*sided\b", lambda m: f"{num.get(m.group(1), m.group(1))}-sided", v)
        return v
    return v

def _negate(name: str, val):
    if name in ("head_shape", "body_shape"):
        v = str(val).lower()
        if v == "square":
            return "round"
        if v == "round":
            return "square"
        return None
    if name in ("has_knees", "has_elbows", "has_antennae"):
        b = (str(val).lower() in {"1","true","t","yes","y"}) if isinstance(val, str) else bool(val)
        return not b
    return None

class _Safe(dict):
    def __missing__(self, k):
        return "unknown"

DEFAULT_TEMPLATES = [
    "A {color_mode} robot with a {head_shape} head and a {body_shape} body, {has_elbows}, {has_knees}, and {foot_shape} feet.",
    "{color_mode} robot: {body_shape} torso, {head_shape} head; {has_elbows}; {has_knees}; feet are {foot_shape}.",
    "Robot ({color_mode}): head={head_shape}, body={body_shape}, elbows={has_elbows_bool}, knees={has_knees_bool}, feet={foot_shape}.",
]

def create_synthetic_dataset(source, templates: Sequence[str] | None = None, variants_per_row: int = 3, include_color: bool = True, rng_seed: int = 0, head_col: str = "head_shape", body_col: str = "body_shape", knees_col: str = "has_knees", elbows_col: str = "has_elbows", foot_col: str = "foot_shape", color_mode_col: str = "color_mode", concept_cols: Iterable[str] | None = None, label_col: str | None = None, label_map: dict | None = None, drop_unknown: bool = True, text_mode: str | None = None, use_llm: bool = False, llm_provider: str = "gemini", llm_model: str = "gemini-1.5-flash", llm_api_key: str | None = None, llm_system: str | None = None, llm_user_prompt: str | None = None) -> ConceptDataset:
    return create_robot_text_dataset(source=source, templates=templates, variants_per_row=variants_per_row, include_color=include_color, rng_seed=rng_seed, head_col=head_col, body_col=body_col, knees_col=knees_col, elbows_col=elbows_col, foot_col=foot_col, color_mode_col=color_mode_col, concept_cols=concept_cols, label_col=label_col, label_map=label_map, drop_unknown=drop_unknown, text_mode=text_mode, use_llm=use_llm, llm_provider=llm_provider, llm_model=llm_model, llm_api_key=llm_api_key, llm_system=llm_system, llm_user_prompt=llm_user_prompt)

def create_robot_text_dataset(source, templates: Sequence[str] | None = None, variants_per_row: int = 3, include_color: bool = True, rng_seed: int = 0, head_col: str = "head_shape", body_col: str = "body_shape", knees_col: str = "has_knees", elbows_col: str = "has_elbows", foot_col: str = "foot_shape", color_mode_col: str = "color_mode", concept_cols: Iterable[str] | None = None, label_col: str | None = None, label_map: dict | None = None, drop_unknown: bool = True, text_mode: str | None = None, use_llm: bool = False, llm_provider: str = "gemini", llm_model: str = "gemini-1.5-flash", llm_api_key: str | None = None, llm_system: str | None = None, llm_user_prompt: str | None = None) -> ConceptDataset:
    if templates is None or text_mode == "structured":
        templates = DEFAULT_TEMPLATES
    templates = [re.sub(r'^[\uFEFF\u200B-\u200D]+', '', t) for t in templates]
    templates = [_rewrite_modifiers(t) for t in templates]
    rng = np.random.default_rng(rng_seed)
    mode = (text_mode or ("llm" if use_llm else "unstructured")).strip().lower()

    if isinstance(source, ConceptDataset):
        df = getattr(source, "catalog_df", None)
        if df is None:
            raise ValueError("ConceptDataset missing catalog_df")
        C = np.asarray(source.C)
        y = np.asarray(source.y)
        concept_names = list(source.meta.get("concepts", [f"c{i}" for i in range(C.shape[1])]))
        classes = list(source.meta.get("classes", sorted(map(int, np.unique(y)))))
    else:
        if not isinstance(source, pd.DataFrame):
            raise TypeError("source must be ConceptDataset or pd.DataFrame")
        if concept_cols is None or label_col is None:
            raise ValueError("Provide concept_cols and label_col when source is a DataFrame")
        df = source
        C, concept_names = _binarize_concepts(df, concept_cols)
        y = _to_label(df[label_col].to_numpy(), label_map)
        if label_map is not None:
            classes = [str(k) for k, v in sorted(label_map.items(), key=lambda kv: kv[1])]
        else:
            raw_uniqs = pd.Series(df[label_col]).astype(str).str.lower().unique().tolist()
            classes = ["drent","glorp"] if {"glorp","drent"} <= set(raw_uniqs) else [str(v) for v in sorted(np.unique(y).tolist())]

    X, idxs = [], []
    tbool = lambda v: (v.lower() in {"true","t","yes","y","1"}) if isinstance(v,str) else bool(v)
    colorish = lambda d: ("color" if any(c in d.columns for c in ("color","left_color","right_color","primary_color","secondary_color")) else "greyscale")

    def _colors_for_row(r):
        if "left_color" in df.columns and "right_color" in df.columns:
            return str(r["left_color"]), str(r["right_color"])
        if "primary_color" in df.columns and "secondary_color" in df.columns:
            return str(r["primary_color"]), str(r["secondary_color"])
        if "color1" in df.columns and "color2" in df.columns:
            return str(r["color1"]), str(r["color2"])
        return None, None

    structured_templates_default = [
        "This robot has a {head_shape} head and a {body_shape} body. It {has_elbows} and {has_knees}. Its feet are {foot_shape}.",
        "Head: {head_shape}. Body: {body_shape}. Elbows: {has_elbows}. Knees: {has_knees}. Feet: {foot_shape}.",
    ]

    for i, row in df.iterrows():
        cms = (row.get(color_mode_col, None) if (include_color and color_mode_col in df.columns) else (colorish(df) if include_color else "greyscale"))
        knees_b  = tbool(row.get(knees_col, False))
        elbows_b = tbool(row.get(elbows_col, False))
        ant_b    = tbool(row.get("has_antennae", False))
        c1, c2 = _colors_for_row(row)
        if c1 and c2:
            color_pair = f"{c1} and {c2}"
            color_single = f"{c1}/{c2}"
        else:
            color_pair = str(row.get("color")) if "color" in df.columns else None
            color_single = color_pair
        fill = {
            "head_shape": str(row.get(head_col, "")).replace("_", " "),
            "body_shape": str(row.get(body_col, "")).replace("_", " "),
            "foot_shape": str(row.get(foot_col, "")).replace("_", " "),
            "ears_shape": str(row.get("ears_shape", "")).replace("_", " "),
            "mouth_type": str(row.get("mouth_type", "")).replace("_", " "),
            "hand_shape": str(row.get("hand_shape", "")).replace("_", " "),
            "has_knees": "has knees" if knees_b else "no knees",
            "has_elbows": "has elbows" if elbows_b else "no elbows",
            "has_antennae": "has antennae" if ant_b else "no antennae",
            "has_knees_word": "has" if knees_b else "no",
            "has_elbows_word": "has" if elbows_b else "no",
            "has_antennae_word": "has" if ant_b else "no",
            "has_knees_bool": "true" if knees_b else "false",
            "has_elbows_bool": "true" if elbows_b else "false",
            "has_antennae_bool": "true" if ant_b else "false",
            "color": color_single if color_single else "unknown",
            "color_pair": color_pair if color_pair else "unknown",
            "color_left": c1 if c1 else "unknown",
            "color_right": c2 if c2 else "unknown",
            "color_mode": "greyscale" if not include_color else str(cms),
        }
        fill["hand_shape"] = _synonym("hand_shape", fill["hand_shape"])
        fill["foot_shape"] = _synonym("foot_shape", fill["foot_shape"])
        for name in ("head_shape","body_shape","ears_shape","mouth_type","hand_shape","foot_shape"):
            val = fill.get(name, "")
            fill[name + "_syn"] = _synonym(name, val)
            neg = _negate(name, val)
            if neg is not None:
                fill[name + "_not"] = str(neg if isinstance(neg, str) else neg).replace("_"," ")
            fill[name + "_not"] = (str(neg).replace("_", " ") if neg is not None else f"not {val}")
        fill["has_knees_not"]     = "no knees" if knees_b else "with knees"
        fill["has_elbows_not"]    = "no elbows" if elbows_b else "with elbows"
        fill["has_antennae_not"]  = "no antennae" if ant_b else "with antennae"
        sfill = _Safe(fill)

        if mode == "llm":
            if concept_cols is None:
                concept_keys = [head_col, body_col, foot_col, knees_col, elbows_col, "has_antennae", "mouth_type", "ears_shape", "hand_shape", "color"]
                concept_keys = [k for k in concept_keys if k in df.columns]
            else:
                concept_keys = list(concept_cols)
            concepts_dict = {k: (str(row[k]) if k in row else "") for k in concept_keys}
            llm_texts = unstructured_caption_via_llm(concepts=concepts_dict, provider=llm_provider, model=llm_model, api_key=llm_api_key, system=llm_system, user_prompt=llm_user_prompt, n=variants_per_row)
            for out in llm_texts:
                out = _polish_text(out)
                X.append(out); idxs.append(i)
        elif mode == "structured":
            base_tpls = templates if templates else structured_templates_default
            for tpl in rng.choice(base_tpls, size=variants_per_row, replace=True):
                s = _rewrite_modifiers(tpl)
                out = s.format_map(sfill)
                if drop_unknown:
                    out = _clean_unknown(out)
                out = _polish_text(out)
                X.append(out); idxs.append(i)
        else:
            for tpl in rng.choice(templates, size=variants_per_row, replace=True):
                s = _rewrite_modifiers(tpl)
                out = s.format_map(sfill)
                if drop_unknown:
                    out = _clean_unknown(out)
                out = _polish_text(out)
                X.append(out); idxs.append(i)

    idxs = np.asarray(idxs, dtype=int)
    C_out = C[idxs]
    y_out = y[idxs]
    meta = {
        "data_type": "text",
        "templates": list(templates),
        "concepts": concept_names,
        "classes": classes,
        "row_index": idxs,
    }
    return ConceptDataset(X=list(X), C=np.asarray(C_out, dtype=np.int8), y=np.asarray(y_out, dtype=np.int32), meta=meta)

def _binarize_concepts(df: pd.DataFrame, cols: Iterable[str]):
    mats, names = [], []
    for col in cols:
        s = df[col]
        if s.dropna().isin([0,1,True,False,"0","1","true","false","t","f","yes","no","y","n"]).all():
            v = s.map(lambda x: (str(x).lower() in {"1","true","t","yes","y"}) if isinstance(x,str) else bool(x)).astype(int).to_numpy()[:,None]
            mats.append(v)
            names.append(col)
        else:
            cats = pd.Categorical(s)
            onehot = np.zeros((len(df), len(cats.categories)), dtype=int)
            for j, cat in enumerate(cats.categories):
                onehot[(cats.codes==j), j] = 1
            mats.append(onehot)
            names.extend([f"{col}={cat}" for cat in cats.categories])
    C = np.concatenate(mats, axis=1) if mats else np.zeros((len(df),0), dtype=int)
    return C, names

def _to_label(arr, label_map: dict | None):
    s = pd.Series(arr)
    if arr.dtype.kind in "iu":
        return arr.astype(int)
    if label_map is None:
        s_lower = s.astype(str).str.lower()
        uniq = set(s_lower.unique())
        if {"glorp", "drent"} <= uniq:
            m = {"glorp": 1, "drent": 0}
            return s_lower.map(m).astype(int).to_numpy()
        return s_lower.isin({"1", "true", "t", "yes", "y"}).astype(int).to_numpy()
    # label_map is provided
    return s.map(lambda v: label_map.get(v, v)).astype(int).to_numpy()

def unstructured_caption_via_llm(concepts: dict, *, provider: str = "gemini", model: str = "", api_key: str | None = None, system: str | None = None, user_prompt: str | None = None, n: int = 1, temperature: float = 0.7):
    import os
    import google.generativeai as genai
    import openai
    import anthropic
    api_key = api_key or os.getenv("LM_API_KEY")
    if not api_key:
        raise RuntimeError("Set LM_API_KEY (Insert API Key Here) and optionally LM_PROVIDER/LM_MODEL.")
    provider = os.getenv("LM_PROVIDER", provider)
    prov = provider.lower()
    model = os.getenv("LM_MODEL", model) or {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash",
    }.get(prov, "")
    default_user_prompt = ("Using the provided attributes, write a description in plain, spoken language that sounds like a person describing an image they saw. Keep it concise and natural, between 1 and 3 sentences. Avoid list-like phrasing, avoid locations/situations, and focus only on what the attributes imply.")
    uprompt = user_prompt.strip() if user_prompt else default_user_prompt
    attr_json = json.dumps(concepts, ensure_ascii=False)
    system = system or "You are concise and concrete. Use everyday language. Do not invent locations or scenarios."
    user_message = f"{uprompt}\n\nAttributes (JSON): {attr_json}"
    out = []

    if prov == "openai":
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user_message}], n=n, temperature=temperature)
        for c in resp.choices:
            out.append((c.message.content or "").strip())
        return out

    elif prov in ("anthropic", "claude"):
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(model=model, max_tokens=160, temperature=temperature, system=system, messages=[{"role": "user", "content": user_message}])
        text = "".join([b.text for b in msg.content if hasattr(b, "text")]) if getattr(msg, "content", None) else ""
        out.append(text.strip())
        for _ in range(max(0, n - 1)):
            msg = client.messages.create(model=model, max_tokens=160, temperature=temperature, system=system, messages=[{"role": "user", "content": user_message}])
            text = "".join([b.text for b in msg.content if hasattr(b, "text")]) if getattr(msg, "content", None) else ""
            out.append((text or "").strip())
        return out

    elif prov in ("gemini", "google", "googleai", "google-genai"):
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(model or "gemini-1.5-flash")
        gen_cfg = dict(temperature=temperature, max_output_tokens=160)
        for _ in range(max(1, n)):
            r = gm.generate_content(user_message, generation_config=gen_cfg)
            out.append((getattr(r, "text", "") or "").strip())
        return out
    else:
        raise RuntimeError(f"Unknown provider: {provider}")
