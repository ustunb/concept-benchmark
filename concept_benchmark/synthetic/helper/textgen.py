from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
from typing import Iterable

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
