
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json, random, re, os

try:
    from .robots import ALL_ROBOT_FEATURES
except Exception:
    ALL_ROBOT_FEATURES = {
        "head_shape": ("round", "square"),
        "body_shape": ("round", "square"),
        "foot_shape": ("flat_4sided", "flat_5sided", "flat_lshaped", "pointy_3sided", "pointy_4sided", "pointy_6sided"),
        "ears_shape": ("pointy", "flat", "none"),
        "mouth_type": ("grille", "slot", "circle", "none"),
        "hand_shape": ("claw", "mitten", "none"),
        "has_knees": ("true", "false"),
        "has_elbows": ("true", "false"),
        "has_antennae": ("true", "false"),
        "color": ("red", "blue", "green", "yellow", "black", "white", "silver"),
    }

VOWELS = set("aeiou")

SYNONYMS = {
    "round": ["circular", "curved", "rounded"],
    "square": ["boxy", "rectilinear"],
    "pointy": ["sharp", "tapered"],
    "flat": ["level", "even"],
    "claw": ["pincer", "gripper"],
    "mitten": ["paddle", "pad-like"],
    "grille": ["vented", "slotted"],
    "slot": ["slit", "narrow opening"],
    "circle": ["disc", "ring"],
}

def _indef_article(word: str) -> str:
    return "an" if word[:1].lower() in VOWELS else "a"

def _pretty(name: str) -> str:
    return name.replace("_", " ")

def _bool_to_text(name: str, val: str) -> str:
    yes = val.lower() in ("true", "1", "yes")
    label = _pretty(name[4:]) if name.startswith("has_") else _pretty(name)
    return f"has {label}" if yes else f"does not have {label}"

def _resolve_syn(value: str) -> str:
    opts = SYNONYMS.get(value, [])
    return random.choice(opts) if opts else value

def _resolve_not(name: str, true_val: str, domains: Dict[str, List[str]]) -> str:
    dom = domains.get(name, [])
    if not dom:
        return f"not {true_val}"
    others = [v for v in dom if v != true_val]
    return random.choice(others) if others else f"not {true_val}"

@dataclass
class CaptionTemplateEngine:
    domains: Dict[str, List[str]] | None = None

    def __post_init__(self):
        if self.domains is None:
            self.domains = {k:list(v) for k,v in ALL_ROBOT_FEATURES.items()}

   
    def consistent(self, concepts: Dict[str, str], *, two_sentences: bool = True) -> str:
        items = list(concepts.items())
        if two_sentences and len(items) >= 2:
            (n1, v1), (n2, v2) = items[0], items[1]
            s1 = self._sentence(n1, v1, lead="This robot")
            s2 = self._sentence(n2, v2, lead="It")
            return f"{s1} {s2}"
        parts = [self._sentence(n, v, lead=None) for n, v in items]
        return " ".join(parts)

    def _sentence(self, name: str, value: str, lead: Optional[str]) -> str:
        if name.startswith("has_") or value in ("true", "false"):
            phrase = _bool_to_text(name, value)
            if lead: return f"{lead} {phrase}."
            return phrase.capitalize() + "."
        else:
            article = _indef_article(value)
            if lead: return f"{lead} has {article} {value} {_pretty(name)}."
            return f"Has {article} {value} {_pretty(name)}."

    
    def load_templates(self, path: Path) -> List[str]:
        txt = path.read_text(encoding="utf-8")
        raw = [line.strip() for line in txt.splitlines() if line.strip()]
        return [r for r in raw if "{" in r and "}" in r]

    def _fill_template(self, tpl: str, concepts: Dict[str, str]) -> str:
        def repl(m: re.Match) -> str:
            key = m.group(1)
            mod = m.group(2) or ""
            name = key
            if name not in concepts and name in ("color_desc",):
                name = "color"
            true = concepts.get(name, "unknown")
            if mod == "~syn":
                return _resolve_syn(true)
            if mod == "~not":
                return _resolve_not(name, true, self.domains)
            return true
        return re.sub(r"\{([a-zA-Z0-9_]+)(~syn|~not)?\}", repl, tpl)

    def inconsistent(self, concepts: Dict[str, str], *, templates_file: Optional[Path], n: int = 50) -> List[str]:
        if templates_file and templates_file.exists():
            templates = self.load_templates(templates_file)
        else:
            templates = [
                "Head {head_shape}; body {body_shape}. Knees {has_knees}; antennae {has_antennae}.",
                "It’s got a {head_shape} head on a {body_shape} body. Knees {has_knees}, elbows {has_elbows}.",
                "{color} color. {hand_shape} hands. Elbows {has_elbows}.",
                "Head isn’t {head_shape~not}; it’s {head_shape}. Body isn’t {body_shape~not}; it’s {body_shape}.",
                "Feet {foot_shape}; mouth {mouth_type}; ears {ears_shape}.",
            ]
        random.shuffle(templates)
        if not templates:
            return []
        picks = (templates * ((n // len(templates)) + 1))[:n]
        return [self._fill_template(t, concepts) for t in picks]

def unstructured_caption_via_llm(concepts: Dict[str, str], *, provider: str = "openai", model: str = "", api_key: Optional[str] = None, system: Optional[str] = None, n: int = 1) -> List[str]:
    """
    provider in {"openai", "anthropic", "gemini"} (case-insensitive).
    Requires env var LM_API_KEY unless api_key passed directly.
    Optional overrides: LM_PROVIDER, LM_MODEL.
    """
    api_key = api_key or os.getenv("LM_API_KEY")
    if not api_key:
        raise RuntimeError("Set LM_API_KEY (Insert API Key Here) and optionally LM_PROVIDER/LM_MODEL.")
    provider = os.getenv("LM_PROVIDER", provider)
    model = os.getenv("LM_MODEL", model) or {"openai":"gpt-4o-mini","anthropic":"claude-3-haiku-20240307","gemini":"gemini-1.5-flash"}.get(provider.lower(), "")
    prompt = (
        "Write a one-sentence natural caption describing a toy robot given its attributes.\n"
        f"Attributes (JSON): {json.dumps(concepts, ensure_ascii=False)}\n"
        "Avoid list-like phrasing; write like a human observation."
    )
    system = system or "You are concise and concrete. Use everyday language."
    out: List[str] = []

    prov = provider.lower()
    if prov == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=[{"role":"system","content":system},{"role":"user","content":prompt}], n=n, temperature=0.7)
        for c in resp.choices:
            out.append(c.message.content.strip())
        return out
    elif prov in ("anthropic","claude"):
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(model=model, max_tokens=120, temperature=0.7, system=system, messages=[{"role":"user","content":prompt}],)
        out.append("".join([b.text for b in msg.content if hasattr(b,'text')]))
        return out
    elif prov in ("gemini","google","googleai","google-genai"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(model or "gemini-1.5-flash")
        for _ in range(max(1,n)):
            r = gm.generate_content(prompt)
            out.append(r.text.strip())
        return out
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

def create_robot_text_dataset(concept_values: Dict[str, str], *, templates_path: Optional[Path] = None, n_inconsistent: int = 50) -> Dict[str, Any]:
    """
    Minimal text dataset constructor for (x, c[k]) style training.
    Returns a dict with:
      - 'texts': List[str] = [1 consistent + n_inconsistent inconsistent]
      - 'concepts': Dict[str,str] the input concept_values
    """
    engine = CaptionTemplateEngine()
    consistent = engine.consistent(concept_values)
    tpl_path = templates_path
    if tpl_path is None:
        # default to repo static/text_templates/Templates.txt if present
        try:
            from .paths import static_dir
            candidate = static_dir / "text_templates" / "Templates.txt"
            if candidate.exists():
                tpl_path = candidate
        except Exception:
            pass
    inconsistent = engine.inconsistent(concept_values, templates_file=tpl_path, n=n_inconsistent)
    return {"texts": [consistent] + inconsistent, "concepts": concept_values}