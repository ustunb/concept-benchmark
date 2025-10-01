from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd

def _read_concepts(csv_path):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"concepts.csv not found: {p} (cwd={Path.cwd()})")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("concepts.csv is empty")
    lower_map = {c: c for c in df.columns}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_key = None
    for k in ["name", "concept", "concept_name", "token"]:
        if k in cols_lower:
            name_key = cols_lower[k]
            break
    if name_key is None:
        name_key = df.columns[0]
    aliases_key = cols_lower.get("aliases") or cols_lower.get("alias") or cols_lower.get("synonyms")
    regex_key = cols_lower.get("regex") or cols_lower.get("pattern")
    names = []
    aliases = []
    regex = []
    for _, r in df.iterrows():
        raw = r[name_key]
        n = str(raw).strip() if pd.notna(raw) else ""
        if not n:
            continue
        if aliases_key is not None and aliases_key in r and isinstance(r[aliases_key], str):
            a = [t.strip() for t in str(r[aliases_key]).split(";") if t.strip()]
        else:
            a = []
        if regex_key is not None and regex_key in r and isinstance(r[regex_key], str) and str(r[regex_key]).strip():
            rgx = str(r[regex_key]).strip()
        else:
            rgx = None
        names.append(n)
        aliases.append(a)
        regex.append(rgx)
    if not names:
        raise ValueError("concepts.csv did not yield any concept names")
    return names, aliases, regex

def _compile_regex(patterns):
    out = []
    for p in patterns:
        if p is None:
            out.append(None)
        else:
            out.append(re.compile(p, flags=re.IGNORECASE))
    return out

class _Encoder:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self._backend = None
        self._is_st = None

    def _lazy_init(self):
        if self._backend is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._backend = SentenceTransformer(self.model_name, device=self.device)
            self._is_st = True
        except Exception:
            from transformers import AutoTokenizer, AutoModel
            import torch
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mod = AutoModel.from_pretrained(self.model_name)
            if self.device and self.device != "cpu":
                mod = mod.to(self.device)
            self._backend = (tok, mod)
            self._is_st = False

    def encode(self, texts, batch_size):
        self._lazy_init()
        if self._is_st:
            v = self._backend.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
            return v.astype(np.float32, copy=False)
        else:
            import torch
            tok, mod = self._backend
            vs = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                enc = tok(chunk, padding=True, truncation=True, return_tensors="pt")
                if self.device and self.device != "cpu":
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    out = mod(**enc)
                    last = out.last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1)
                    last = last * mask
                    s = last.sum(dim=1)
                    d = mask.sum(dim=1).clamp(min=1)
                    mean = s / d
                    v = mean.detach().cpu().numpy()
                    vs.append(v.astype(np.float32, copy=False))
            return np.concatenate(vs, axis=0) if vs else np.zeros((0, mod.config.hidden_size), dtype=np.float32)

def _l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _cosine(h, e):
    h = _l2norm(h)
    e = _l2norm(e)
    return h @ e.T

def _lex_hits(texts, names, aliases, regex):
    m = len(texts)
    k = len(names)
    z = np.zeros((m, k), dtype=np.float32)
    low_names = [n.lower() for n in names]
    low_aliases = [[a.lower() for a in al] for al in aliases]
    for i, t in enumerate(texts):
        tl = t.lower()
        for j in range(k):
            hit = 0
            if regex[j] is not None:
                if regex[j].search(t) is not None:
                    hit = 1
            else:
                if low_names[j] in tl:
                    hit = 1
                else:
                    for a in low_aliases[j]:
                        if a in tl:
                            hit = 1
                            break
            z[i, j] = 1.0 if hit else 0.0
    return z

def _ridge(h, z, alpha):
    d = h.shape[1]
    a = alpha
    ht = h.T
    m = ht @ h
    m[np.diag_indices(d)] += a
    w = np.linalg.solve(m, ht @ z)
    return w.astype(np.float32, copy=False)

class LabelFreeDetector:
    def __init__(self, settings):
        self.settings = dict(settings)
        self.concept_names = []
        self._concept_aliases = []
        self._concept_regex = []
        self._regex_compiled = []
        self._encoder = None
        self._E = None
        self._W = None
        self._fitted = False

    def fit(self, train_texts):
        names, aliases, regex = _read_concepts(self.settings["concepts_csv"])
        self.concept_names = names
        self._concept_aliases = aliases
        self._concept_regex = regex
        self._regex_compiled = _compile_regex(regex)
        enc = _Encoder(self.settings["lf_encoder"], self.settings["lf_device"])
        self._encoder = enc
        terms = []
        for i in range(len(names)):
            t = [names[i]] + aliases[i]
            terms.append(" ".join(t))
        self._E = enc.encode(terms, self.settings["lf_batch_size"])
        zcos = _cosine(self._E, self._E)
        if np.isnan(zcos).any():
            raise ValueError("concept embedding produced NaN")
        z0_train = self._mix(enc.encode(train_texts, self.settings["lf_batch_size"]), train_texts)
        if self.settings["lf_ridge"]:
            self._W = _ridge(self._H_cache, z0_train, self.settings["lf_ridge_alpha"])
        self._fitted = True
        return self

    def _mix(self, H, texts):
        self._H_cache = H.astype(np.float32, copy=False)
        Z_cos = _cosine(H, self._E)
        Z_lex = _lex_hits(texts, self.concept_names, self._concept_aliases, self._regex_compiled)
        a = float(self.settings["lf_alpha"])
        Z = a * Z_cos + (1.0 - a) * Z_lex
        return Z

    def predict(self, texts):
        if not self._fitted:
            raise RuntimeError("fit required before predict")
        H = self._encoder.encode(texts, self.settings["lf_batch_size"])
        if self._W is not None:
            Z = H @ self._W
        else:
            Z = self._mix(H, texts)
        if self.settings["lf_mode"] == "hard":
            thr = float(self.settings["lf_threshold"])
            return (Z >= thr).astype(np.float32, copy=False)
        return Z.astype(np.float32, copy=False)

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": 1,
            "settings": self.settings,
            "concept_names": self.concept_names,
            "concept_aliases": self._concept_aliases,
            "concept_regex": self._concept_regex,
            "has_W": self._W is not None,
        }
        (p / "lfcbm.json").write_text(json.dumps(meta, ensure_ascii=False))
        np.save(p / "E.npy", self._E)
        if self._W is not None:
            np.save(p / "W.npy", self._W)

    @classmethod
    def load(cls, path):
        p = Path(path)
        meta = json.loads((p / "lfcbm.json").read_text())
        obj = cls(meta["settings"])
        obj.concept_names = meta["concept_names"]
        obj._concept_aliases = meta["concept_aliases"]
        obj._concept_regex = meta["concept_regex"]
        obj._regex_compiled = _compile_regex(obj._concept_regex)
        obj._encoder = _Encoder(obj.settings["lf_encoder"], obj.settings["lf_device"])
        obj._E = np.load(p / "E.npy")
        if meta.get("has_W", False):
            obj._W = np.load(p / "W.npy")
        obj._fitted = True
        return obj

settings = {
    "concepts_csv": "data/robot_text/concepts/concepts.csv",
    "lf_alpha": 0.5,
    "lf_threshold": 0.5,
    "lf_mode": "soft",
    "lf_ridge": False,
    "lf_ridge_alpha": 1.0,
    "lf_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "lf_device": "cuda" if (lambda: hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available())() else "cpu",
    "lf_batch_size": 64,
}

_parser = argparse.ArgumentParser(prog="lfcbm_text")
_parser.add_argument("--concepts-csv", type=str, default=settings["concepts_csv"])
_parser.add_argument("--lf-alpha", type=float, default=settings["lf_alpha"])
_parser.add_argument("--lf-threshold", type=float, default=settings["lf_threshold"])
_parser.add_argument("--lf-mode", type=str, choices=["hard", "soft"], default=settings["lf_mode"])
_parser.add_argument("--lf-ridge", action="store_true", default=settings["lf_ridge"])
_parser.add_argument("--lf-ridge-alpha", type=float, default=settings["lf_ridge_alpha"])
_parser.add_argument("--lf-encoder", type=str, default=settings["lf_encoder"])
_parser.add_argument("--lf-device", type=str, default=settings["lf_device"])
_parser.add_argument("--lf-batch-size", type=int, default=settings["lf_batch_size"])
_args, _unknown = _parser.parse_known_args()
settings.update(vars(_args))

