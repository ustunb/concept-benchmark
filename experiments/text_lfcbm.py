"""Label-free concept bottleneck model for text.

Extracted from scripts/utils/lfcbm_text.py with module-level settings and
argparse removed. All configuration is passed via the settings dict argument
to LabelFreeDetector.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _read_concepts(csv_path):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"concepts.csv not found: {p} (cwd={Path.cwd()})")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("concepts.csv is empty")
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_key = None
    for key in ("concept", "name", "term", "concept_name", "text"):
        if key in cols_lower:
            name_key = cols_lower[key]
            break
    if name_key is None:
        name_key = df.columns[0]
    aliases_key = (
        cols_lower.get("aliases")
        or cols_lower.get("alias")
        or cols_lower.get("synonyms")
    )
    regex_key = cols_lower.get("regex") or cols_lower.get("pattern")
    names = []
    aliases = []
    regex = []
    for _, r in df.iterrows():
        raw = r[name_key]
        n = str(raw).strip() if pd.notna(raw) else ""
        if not n:
            continue
        if (
            aliases_key is not None
            and aliases_key in r
            and isinstance(r[aliases_key], str)
        ):
            a = [t.strip() for t in str(r[aliases_key]).split(";") if t.strip()]
        else:
            a = []
        if regex_key is not None and regex_key in r and isinstance(r[regex_key], str):
            rx = str(r[regex_key]).strip()
        else:
            rx = ""
        names.append(n)
        aliases.append(a)
        regex.append(rx)
    if not names:
        raise ValueError("no concepts found")
    return names, aliases, regex


def _compile_regex(regex_list):
    out = []
    for rx in regex_list:
        try:
            out.append(re.compile(rx, re.I)) if rx else out.append(None)
        except re.error:
            out.append(None)
    return out


def _l2norm(a):
    v = np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return a / v


def _cosine(A, B):
    return _l2norm(A) @ _l2norm(B).T


def _lex_hits(texts, names, aliases, regex_compiled):
    m = len(names)
    Z = np.zeros((len(texts), m), dtype=np.float32)
    for i, t in enumerate(texts):
        s = str(t).lower()
        for j in range(m):
            hit = 0
            base = names[j].lower()
            if base and base in s:
                hit = 1
            if not hit:
                for al in aliases[j]:
                    if al.lower() in s:
                        hit = 1
                        break
            if not hit and regex_compiled[j] is not None:
                if regex_compiled[j].search(s) is not None:
                    hit = 1
            Z[i, j] = float(hit)
    return Z


def _ridge(h, z, alpha):
    d = h.shape[1]
    ht = h.T
    m = ht @ h
    m[np.diag_indices(d)] += alpha
    w = np.linalg.solve(m, ht @ z)
    return w.astype(np.float32, copy=False)


class _Encoder:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self._backend = None
        self._is_st = None

    def _lazy(self):
        if self._backend is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(self.model_name, device=self.device)
            self._is_st = True
        except (ImportError, OSError):
            from transformers import AutoTokenizer, AutoModel
            import torch

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._mod = AutoModel.from_pretrained(self.model_name)
            self._mod.eval()
            self._device = (
                torch.device(self.device)
                if self.device and self.device != "cpu"
                else torch.device("cpu")
            )
            self._mod.to(self._device)
            self._is_st = False

    def encode(self, texts, batch_size):
        self._lazy()
        if self._is_st:
            return self._backend.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        import torch

        tok = self._tok
        mod = self._mod
        vs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                enc = tok(chunk, padding=True, truncation=True, return_tensors="pt")
                if self._device and self._device.type != "cpu":
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                out = mod(**enc)
                last = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                last = last * mask
                s = last.sum(dim=1)
                d = mask.sum(dim=1).clamp(min=1)
                mean = s / d
                v = mean.detach().cpu().numpy()
                vs.append(v)
        return np.concatenate(vs, axis=0)


class LabelFreeDetector:
    """Label-free concept bottleneck model for text.

    Settings dict keys:
        concepts_csv, lf_alpha, lf_threshold, lf_mode, lf_ridge,
        lf_ridge_alpha, lf_encoder, lf_device, lf_batch_size,
        lf_keep_k, lf_group_threshold
    """

    DEFAULT_SETTINGS = {
        "concepts_csv": "",
        "lf_alpha": 0.5,
        "lf_threshold": 0.5,
        "lf_mode": "soft",
        "lf_ridge": False,
        "lf_ridge_alpha": 1.0,
        "lf_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "lf_device": "cpu",
        "lf_batch_size": 64,
        "lf_keep_k": 9,
        "lf_group_threshold": 0.9,
    }

    def __init__(self, settings: dict | None = None):
        _cfg = dict(self.DEFAULT_SETTINGS)
        if settings:
            _cfg.update(settings)
        self.settings = _cfg
        self.concept_names: list[str] = []
        self._concept_aliases: list[list[str]] = []
        self._concept_regex: list[str] = []
        self._regex_compiled: list = []
        self._encoder = None
        self._E = None
        self._W = None
        self._groups = None
        self._keep_idx = None
        self._fitted = False

    def _mix(self, H, texts):
        Z_cos = _cosine(H, self._E)
        Z_lex = _lex_hits(
            texts, self.concept_names, self._concept_aliases, self._regex_compiled
        )
        a = float(self.settings["lf_alpha"])
        return a * Z_cos + (1.0 - a) * Z_lex

    def _build_groups(self):
        K = len(self.concept_names)
        if K <= 9 and int(self.settings["lf_keep_k"]) >= K:
            self._groups = [[i] for i in range(K)]
            self._keep_idx = list(range(K))
            return
        S = _cosine(self._E, self._E)
        thr = float(self.settings.get("lf_group_threshold", 0.9))
        used = np.zeros(K, dtype=bool)
        groups = []
        for i in range(K):
            if used[i]:
                continue
            grp = [i]
            used[i] = True
            sim = S[i]
            idx = np.where((sim >= thr) & (~used))[0]
            for j in idx:
                grp.append(j)
                used[j] = True
            groups.append(grp)
        self._groups = groups

    def _rank_groups(self, Z, y):
        G = len(self._groups)
        scores = np.zeros(G, dtype=np.float32)
        if y is None:
            for g, grp in enumerate(self._groups):
                zg = Z[:, grp].mean(axis=1)
                scores[g] = np.var(zg)
        else:
            yb = np.asarray(y).astype(np.float32).ravel()
            yb = (yb - yb.mean()) / (yb.std() + 1e-8)
            for g, grp in enumerate(self._groups):
                zg = Z[:, grp].mean(axis=1)
                zg = (zg - zg.mean()) / (zg.std() + 1e-8)
                scores[g] = abs(float((zg * yb).mean()))
        k = int(self.settings["lf_keep_k"])
        order = np.argsort(-scores)
        keep = order[: min(k, len(order))].tolist()
        self._keep_idx = keep

    def fit(self, train_texts, y=None):
        names, aliases, regex = _read_concepts(self.settings["concepts_csv"])
        self.concept_names = names
        self._concept_aliases = aliases
        self._concept_regex = regex
        self._regex_compiled = _compile_regex(regex)

        enc = _Encoder(self.settings["lf_encoder"], self.settings["lf_device"])
        self._encoder = enc

        terms = [" ".join([names[i]] + aliases[i]) for i in range(len(names))]
        self._E = enc.encode(terms, self.settings["lf_batch_size"])
        H = enc.encode(list(train_texts), self.settings["lf_batch_size"])
        Z0 = self._mix(H, list(train_texts))

        if bool(self.settings["lf_ridge"]):
            self._W = _ridge(H, Z0, float(self.settings["lf_ridge_alpha"]))

        self._build_groups()
        self._rank_groups(Z0, y)

        kept_groups = (
            [self._groups[g] for g in (self._keep_idx or [])] if self._groups else []
        )
        if kept_groups:
            rep_idx = [grp[0] for grp in kept_groups]
            self.concept_names = [self.concept_names[i] for i in rep_idx]
            self._concept_aliases = [self._concept_aliases[i] for i in rep_idx]
            self._concept_regex = [self._concept_regex[i] for i in rep_idx]
            self._regex_compiled = _compile_regex(self._concept_regex)
            self._E = self._E[rep_idx]
            if self._W is not None:
                self._W = self._W[:, rep_idx]
            self._groups = [[i] for i in range(len(rep_idx))]
            self._keep_idx = list(range(len(rep_idx)))

        self._fitted = True

    def predict(self, texts):
        if not self._fitted:
            raise RuntimeError("fit required before predict")
        H = self._encoder.encode(list(texts), self.settings["lf_batch_size"])
        if self._W is not None:
            Z_base = H @ self._W
        else:
            Z_base = self._mix(H, list(texts))
        K = len(self.concept_names)
        out = np.zeros((Z_base.shape[0], K), dtype=np.float32)
        if self._groups is None:
            out[:] = Z_base
        else:
            for grp in self._groups:
                z = Z_base[:, grp].mean(axis=1)
                out[:, grp] = z[:, None]
        if self.settings["lf_mode"] == "hard":
            thr = float(self.settings["lf_threshold"])
            return (out >= thr).astype(np.float32, copy=False)
        return out.astype(np.float32, copy=False)

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": 2,
            "settings": self.settings,
            "concept_names": self.concept_names,
            "concept_aliases": self._concept_aliases,
            "concept_regex": self._concept_regex,
            "has_W": self._W is not None,
            "groups": self._groups,
            "keep_idx": self._keep_idx,
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
        obj.concept_names = list(meta.get("concept_names", []))
        obj._concept_aliases = list(meta.get("concept_aliases", []))
        obj._concept_regex = list(meta.get("concept_regex", []))
        obj._regex_compiled = _compile_regex(obj._concept_regex)
        obj._E = np.load(p / "E.npy")
        if meta.get("has_W", False):
            obj._W = np.load(p / "W.npy")
        obj._groups = meta.get("groups")
        obj._keep_idx = meta.get("keep_idx")
        obj._encoder = _Encoder(obj.settings["lf_encoder"], obj.settings["lf_device"])
        obj._fitted = True
        return obj
