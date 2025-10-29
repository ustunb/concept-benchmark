from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
import torch

def _read_concepts(csv_path):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("empty concepts csv")
    cols = {c.lower().strip(): c for c in df.columns}
    name_key = None
    for k in ["concept", "name", "concept_name", "label", "term"]:
        if k in cols:
            name_key = cols[k]
            break
    if name_key is None:
        raise KeyError("missing concept column")
    alias_key = cols.get("aliases", None)
    regex_key = cols.get("regex", None)
    names = [str(x).strip() for x in df[name_key].tolist()]
    aliases = []
    regex = []
    for i in range(len(names)):
        a = []
        if alias_key is not None:
            v = df[alias_key].iloc[i]
            if isinstance(v, str) and v.strip():
                a = [s.strip() for s in v.split("|") if s.strip()]
        aliases.append(a)
        r = None
        if regex_key is not None:
            v = df[regex_key].iloc[i]
            if isinstance(v, str) and v.strip():
                r = v.strip()
        regex.append(r)
    return names, aliases, regex

def _compile_regex(rs):
    out = []
    for r in rs:
        if r is None or str(r).strip() == "":
            out.append(None)
            continue
        try:
            out.append(re.compile(str(r), flags=re.IGNORECASE))
        except Exception:
            out.append(None)
    return out

def _l2norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    return A / n

def _cosine(A, B):
    return _l2norm(A) @ _l2norm(B).T

def _sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

def _ridge(h, z, alpha):
    d = h.shape[1]
    a = alpha
    ht = h.T
    m = ht @ h
    m[np.diag_indices(d)] += a
    w = np.linalg.solve(m, ht @ z)
    return w.astype(np.float32, copy=False)

def _ridge_w(h, z, alpha, w):
    d = h.shape[1]
    a = alpha
    wv = w.reshape(-1, 1)
    ht = h.T
    m = ht @ (wv * h)
    m[np.diag_indices(d)] += a
    r = ht @ (wv * z)
    wmat = np.linalg.solve(m, r)
    return wmat.astype(np.float32, copy=False)

def _proj_learn(H, Zt, epochs, lr, device):
    dev = torch.device("cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu")
    Ht = torch.from_numpy(H).float().to(dev)
    Tt = torch.from_numpy(Zt).float().to(dev)
    d = Ht.shape[1]
    k = Tt.shape[1]
    W = torch.zeros((d, k), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W], lr=float(lr))
    for _ in range(int(epochs)):
        Y = Ht @ W
        Ym = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-6)
        Tm = (Tt - Tt.mean(dim=0, keepdim=True)) / (Tt.std(dim=0, keepdim=True) + 1e-6)
        c = torch.nn.functional.cosine_similarity(Ym.T, Tm.T, dim=0)
        loss = -(c.pow(3).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach().cpu().numpy().astype(np.float32, copy=False)

class _Encoder:
    def __init__(self, model_name, device):
        self._model_name = str(model_name)
        self._device = None
        try:
            if device and str(device) != "cpu" and torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        except Exception:
            self._device = None
        self._tok = None
        self._mod = None

    def _load(self):
        if self._tok is not None and self._mod is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        self._tok = AutoTokenizer.from_pretrained(self._model_name)
        self._mod = AutoModel.from_pretrained(self._model_name)
        if self._device and hasattr(self._mod, "to"):
            self._mod.to(self._device)
        self._mod.eval()

    def encode(self, texts, batch_size=64):
        self._load()
        tok = self._tok
        mod = self._mod
        vs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
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

def _dup_weights(texts):
    n = len(texts)
    d = {}
    for t in texts:
        d[t] = d.get(t, 0) + 1
    w = np.array([1.0 / d[t] for t in texts], dtype=np.float32)
    s = w.sum()
    if s <= 0:
        return np.ones(n, dtype=np.float32)
    w = w * (n / s)
    return w

def _class_balanced_weights(y):
    yv = np.asarray(y).reshape(-1)
    n = len(yv)
    n_pos = int((yv == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=np.float32)
    w = np.empty(n, dtype=np.float32)
    w[yv == 1] = 0.5 * n / n_pos
    w[yv == 0] = 0.5 * n / n_neg
    return w

def _weighted_corr(y, z, w):
    yv = np.asarray(y).astype(np.float32).reshape(-1)
    zv = np.asarray(z).astype(np.float32).reshape(-1)
    wv = np.asarray(w).astype(np.float32).reshape(-1)
    sw = wv.sum()
    my = (wv * yv).sum() / max(sw, 1e-8)
    mz = (wv * zv).sum() / max(sw, 1e-8)
    cyz = (wv * (yv - my) * (zv - mz)).sum() / max(sw, 1e-8)
    vy = (wv * (yv - my) ** 2).sum() / max(sw, 1e-8)
    vz = (wv * (zv - mz) ** 2).sum() / max(sw, 1e-8)
    denom = np.sqrt(max(vy, 0.0) * max(vz, 0.0)) + 1e-12
    r = cyz / denom
    if not np.isfinite(r):
        return 0.0
    return float(abs(r))

def _rank_topk(Z, y, keep_k, supervised=True, weights=None, metric="corr"):
    K = Z.shape[1]
    if keep_k <= 0 or keep_k >= K:
        return list(range(K))
    if supervised:
        if y is None:
            return list(range(K))
        w = None
        if weights is not None:
            w = np.asarray(weights).reshape(-1)
        scores = []
        for j in range(K):
            zj = Z[:, j]
            if metric == "corr":
                s = _weighted_corr(y, zj, w if w is not None else np.ones(len(zj), dtype=np.float32))
            else:
                s = _weighted_corr(y, zj, w if w is not None else np.ones(len(zj), dtype=np.float32))
            scores.append(s)
        idx = np.argsort(-np.asarray(scores))
        return idx[:keep_k].tolist()
    var = Z.var(axis=0)
    idx = np.argsort(-var)
    return idx[:keep_k].tolist()

class LabelFreeDetector:
    def __init__(self, settings):
        base = dict(globals().get("settings", {}))
        cfg = dict(base)
        if settings:
            cfg.update(settings)
        self.settings = cfg
        self.concept_names = []
        self._concept_aliases = []
        self._concept_regex = []
        self._regex_compiled = []
        self._encoder = None
        self._E = None
        self._W = None
        self._keep_idx = None
        self._fitted = False
        self._proj_on = False

    def fit(self, train_texts, y=None):
        names, aliases, regex = _read_concepts(self.settings["concepts_csv"])
        self.concept_names = names
        self._concept_aliases = aliases
        self._concept_regex = regex
        self._regex_compiled = _compile_regex(regex)
        enc = _Encoder(self.settings["lf_encoder"], self.settings["lf_device"])
        self._encoder = enc
        terms = [" ".join([names[i]] + aliases[i]) for i in range(len(names))]
        self._E = enc.encode(terms, int(self.settings["lf_batch_size"]))
        H = enc.encode(list(train_texts), int(self.settings["lf_batch_size"]))
        Zt = _cosine(H, self._E)
        Zt = 0.5 * (Zt + 1.0)
        if bool(self.settings.get("proj_enable", False)):
            self._W = _proj_learn(H, Zt, int(self.settings["proj_epochs"]), float(self.settings["proj_lr"]), self.settings["lf_device"])
            Z = _sigmoid(H @ self._W)
            self._proj_on = True
        elif bool(self.settings.get("lf_ridge", False)):
            if int(self.settings.get("lf_ridge_dedup", 0)) == 1:
                wd = _dup_weights(list(train_texts))
                self._W = _ridge_w(H, Zt, float(self.settings.get("lf_ridge_alpha", 1.0)), wd)
            else:
                self._W = _ridge(H, Zt, float(self.settings.get("lf_ridge_alpha", 1.0)))
            Z = _sigmoid(H @ self._W)
            self._proj_on = False
        else:
            self._W = None
            self._proj_on = False
            Z = Zt
        keep_k = int(self.settings.get("lf_keep_k", 0))
        if keep_k > 0:
            w_sel = None
            if bool(self.settings.get("lf_topk_supervised", True)) and y is not None:
                w_sel = np.ones(len(train_texts), dtype=np.float32)
                if int(self.settings.get("lf_topk_dedup", 0)) == 1:
                    w_sel *= _dup_weights(list(train_texts))
                if str(self.settings.get("lf_topk_weighting", "none")) == "class" and y is not None:
                    w_sel *= _class_balanced_weights(y)
            self._keep_idx = _rank_topk(
                Z, y, keep_k,
                supervised=bool(self.settings.get("lf_topk_supervised", True)),
                weights=w_sel,
                metric=str(self.settings.get("lf_topk_metric", "corr")),
            )
        else:
            self._keep_idx = list(range(Z.shape[1]))
        self._fitted = True

    def predict(self, texts):
        if not self._fitted:
            raise RuntimeError("not fitted")
        H = self._encoder.encode(list(texts), int(self.settings["lf_batch_size"]))
        if self._W is not None:
            Z = _sigmoid(H @ self._W)
        else:
            Z = _cosine(H, self._E)
            Z = 0.5 * (Z + 1.0)
        Z = Z[:, self._keep_idx]
        if str(self.settings.get("lf_mode", "hard")) == "hard":
            thr = float(self.settings.get("lf_threshold", 0.5))
            Z = (Z >= thr).astype(np.float32, copy=False)
        else:
            Z = Z.astype(np.float32, copy=False)
        return Z

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": 3,
            "settings": self.settings,
            "concept_names": self.concept_names,
            "concept_aliases": self._concept_aliases,
            "concept_regex": self._concept_regex,
            "keep_idx": self._keep_idx,
            "has_W": self._W is not None,
        }
        (p / "meta.json").write_text(json.dumps(meta))
        if self._E is not None:
            np.save(p / "E.npy", self._E)
        if self._W is not None:
            np.save(p / "W.npy", self._W)

    @classmethod
    def load(cls, path):
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj = cls(meta.get("settings", {}))
        obj.concept_names = list(meta.get("concept_names", []))
        obj._concept_aliases = list(meta.get("concept_aliases", []))
        obj._concept_regex = list(meta.get("concept_regex", []))
        obj._regex_compiled = _compile_regex(obj._concept_regex)
        obj._keep_idx = list(meta.get("keep_idx", []))
        E_path = p / "E.npy"
        if E_path.exists():
            obj._E = np.load(E_path)
        W_path = p / "W.npy"
        if W_path.exists():
            obj._W = np.load(W_path)
        obj._encoder = _Encoder(obj.settings["lf_encoder"], obj.settings["lf_device"])
        obj._fitted = True
        return obj

settings = {
    "concepts_csv": "data/robot_text/concepts/concepts.csv",
    "lf_alpha": 1.0,
    "lf_threshold": 0.5,
    "lf_mode": "hard",
    "lf_ridge": True,
    "lf_ridge_alpha": 1.0,
    "lf_ridge_dedup": 1,
    "lf_topk_supervised": True,
    "lf_topk_metric": "corr",
    "lf_topk_weighting": "none",
    "lf_topk_dedup": 0,
    "lf_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "lf_device": "cuda" if (lambda: hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available())() else "cpu",
    "lf_batch_size": 64,
    "lf_keep_k": 0,
    "lf_group_threshold": 0.9,
    "proj_enable": False,
    "proj_epochs": 200,
    "proj_lr": 0.05,
}

_parser = argparse.ArgumentParser(prog="lfcbm_text")
_parser.add_argument("--concepts-csv", type=str, default=settings["concepts_csv"])
_parser.add_argument("--lf-alpha", type=float, default=settings["lf_alpha"])
_parser.add_argument("--lf-threshold", type=float, default=settings["lf_threshold"])
_parser.add_argument("--lf-mode", type=str, choices=["hard", "soft"], default=settings["lf_mode"])
_parser.add_argument("--lf-ridge", action="store_true", default=settings["lf_ridge"])
_parser.add_argument("--lf-ridge-alpha", type=float, default=settings["lf_ridge_alpha"])
_parser.add_argument("--lf-ridge-dedup", type=int, default=settings["lf_ridge_dedup"])
_parser.add_argument("--lf-encoder", type=str, default=settings["lf_encoder"])
_parser.add_argument("--lf-device", type=str, default=settings["lf_device"])
_parser.add_argument("--lf-batch-size", type=int, default=settings["lf_batch_size"])
_parser.add_argument("--lf-keep-k", type=int, default=settings["lf_keep_k"])
_parser.add_argument("--lf-group-threshold", type=float, default=settings["lf_group_threshold"])
_parser.add_argument("--lf-topk-supervised", type=int, default=1)
_parser.add_argument("--lf-topk-metric", type=str, choices=["corr"], default=settings["lf_topk_metric"])
_parser.add_argument("--lf-topk-weighting", type=str, choices=["none", "class"], default=settings["lf_topk_weighting"])
_parser.add_argument("--lf-topk-dedup", type=int, default=settings["lf_topk_dedup"])
_parser.add_argument("--proj-enable", action="store_true", default=settings["proj_enable"])
_parser.add_argument("--proj-epochs", type=int, default=settings["proj_epochs"])
_parser.add_argument("--proj-lr", type=float, default=settings["proj_lr"])
_args, _unknown = _parser.parse_known_args()
settings.update(vars(_args))

print(settings)
