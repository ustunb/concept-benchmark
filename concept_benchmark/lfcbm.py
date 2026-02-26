# concept_benchmark/lfcbm.py
from __future__ import annotations

__all__ = ["LFTrainingConfig", "LFConceptSet", "LabelFreeCBM"]

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_EPS = 1e-8


# --------- Configs and concept set handling ---------

@dataclass
class LFTrainingConfig:
    # CLIP backbone
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: str = "cpu"

    # Projection learning (Wc)
    lr: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 10
    seed: int = 0

    # Final sparse layer (Elastic-Net via sklearn SAGA)
    l1_ratio: float = 0.99
    # Grid for inverse regularization strength C; we pick the one that hits ~25–35 nnz/class
    C_grid: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    target_nonzero_per_class: Tuple[int, int] = (25, 35)

    # Caching of embeddings
    cache_dir: Optional[Path] = None
    batch_size: int = 256


class LFConceptSet:
    """Stores concept keys and display texts, aligns to dataset concept order if provided."""

    def __init__(self, keys: List[str], texts: List[str]) -> None:
        if len(keys) != len(texts):
            raise ValueError("keys and texts must have equal length")
        self.keys = list(keys)
        self.texts = list(texts)

    @staticmethod
    def _normalize_key(s: str) -> str:
        # turn "foot_shape_pointy_4sided" into a more readable default
        return s.replace("_", " ")

    @classmethod
    def from_file(cls, path: str | Path, dataset_keys: Optional[Sequence[str]] = None) -> "LFConceptSet":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Concept file not found: {p}")
        ext = p.suffix.lower()

        keys: List[str] = []
        texts: List[str] = []

        if ext == ".csv":
            import csv
            with p.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                hdr = [h.strip().lower() for h in reader.fieldnames or []]
                if "text" not in hdr and len(hdr) == 1:
                    # single column CSV
                    f.seek(0)
                    reader = csv.reader(f)
                    next(reader, None)
                    texts = [row[0].strip() for row in reader if row]
                    keys = [str(i) for i in range(len(texts))]
                else:
                    for row in reader:
                        t = (row.get("text") or row.get("concept") or "").strip()
                        k = (row.get("key") or row.get("id") or t).strip()
                        if t:
                            texts.append(t)
                            keys.append(k if k else t)
        elif ext in {".json", ".jsonl"}:
            import json
            if ext == ".jsonl":
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        t = (obj.get("text") or obj.get("concept") or "").strip()
                        k = (obj.get("key") or obj.get("id") or t).strip()
                        if t:
                            texts.append(t)
                            keys.append(k if k else t)
            else:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and "concepts" in obj:
                    obj = obj["concepts"]
                if not isinstance(obj, (list, tuple)):
                    raise ValueError("JSON concept file must be a list of objects with 'text' (and optional 'key').")
                for it in obj:
                    t = (it.get("text") or it.get("concept") or "").strip()
                    k = (it.get("key") or it.get("id") or t).strip()
                    if t:
                        texts.append(t)
                        keys.append(k if k else t)
        else:  # .txt or others -> one concept per line
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            texts = lines
            keys = [str(i) for i in range(len(texts))]

        if dataset_keys is not None:
            ds = list(dataset_keys)
            # If keys match dataset keys, reorder to dataset order.
            if set(keys) >= set(ds):
                idx = [keys.index(k) for k in ds]
                keys = [keys[i] for i in idx]
                texts = [texts[i] for i in idx]
            # If no keys, assume file order matches dataset concept order.
            elif not any(k for k in keys if not k.isdigit()) and len(keys) == len(ds):
                keys = ds
                # Keep texts as they are
            # If neither aligns, fall back to mapping by simple normalization if possible.
            else:
                norm = {k.lower().replace("_", " "): i for i, k in enumerate(keys)}
                remap = []
                for k in ds:
                    i = norm.get(k.lower().replace("_", " "))
                    if i is None:
                        raise ValueError(f"Cannot align concept key '{k}' from dataset to provided concept file.")
                    remap.append(i)
                keys = ds
                texts = [texts[i] for i in remap]

        # Fill any missing text from key
        texts = [t if t else cls._normalize_key(k) for k, t in zip(keys, texts)]
        return cls(keys, texts)


# --------- CLIP feature extraction (open_clip preferred, clip as fallback) ---------

class _CLIPEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._device_str = str(device)
        self._loaded = False
        self._init_model()

    def _init_model(self) -> None:
        self.device = torch.device(self._device_str)
        self.backend = "open_clip"
        try:
            import open_clip  # type: ignore
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self._encode_text = self._encode_text_openclip
            self._encode_image = self._encode_image_openclip
        except ImportError:
            import clip  # type: ignore
            self.backend = "clip"
            name = self.model_name.replace("-", "/")
            self.model, self.preprocess = clip.load(name, device=self.device)
            self._encode_text = self._encode_text_clip
            self._encode_image = self._encode_image_clip
        self.model.eval()
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._init_model()

    def __getstate__(self):
        return {"model_name": self.model_name, "pretrained": self.pretrained, "_device_str": self._device_str}

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self.pretrained = state["pretrained"]
        self._device_str = state["_device_str"]
        self._loaded = False

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        self._ensure_loaded()
        out: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            feats = self._encode_text(batch)
            out.append(feats)
        x = np.concatenate(out, axis=0)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + _EPS)
        return x.astype(np.float32)

    @torch.no_grad()
    def encode_images(self, paths: Sequence[str | Path], batch_size: int = 256) -> np.ndarray:
        self._ensure_loaded()
        from torch.utils.data import DataLoader, Dataset

        class _ImgDS(Dataset):
            def __init__(self, items, preprocess):
                self.items = list(items)
                self.preprocess = preprocess
                self._base_dir = None
                for it in self.items:
                    p = Path(str(it))
                    if p.is_absolute() or (p.parent and p.parent != Path('.')):
                        self._base_dir = p.parent
                        break

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                p = Path(str(self.items[idx]))
                if not p.is_absolute() and not p.exists() and self._base_dir is not None:
                    p = self._base_dir / p
                try:
                    im = Image.open(p).convert("RGB")
                except FileNotFoundError:
                    # drop an accidentally duplicated tail directory, e.g., ".../d/d/file" -> ".../d/file"
                    im = None
                    parts = p.parts
                    if len(parts) >= 3 and parts[-3] == parts[-2]:
                        p2 = Path(*parts[:-2]) / parts[-1]
                        if p2.exists():
                            im = Image.open(p2).convert("RGB")
                    # try parent-of-base + last two segments, handles inputs like "test_images/file"
                    if im is None and self._base_dir is not None and len(parts) >= 2:
                        cand = self._base_dir.parent / Path(*parts[-2:])
                        if cand.exists():
                            im = Image.open(cand).convert("RGB")
                    if im is None:
                        raise
                return self.preprocess(im)

        ds = _ImgDS(paths, self.preprocess)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_feats: List[np.ndarray] = []
        for xb in dl:
            xb = xb.to(self.device)
            feats = self._encode_image(xb)
            all_feats.append(feats)
        x = np.concatenate(all_feats, axis=0)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + _EPS)
        return x.astype(np.float32)

    # backends
    def _encode_text_openclip(self, batch: Sequence[str]) -> np.ndarray:
        toks = self.tokenizer(batch).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(toks)
        return feats.detach().cpu().numpy()

    def _encode_image_openclip(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            feats = self.model.encode_image(x)
        return feats.detach().cpu().numpy()

    def _encode_text_clip(self, batch: Sequence[str]) -> np.ndarray:
        import clip  # type: ignore

        toks = clip.tokenize(batch).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(toks)
        return feats.detach().cpu().numpy()

    def _encode_image_clip(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            feats = self.model.encode_image(x)
        return feats.detach().cpu().numpy()


# --------- Label-free CBM core ---------

def _cos_cubed_similarity(q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, p: shape (M, N), rows = concepts, columns = dataset items
    Implements Eq. (1) with per-row z-score, element-wise cube, then cosine similarity.
    Returns (mean_loss, per_concept_sim).
    """
    # z-score per row
    def _z3(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, unbiased=False, keepdim=True) + 1e-6
        z = (x - mu) / sd
        return z * z * z

    q3 = _z3(q)
    p3 = _z3(p)
    num = (q3 * p3).sum(dim=1)
    den = torch.linalg.vector_norm(q3, dim=1) * torch.linalg.vector_norm(p3, dim=1) + 1e-8
    sim = num / den
    loss = -sim.mean()
    return loss, sim


class LabelFreeCBM:
    """Implements LF-CBM steps: CLIP features, Wc via cos^3, concept filtering, sparse final layer."""

    def __init__(self, cfg: LFTrainingConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        self.encoder = _CLIPEncoder(cfg.clip_model, cfg.clip_pretrained, cfg.device)
        self.concept_set: Optional[LFConceptSet] = None

        # Learned artefacts
        self.Wc: Optional[torch.Tensor] = None  # shape (M, D)
        self.keep_mask: Optional[np.ndarray] = None  # shape (M,)
        self.scaler: Optional[StandardScaler] = None
        self.classifier: Optional[LogisticRegression] = None

        # Cached embeddings
        self._img_train: Optional[np.ndarray] = None
        self._img_valid: Optional[np.ndarray] = None
        self._img_test: Optional[np.ndarray] = None
        self._txt_concepts: Optional[np.ndarray] = None

        self.train_stats: Dict[str, Any] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop cached embeddings (large, not needed for inference)
        for k in ("_img_train", "_img_valid", "_img_test", "_txt_concepts"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in ("_img_train", "_img_valid", "_img_test", "_txt_concepts"):
            if k not in self.__dict__:
                self.__dict__[k] = None

    # --------- Fit / Transform ---------

    def fit(
        self,
        *,
        train_X: Sequence[str | Path],
        train_y: np.ndarray,
        valid_X: Sequence[str | Path],
        valid_y: np.ndarray,
        concept_set: LFConceptSet,
        cache_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        self.concept_set = concept_set
        cache = Path(cache_dir or self.cfg.cache_dir or ".").resolve()
        cache.mkdir(parents=True, exist_ok=True)
        logger.debug("LFCBM cache dir: %s", cache)

        # 1) Encode images and concepts (CLIP) and cache them
        def _cache(path: Path, arr: np.ndarray) -> None:
            np.save(path, arr)

        def _maybe(path: Path, build) -> np.ndarray:
            if path.exists():
                return np.load(path)
            arr = build()
            _cache(path, arr)
            return arr

        self._img_train = _maybe(cache / "clip_img_train.npy", lambda: self.encoder.encode_images(train_X, self.cfg.batch_size))
        self._img_valid = _maybe(cache / "clip_img_valid.npy", lambda: self.encoder.encode_images(valid_X, self.cfg.batch_size))
        self._txt_concepts = _maybe(cache / "clip_txt_concepts.npy", lambda: self.encoder.encode_texts(concept_set.texts, self.cfg.batch_size))

        # 2) CLIP similarity matrix P (train/valid)
        P_tr = (self._img_train @ self._txt_concepts.T)  # (Ntr, M)
        P_va = (self._img_valid @ self._txt_concepts.T)  # (Nva, M)

        # 3) Learn Wc to maximize cos^3 similarity against CLIP concepts
        device = torch.device(self.cfg.device)
        F_tr = torch.from_numpy(self._img_train).to(device)  # (Ntr, D)
        F_va = torch.from_numpy(self._img_valid).to(device)  # (Nva, D)
        P_tr_t = torch.from_numpy(P_tr.T).to(device)  # (M, Ntr)
        P_va_t = torch.from_numpy(P_va.T).to(device)  # (M, Nva)

        M, D = P_tr_t.shape[0], F_tr.shape[1]
        W = torch.randn(M, D, device=device) * 0.02
        W.requires_grad_(True)

        opt = torch.optim.Adam([W], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        best_val = -1e9
        best_W = None
        best_val_per = None
        no_improv = 0

        for epoch in range(int(self.cfg.max_epochs)):
            opt.zero_grad()
            Q_tr = W @ F_tr.T  # (M, Ntr)
            loss, sim_tr = _cos_cubed_similarity(Q_tr, P_tr_t)
            loss.backward()
            opt.step()

            with torch.no_grad():
                Q_va = W @ F_va.T
                val_loss, sim_va = _cos_cubed_similarity(Q_va, P_va_t)
                val_mean = float(sim_va.mean().item())

            self.train_stats.setdefault("train_sim", []).append(float(sim_tr.mean().item()))
            self.train_stats.setdefault("valid_sim", []).append(val_mean)

            if val_mean > best_val + 1e-6:
                best_val = val_mean
                best_W = W.detach().clone()
                best_val_per = sim_va.detach().clone()
                no_improv = 0
            else:
                no_improv += 1
                if no_improv >= int(self.cfg.patience):
                    break

        assert best_W is not None and best_val_per is not None
        self.Wc = best_W  # (M, D)

        # Filter concepts by validation similarity >= 0.45 (paper threshold)
        per = best_val_per.detach().cpu().numpy()
        keep = per >= 0.45
        if not np.any(keep):
            # fall back to keep everything if threshold is too strong
            keep = np.ones_like(keep, dtype=bool)
        self.keep_mask = keep

        # Reduce Wc and concept set to kept concepts
        self.Wc = self.Wc[keep]  # (Mk, D)
        kept_idx = np.where(keep)[0].tolist()
        self.concept_set = LFConceptSet(
            [self.concept_set.keys[i] for i in kept_idx],
            [self.concept_set.texts[i] for i in kept_idx],
        )
        # Recompute P with kept concepts
        P_tr_k = P_tr[:, keep]  # (Ntr, Mk)
        P_va_k = P_va[:, keep]  # (Nva, Mk)

        # 4) Build fc(x) and normalize features on training data
        Wk = self.Wc.detach().to(device)  # (Mk, D)
        fc_tr = (F_tr @ Wk.T).cpu().numpy()  # (Ntr, Mk)
        fc_va = (F_va @ Wk.T).cpu().numpy()  # (Nva, Mk)

        scaler = StandardScaler(with_mean=True, with_std=True)
        Z_tr = scaler.fit_transform(fc_tr)
        Z_va = scaler.transform(fc_va)
        self.scaler = scaler

        # 5) Train sparse final layer (multinomial logistic, Elastic-Net, SAGA)
        y_tr = np.asarray(train_y).astype(int)
        y_va = np.asarray(valid_y).astype(int)

        best_clf = None
        best_metric = -1e9

        low, high = self.cfg.target_nonzero_per_class

        for C in self.cfg.C_grid:
            clf = LogisticRegression(
                solver="saga",
                l1_ratio=float(self.cfg.l1_ratio),
                C=float(C),
                multi_class="multinomial",
                max_iter=5000,
                random_state=int(self.cfg.seed),
            )
            clf.fit(Z_tr, y_tr)
            nnz_per_class = (np.abs(clf.coef_) > 1e-8).sum(axis=1)
            ok = int((nnz_per_class >= low).all() and (nnz_per_class <= high).all())
            acc = float(clf.score(Z_va, y_va))
            # prefer hitting target sparsity; break ties with validation accuracy
            score = 2 * ok + acc
            if score > best_metric:
                best_metric = score
                best_clf = clf

        assert best_clf is not None
        self.classifier = best_clf

        # Return small summary
        self.train_stats.update(
            {
                "kept_concepts": int(keep.sum()),
                "total_concepts": int(len(keep)),
                "target_nnz_per_class": [low, high],
                "coef_nnz_per_class": (np.abs(self.classifier.coef_) > 1e-8).sum(axis=1).tolist(),
            }
        )
        return self.train_stats

    # Transform raw images to concept activations fc(x)
    def transform(self, X: Sequence[str | Path]) -> np.ndarray:
        if self.Wc is None:
            raise RuntimeError("Wc not learned. Call fit() first.")
        img = self.encoder.encode_images(X, self.cfg.batch_size)  # (N, D)
        Wk = self.Wc.detach().cpu().numpy()  # (Mk, D)
        fc = img @ Wk.T  # (N, Mk)
        return fc.astype(np.float32)

    # Convenience: compute concept probabilities σ(zscore(fc(x)))
    def concept_proba(self, X: Sequence[str | Path]) -> np.ndarray:
        fc = self.transform(X)  # (N, Mk)
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        Z = self.scaler.transform(fc)
        return 1.0 / (1.0 + np.exp(-Z))

    # Predict class probabilities from concept probabilities (P in [0,1]) using logit(P) as features.
    def predict_from_probs(self, P: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Classifier not fit. Call fit() first.")
        P = np.clip(P, 1e-6, 1 - 1e-6)
        Z = np.log(P / (1.0 - P))
        return self.classifier.predict_proba(Z)

    # Persist artefacts
    def save(self, out_dir: str | Path) -> Dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}
        # Wc
        W_path = out / "lfcbm_Wc.npy"
        np.save(W_path, self.Wc.detach().cpu().numpy())
        paths["Wc"] = str(W_path)

        # scaler
        S_path = out / "lfcbm_scaler.pkl"
        with open(S_path, "wb") as f:
            pickle.dump(self.scaler, f)
        paths["scaler"] = str(S_path)

        # classifier
        C_path = out / "lfcbm_classifier.pkl"
        with open(C_path, "wb") as f:
            pickle.dump(self.classifier, f)
        paths["classifier"] = str(C_path)

        # concepts
        meta = {
            "keys": self.concept_set.keys if self.concept_set else [],
            "texts": self.concept_set.texts if self.concept_set else [],
            "train_stats": self.train_stats,
        }
        M_path = out / "lfcbm_meta.json"
        M_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths["meta"] = str(M_path)

        return paths
