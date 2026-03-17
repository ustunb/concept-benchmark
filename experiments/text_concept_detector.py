from __future__ import annotations
import re
from typing import List, Dict, Optional, Union
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from concept_benchmark.data import ConceptDatasetSample
from experiments.models import ConceptDetector
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _tok(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def _make_bigrams(toks: List[str]) -> List[str]:
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def _cross_auroc(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    n = scores.shape[1]
    A = np.full((n, n), np.nan, dtype=np.float32)
    for j in range(n):
        s = scores[:, j]
        for k in range(n):
            y = truth[:, k]
            if (
                y.ndim == 1
                and len(np.unique(y)) == 2
                and y.sum() > 0
                and (1 - y).sum() > 0
            ):
                try:
                    A[j, k] = roc_auc_score(y, s)
                except Exception:
                    A[j, k] = np.nan
    return A


def _ece(scores: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    m = float(len(scores))
    out = 0.0
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i < n_bins - 1:
            mask = (scores >= left) & (scores < right)
        else:
            mask = (scores >= left) & (scores <= right)
        if mask.any():
            acc = float((scores[mask] >= 0.5).astype(int).mean())
            conf = float(scores[mask].mean())
            out += abs(acc - conf) * (mask.sum() / m)
    return float(out)


class _TxtDs(Dataset):
    def __init__(
        self, texts: List[str], labels: np.ndarray, masks: Optional[np.ndarray] = None
    ):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        if masks is None:
            masks = np.ones_like(self.labels, dtype=np.float32)
        self.masks = np.asarray(masks, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i], self.masks[i]


class _AttnPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x, mask):
        s = torch.matmul(x, self.w)
        s = s.masked_fill(mask.eq(0), -1e9)
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return (a * x).sum(dim=1)


def _collate_hf(batch, tokenizer, max_len):
    texts = [b[0] for b in batch]
    y = torch.tensor(np.stack([b[1] for b in batch], axis=0), dtype=torch.float32)
    m = torch.tensor(np.stack([b[2] for b in batch], axis=0), dtype=torch.float32)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc, y, m


class TextConceptDetector(ConceptDetector):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 192,
        epochs: int = 6,
        batch_size: int = 64,
        lr: float = 2e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.1,
        use_bigrams: bool = True,
        max_len: int = 128,
        min_freq: int = 1,
        max_vocab: Optional[int] = 40000,
        pos_weight: Union[None, str, np.ndarray] = "auto",
        output_mode: str = "hard",
        threshold_mode: str = "auto",
        validate: bool = True,
        device: Optional[str] = None,
        model_name: str = "distilbert/distilbert-base-uncased",  # alternatives: "prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-small", "distilbert/distilbert-base-uncased"
        pooling: str = "cls",
        base_lr: float = 2e-5,
        num_warmup_steps: int = 0,
        group_unknown_threshold: float = 0.55,
        **kwargs,
    ) -> None:
        super().__init__(embedding_model=None, concept_layers=None)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.base_lr = float(base_lr)
        self.num_warmup_steps = int(num_warmup_steps)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.use_bigrams = bool(use_bigrams)
        self.max_len = int(max_len)
        self.min_freq = int(min_freq)
        self.max_vocab = max_vocab
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.pos_weight_cfg: Union[None, str, np.ndarray] = pos_weight
        om = (output_mode or "hard").strip().lower()
        self.output_mode = "soft" if om == "soft" else "hard"
        tm = (threshold_mode or "auto").strip().lower()
        self.threshold_mode = "auto" if tm == "auto" else "fixed"
        self.validate_after_fit = bool(validate)
        self.cross_auroc_: Optional[np.ndarray] = None
        self.alignment_: Optional[Dict[str, float]] = None
        self.thresholds_: Optional[np.ndarray] = None
        self.concept_acc_: Optional[np.ndarray] = None
        self._n_concepts: Optional[int] = None
        self.model_name = str(model_name)
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.group_unknown_threshold = float(group_unknown_threshold)

    def _to_text_list(self, dataset: ConceptDatasetSample) -> List[str]:
        X = getattr(dataset, "X", None)
        if X is None:
            raise ValueError("Dataset missing X")
        return [str(v) for v in list(X)]

    def _build_model(self, n_out: int) -> nn.Module:
        base = AutoModel.from_pretrained(self.model_name)
        hsz = base.config.hidden_size
        if self.hidden_dim and self.hidden_dim > 0:
            head = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(hsz, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, n_out),
            )
        else:
            head = nn.Linear(hsz, n_out)
        m = nn.ModuleDict({"base": base, "head": head})
        if self.pooling == "attn":
            m["pool"] = _AttnPool(hsz)
        return m

    def _forward(self, model: nn.Module, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = model["base"](**enc, output_hidden_states=False, return_dict=True)
        h = out.last_hidden_state
        if self.pooling == "cls":
            z = h[:, 0]
        elif self.pooling == "mean":
            attn = enc["attention_mask"].unsqueeze(-1).float()
            z = (h * attn).sum(1) / attn.sum(1).clamp(min=1e-6)
        elif self.pooling == "attn":
            z = model["pool"](h, enc["attention_mask"])
        else:
            z = h[:, 0]
        logits = model["head"](z)
        return logits

    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: Optional[ConceptDatasetSample] = None,
        embed_params: Optional[dict] = None,
        l1_size: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs,
    ) -> None:
        Xtr = self._to_text_list(train_dataset)
        Ytr = np.asarray(train_dataset.C, dtype=np.float32)
        self._n_concepts = int(Ytr.shape[1])
        meta_tr = getattr(train_dataset, "meta", {}) or {}
        names = list(meta_tr.get("concepts", []))
        group_map = {}
        if len(names) == int(self._n_concepts):
            for j, name in enumerate(names):
                if "=" in name:
                    g = name.split("=", 1)[0]
                    group_map.setdefault(g, []).append(j)
        self._group_map = {g: idxs for g, idxs in group_map.items() if len(idxs) > 1}

        meta_tr = getattr(train_dataset, "meta", {}) or {}
        Mtr = meta_tr.get("observed_mask", None)
        if Mtr is None:
            Mtr = np.ones_like(Ytr, dtype=np.float32)

        tr_ds = _TxtDs(Xtr, Ytr, Mtr)
        tr_dl = DataLoader(
            tr_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: _collate_hf(b, self.tokenizer, self.max_len),
            pin_memory=(self.device in {"cuda", "mps"}),
            num_workers=0,
        )

        if valid_dataset is not None:
            Xva = self._to_text_list(valid_dataset)
            Yva = np.asarray(valid_dataset.C, dtype=np.float32)
            meta_va = getattr(valid_dataset, "meta", {}) or {}
            Mva = meta_va.get("observed_mask", None)
            if Mva is None:
                Mva = np.ones_like(Yva, dtype=np.float32)
            va_ds = _TxtDs(Xva, Yva, Mva)
            va_dl = DataLoader(
                va_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda b: _collate_hf(b, self.tokenizer, self.max_len),
                pin_memory=(self.device in {"cuda", "mps"}),
                num_workers=0,
            )
        else:
            va_dl = None
            Yva = None

        model = self._build_model(self._n_concepts).to(self.device)

        if (
            isinstance(self.pos_weight_cfg, str)
            and self.pos_weight_cfg.lower() == "auto"
        ):
            pos = torch.tensor(Ytr, dtype=torch.float32).sum(dim=0)
            neg = Ytr.shape[0] - pos
            pw = (neg / pos.clamp_min(1.0)).clamp_max(100.0)
            pos_weight = pw.to(self.device)
        elif isinstance(self.pos_weight_cfg, np.ndarray):
            pos_weight = torch.tensor(
                self.pos_weight_cfg, dtype=torch.float32, device=self.device
            )
        else:
            pos_weight = None

        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        base_params = list(model["base"].parameters())
        head_params = list(model["head"].parameters())
        other_params = list(model["pool"].parameters()) if "pool" in model else []
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": self.base_lr},
                {"params": head_params + other_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        total_steps = max(1, self.epochs * len(tr_dl))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.num_warmup_steps, total_steps
        )

        def _epoch(dloader, train_flag: bool):
            model.train() if train_flag else model.eval()
            total = 0.0
            denom_total = 0.0
            with torch.set_grad_enabled(train_flag):
                for enc, y, m in dloader:
                    enc = {
                        k: v.to(self.device, non_blocking=False) for k, v in enc.items()
                    }
                    y = y.to(self.device, non_blocking=False)
                    m = m.to(self.device, non_blocking=False)
                    logits = self._forward(model, enc)
                    loss_mat = crit(logits, y) * m
                    denom = m.sum().clamp(min=1.0)
                    loss = loss_mat.sum() / denom
                    if train_flag:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        scheduler.step()
                    total += float(loss_mat.sum().item())
                    denom_total += float(denom.item())
            denom_total = max(1.0, denom_total)
            return total / denom_total

        for _ in range(self.epochs):
            _ = _epoch(tr_dl, True)
            if va_dl is not None:
                _ = _epoch(va_dl, False)

        self.model = model.eval()
        self.embedding_model = None

        if valid_dataset is not None and self.validate_after_fit:
            old = self.output_mode
            self.output_mode = "soft"
            scores = self.predict(valid_dataset)
            self.output_mode = old
            truth = np.asarray(valid_dataset.C, dtype=np.float32)
            A = _cross_auroc(scores, truth.astype(int))
            self.cross_auroc_ = A
            diag = np.diag(A) if A.size else np.array([])
            off = A[~np.eye(A.shape[0], dtype=bool)] if A.size else np.array([])
            diag_mean = float(np.nanmean(diag)) if diag.size else float("nan")
            off_mean = float(np.nanmean(off)) if off.size else float("nan")
            if A.shape[0] > 1:
                row_max = np.nanmax(A, axis=1)
                row_second = []
                for r in range(A.shape[0]):
                    row = A[r].copy()
                    row[r] = np.nan
                    row_second.append(np.nanmax(row))
                sap_margin = float(
                    np.nanmean(row_max - np.array(row_second, dtype=np.float32))
                )
            else:
                sap_margin = float("nan")
            if A.size:
                argmax = np.nanargmax(A, axis=1)
                diag_top_fraction = float(np.mean(argmax == np.arange(A.shape[0])))
                diag_top90_fraction = float(
                    np.mean(((argmax == np.arange(A.shape[0])) & (diag >= 0.90)))
                )
            else:
                diag_top_fraction = float("nan")
                diag_top90_fraction = float("nan")
            eces = []
            for j in range(scores.shape[1]):
                try:
                    eces.append(_ece(scores[:, j], truth[:, j], n_bins=10))
                except Exception:
                    pass
            ece_macro = float(np.nanmean(eces)) if eces else float("nan")
            self.alignment_ = {
                "diag_mean": diag_mean,
                "off_mean": off_mean,
                "sap_margin": sap_margin,
                "diag_top_fraction": diag_top_fraction,
                "diag_top90_fraction": diag_top90_fraction,
                "ece_macro": ece_macro,
            }
            th = []
            acc = []
            for j in range(scores.shape[1]):
                yj = truth[:, j].astype(int)
                if len(np.unique(yj)) < 2:
                    th.append(0.5)
                    acc.append(np.nan)
                    continue
                try:
                    fpr, tpr, thr = roc_curve(yj, scores[:, j])
                    jstat = tpr - fpr
                    idx = int(np.nanargmax(jstat))
                    t = float(thr[idx])
                    if not np.isfinite(t):
                        t = 0.5
                    th.append(t)
                    pred = (scores[:, j] >= t).astype(int)
                    acc.append(float((pred == yj).mean()))
                except Exception:
                    th.append(0.5)
                    acc.append(np.nan)
            self.thresholds_ = np.asarray(th, dtype=np.float32)
            self.concept_acc_ = np.asarray(acc, dtype=np.float32)
        else:
            if self.threshold_mode == "auto":
                self.thresholds_ = np.full((self._n_concepts,), 0.5, dtype=np.float32)

    @torch.no_grad()
    def predict(
        self,
        dataset: ConceptDatasetSample,
        embed_params: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        texts = self._to_text_list(dataset)
        labels_dummy = np.zeros((len(texts), self._n_concepts), dtype=np.float32)
        ds = _TxtDs(texts, labels_dummy)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda b: _collate_hf(b, self.tokenizer, self.max_len),
            pin_memory=(self.device in {"cuda", "mps"}),
            num_workers=0,
        )
        outs: List[np.ndarray] = []
        for enc, _, _ in dl:
            enc = {k: v.to(self.device, non_blocking=False) for k, v in enc.items()}
            logits = self._forward(self.model, enc)
            prob = torch.sigmoid(logits).cpu().numpy()
            outs.append(prob)
        prob_all = (
            np.vstack(outs)
            if outs
            else np.zeros((0, self._n_concepts), dtype=np.float32)
        )
        if self.output_mode == "hard":
            thr = (
                self.thresholds_
                if self.thresholds_ is not None
                else np.full(self._n_concepts, 0.5, dtype=np.float32)
            )
            hard = (prob_all >= thr.reshape(1, -1)).astype(np.float32)
            gm = getattr(self, "_group_map", {})
            for _, idxs in gm.items() if isinstance(gm, dict) else []:
                sub = hard[:, idxs]
                cnt = sub.sum(axis=1)
                need = cnt != 1
                if np.any(need):
                    rows = np.where(need)[0]
                    psub = prob_all[rows][:, idxs]
                    pmax = psub.max(axis=1)
                    pick = psub.argmax(axis=1)
                    for r, pm, pk in zip(rows, pmax, pick):
                        if pm < self.group_unknown_threshold:
                            hard[r, idxs] = 0.0
                        else:
                            hard[r, idxs] = 0.0
                            hard[r, idxs[int(pk)]] = 1.0
            return hard
        return prob_all

    @property
    def n_concepts(self) -> int:
        if self._n_concepts is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return int(self._n_concepts)
