# concept_benchmark/synthetic/robot_concepts/text_multi_nn.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, re
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

_word_re = re.compile(r"[A-Za-z0-9_]+")

def _tok(s: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(s or "")]

def _tok_bi(s: str) -> List[str]:
    t = _tok(s)
    if not t: return t
    return t + [f"{t[i]}__{t[i+1]}" for i in range(len(t)-1)]

def build_vocab(texts: List[str], max_size: int = 40000, min_freq: int = 1, use_bigrams: bool = True) -> Dict[str,int]:
    ctr = Counter()
    for s in texts:
        ctr.update(_tok_bi(s) if use_bigrams else _tok(s))
    items = [w for w,c in ctr.items() if c >= min_freq]
    items.sort(key=lambda w: (-ctr[w], w))
    items = items[:max_size-2]
    vocab = {"<pad>":0, "<unk>":1}
    for i,w in enumerate(items, start=2):
        vocab[w] = i
    return vocab

def numericalize(s: str, vocab: Dict[str,int], use_bigrams: bool = True) -> List[int]:
    toks = _tok_bi(s) if use_bigrams else _tok(s)
    return [vocab.get(t, 1) for t in toks] or [1]

class _MultiConceptDS(Dataset):
    def __init__(self, texts: List[str], C: np.ndarray, vocab: Dict[str,int], use_bigrams: bool = True):
        self.texts = texts
        self.C = C.astype(np.float32)
        self.vocab = vocab
        self.use_bigrams = use_bigrams
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        ids = numericalize(self.texts[i], self.vocab, self.use_bigrams)
        return ids, self.C[i]

def _collate(batch):
    ids_all: List[int] = []
    offsets = [0]
    labels = []
    acc = 0
    for ids, y in batch:
        ids_all.extend(ids)
        acc += len(ids)
        offsets.append(acc)
        labels.append(y)
    x = torch.tensor(ids_all, dtype=torch.long)
    o = torch.tensor(offsets[:-1], dtype=torch.long)
    y = torch.tensor(np.stack(labels), dtype=torch.float32)
    return x, o, y

class MultiConceptTextNN(nn.Module):
    def __init__(self, vocab_size: int, n_outputs: int, embed_dim: int = 128, hidden_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=0)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs),
        )
    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids, offsets)
        return self.ff(x)  # logits [B, n_outputs]

@dataclass
class TrainResult:
    model: nn.Module
    vocab: Dict[str,int]
    metrics_macro: Dict[str,float]
    metrics_per_concept: Dict[str,Dict[str,float]]
    concepts: List[str]
    device: str
    proba_eval: Optional[np.ndarray] = None
    C_eval: Optional[np.ndarray] = None
    split_used: str = "validation"

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str, concept_names: List[str]) -> Tuple[Dict[str,float], Dict[str,Dict[str,float]], np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    for ids, offs, y in loader:
        ids, offs = ids.to(device), offs.to(device)
        logit = model(ids, offs)
        prob = torch.sigmoid(logit).cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy())
    P = np.concatenate(ps, axis=0)
    Y = np.concatenate(ys, axis=0)
    per = {}
    ap_list, ra_list = [], []
    for j, name in enumerate(concept_names):
        yj, pj = Y[:, j], P[:, j]
        m = {}
        try: m["auprc"] = float(average_precision_score(yj, pj)); ap_list.append(m["auprc"])
        except Exception: m["auprc"] = float("nan")
        try: m["roc_auc"] = float(roc_auc_score(yj, pj)) if len(np.unique(yj)) > 1 else float("nan"); ra_list.append(m["roc_auc"])
        except Exception: m["roc_auc"] = float("nan")
        per[name] = m
    macro = {
        "auprc_macro": float(np.nanmean(ap_list)) if ap_list else float("nan"),
        "roc_auc_macro": float(np.nanmean(ra_list)) if ra_list else float("nan"),
    }
    return macro, per, P, Y

def _pos_weight(C: np.ndarray) -> torch.Tensor:
    pw = []
    for j in range(C.shape[1]):
        p = C[:, j].sum()
        n = C.shape[0] - p
        if p <= 0: pw.append(1.0)
        elif n <= 0: pw.append(1.0)
        else: pw.append(float(n / max(p, 1e-6)))
    return torch.tensor(pw, dtype=torch.float32)

def train_concept_detector_text_multi(dataset, *, eval_split: str = "validation", embed_dim: int = 128, hidden_dim: int = 192, dropout: float = 0.1, max_vocab_size: int = 40000, min_freq: int = 1, use_bigrams: bool = True, epochs: int = 6, batch_size: int = 64, lr: float = 2e-3, weight_decay: float = 0.0, device: str | None = None, seed: int = 0) -> TrainResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if eval_split == "validation" and hasattr(dataset, "validation") and dataset.validation.n > 0:
        eval_sample = dataset.validation; split_used = "validation"
    elif eval_split == "test" and hasattr(dataset, "test") and dataset.test.n > 0:
        eval_sample = dataset.test; split_used = "test"
    elif eval_split == "training" and hasattr(dataset, "training"):
        eval_sample = dataset.training; split_used = "training"
    else:
        eval_sample = dataset._full; split_used = "full"

    train_sample = dataset.training if hasattr(dataset, "training") and dataset.training.n > 0 else dataset._full

    X_tr = [str(x) for x in train_sample.X]
    C_tr = np.asarray(train_sample.C).astype(np.float32)
    X_ev = [str(x) for x in eval_sample.X]
    C_ev = np.asarray(eval_sample.C).astype(np.float32)

    vocab = build_vocab(X_tr, max_size=max_vocab_size, min_freq=min_freq, use_bigrams=use_bigrams)

    ds_tr = _MultiConceptDS(X_tr, C_tr, vocab, use_bigrams)
    ds_ev = _MultiConceptDS(X_ev, C_ev, vocab, use_bigrams)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0)
    dl_ev = DataLoader(ds_ev, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)

    torch.manual_seed(seed)
    model = MultiConceptTextNN(vocab_size=len(vocab), n_outputs=C_tr.shape[1], embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = _pos_weight(C_tr).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_state, best_ap = None, -math.inf
    for _ in range(epochs):
        model.train()
        for ids, offs, y in dl_tr:
            ids, offs, y = ids.to(device), offs.to(device), y.to(device)
            logits = model(ids, offs)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        macro, _, _, _ = _evaluate(model, dl_ev, device, dataset.concepts)
        if macro["auprc_macro"] > best_ap:
            best_ap = macro["auprc_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    macro, per, P, Y = _evaluate(model, dl_ev, device, dataset.concepts)

    return TrainResult(
        model=model,
        vocab=vocab,
        metrics_macro=macro,
        metrics_per_concept=per,
        concepts=list(dataset.concepts),
        device=device,
        proba_eval=P,
        C_eval=Y,
        split_used=split_used,
    )

@torch.no_grad()
def predict_proba_text_multi(model: nn.Module, vocab: Dict[str,int], texts: List[str], *, use_bigrams: bool = True, batch_size: int = 256, device: str | None = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = _MultiConceptDS(texts, np.zeros((len(texts), model.ff[-1].out_features), dtype=np.float32), vocab, use_bigrams)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    model.eval().to(device)
    outs = []
    for ids, offs, _ in dl:
        ids, offs = ids.to(device), offs.to(device)
        prob = torch.sigmoid(model(ids, offs)).cpu().numpy()
        outs.append(prob)
    return np.concatenate(outs, axis=0)
