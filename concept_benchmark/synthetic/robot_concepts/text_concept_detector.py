from __future__ import annotations
import re
from typing import Iterable, List, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.models import ConceptDetector

def _tok(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def _make_bigrams(toks: List[str]) -> List[str]:
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

class Vocab:
    def __init__(self, min_freq: int = 1, max_size: Optional[int] = None):
        self.itos: List[str] = ["<pad>", "<unk>"]
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self.min_freq = int(min_freq)
        self.max_size = max_size

    def build(self, corpus: Iterable[List[str]]):
        from collections import Counter
        cnt = Counter()
        for toks in corpus:
            cnt.update(toks)
        words = [w for w, f in cnt.items() if f >= self.min_freq]
        words.sort(key=lambda w: (-cnt[w], w))
        limit = (max(0, int(self.max_size) - len(self.itos))) if self.max_size is not None else None
        if limit is not None:
            words = words[:limit]
        for w in words:
            if w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in toks]

class TextMultiLabelDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        vocab: Optional[Vocab] = None,
        use_bigrams: bool = True,
        max_len: int = 128,
        min_freq: int = 1,
        max_size: Optional[int] = 40000,
    ):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.use_bigrams = bool(use_bigrams)
        self.max_len = int(max_len)

        tokenized: List[List[str]] = []
        for s in self.texts:
            toks = _tok(str(s))
            if self.use_bigrams:
                toks = toks + _make_bigrams(toks)
            tokenized.append(toks)

        if vocab is None:
            vocab = Vocab(min_freq=min_freq, max_size=max_size)
            vocab.build(tokenized)
        self.vocab = vocab

        seqs: List[List[int]] = []
        for toks in tokenized:
            ids = self.vocab.encode(toks)
            if len(ids) > self.max_len:
                ids = ids[: self.max_len]
            seqs.append(ids if ids else [0])
        self.seqs = seqs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.seqs[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def collate(batch):
        xs, ys = zip(*batch) if batch else ([], [])
        if not xs:
            return torch.zeros((0,1), dtype=torch.long), torch.zeros((0,), dtype=torch.long), torch.zeros((0,0), dtype=torch.float32)
        lengths = [len(x) for x in xs]
        maxlen = max(lengths) if lengths else 1
        padded = torch.full((len(xs), maxlen), 0, dtype=torch.long)
        for i, x in enumerate(xs):
            padded[i, : len(x)] = x
        y = torch.stack(ys, dim=0)
        return padded, torch.tensor(lengths, dtype=torch.long), y

class MeanPoolEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        mask = (x != 0).unsqueeze(-1).float()
        summed = (e * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

class TextConceptHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_out),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

class TextConceptModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = MeanPoolEncoder(vocab_size, embed_dim)
        self.head = TextConceptHead(embed_dim, hidden_dim, n_out, dropout=dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, lengths)
        return self.head(h)

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
        output_mode: str = "hard",  # "hard" or "soft"
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(embedding_model=None, concept_layers=None)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.use_bigrams = bool(use_bigrams)
        self.max_len = int(max_len)
        self.min_freq = int(min_freq)
        self.max_vocab = max_vocab
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.vocab: Optional[Vocab] = None
        self.model: Optional[TextConceptModel] = None
        self._n_concepts: Optional[int] = None
        self.pos_weight_cfg: Union[None, str, np.ndarray] = pos_weight
        om = (output_mode or "hard").strip().lower()
        self.output_mode = "soft" if om == "soft" else "hard"

    def _to_text_list(self, dataset: ConceptDatasetSample) -> List[str]:
        X = getattr(dataset, "X", None)
        if X is None:
            raise ValueError("Dataset missing X")
        return [str(v) for v in list(X)]

    def _build_datasets(self, train_dataset: ConceptDatasetSample, valid_dataset: Optional[ConceptDatasetSample]):
        X_train = self._to_text_list(train_dataset)
        y_train = np.asarray(train_dataset.C, dtype=np.float32)
        vocab = self.vocab or None
        ds_train = TextMultiLabelDataset(
            texts=X_train,
            labels=y_train,
            vocab=vocab,
            use_bigrams=self.use_bigrams,
            max_len=self.max_len,
            min_freq=self.min_freq,
            max_size=self.max_vocab,
        )
        self.vocab = ds_train.vocab
        if valid_dataset is not None:
            X_valid = self._to_text_list(valid_dataset)
            y_valid = np.asarray(valid_dataset.C, dtype=np.float32)
            ds_valid = TextMultiLabelDataset(
                texts=X_valid,
                labels=y_valid,
                vocab=self.vocab,
                use_bigrams=self.use_bigrams,
                max_len=self.max_len,
            )
        else:
            ds_valid = TextMultiLabelDataset(texts=[], labels=np.zeros((0, y_train.shape[1]), dtype=np.float32), vocab=self.vocab)
        return ds_train, ds_valid

    def _make_loaders(self, ds_train: Dataset, ds_valid: Dataset):
        pin = self.device in {"cuda", "mps"}
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, collate_fn=TextMultiLabelDataset.collate, pin_memory=pin, num_workers=0)
        valid_loader = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=False, collate_fn=TextMultiLabelDataset.collate, pin_memory=pin, num_workers=0)
        return train_loader, valid_loader

    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: Optional[ConceptDatasetSample] = None,
        embed_params: Optional[dict] = None,
        l1_size: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs,
    ) -> None:
        ds_train, ds_valid = self._build_datasets(train_dataset, valid_dataset)
        n_out = int(ds_train.labels.shape[1])
        self._n_concepts = n_out
        self.model = TextConceptModel(vocab_size=len(self.vocab.itos), embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, n_out=n_out, dropout=self.dropout).to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if isinstance(self.pos_weight_cfg, str) and self.pos_weight_cfg.lower() == "auto":
            y = torch.tensor(ds_train.labels, dtype=torch.float32)
            pos = y.sum(dim=0)
            neg = y.shape[0] - pos
            pw = (neg / pos.clamp_min(1.0)).clamp_max(100.0)
            pos_weight = pw.to(self.device)
        elif isinstance(self.pos_weight_cfg, np.ndarray):
            pos_weight = torch.tensor(self.pos_weight_cfg, dtype=torch.float32, device=self.device)
        else:
            pos_weight = None

        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loader, valid_loader = self._make_loaders(ds_train, ds_valid)

        def _epoch(loader, train: bool):
            self.model.train() if train else self.model.eval()
            total = 0.0
            with torch.set_grad_enabled(train):
                for x, lengths, y in loader:
                    x = x.to(self.device, non_blocking=True)
                    lengths = lengths.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    logits = self.model(x, lengths)
                    loss = crit(logits, y)
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        opt.step()
                    total += float(loss.item()) * y.size(0)
            denom = max(1, len(loader.dataset))
            return total / denom

        for _ in range(self.epochs):
            _ = _epoch(train_loader, True)
            _ = _epoch(valid_loader, False)

        self.embedding_model = self.model.encoder

    def predict(self, dataset: ConceptDatasetSample, emebed_params: Optional[dict] = None, **kwargs) -> np.ndarray:
        if self.model is None or self.vocab is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        texts = self._to_text_list(dataset)
        labels_dummy = np.zeros((len(texts), self._n_concepts), dtype=np.float32)
        ds = TextMultiLabelDataset(texts=texts, labels=labels_dummy, vocab=self.vocab, use_bigrams=self.use_bigrams, max_len=self.max_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=TextMultiLabelDataset.collate, pin_memory=(self.device in {"cuda", "mps"}), num_workers=0)
        self.model.eval()
        outs: List[np.ndarray] = []
        with torch.no_grad():
            for x, lengths, _ in loader:
                x = x.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                logits = self.model(x, lengths)
                prob = torch.sigmoid(logits).cpu().numpy()
                outs.append(prob)
        prob_all = np.vstack(outs) if outs else np.zeros((0, self._n_concepts), dtype=np.float32)
        if self.output_mode == "hard":
            return (prob_all >= 0.5).astype(np.float32)
        return prob_all

    @property
    def n_concepts(self) -> int:
        if self._n_concepts is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return int(self._n_concepts)
