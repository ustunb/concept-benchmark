from __future__ import annotations
import re
from typing import Iterable, List, Dict, Optional, Tuple
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

def _make_bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]

class Vocab:
    def __init__(self, min_freq: int = 1, max_size: Optional[int] = None):
        self.itos: List[str] = ["<pad>", "<unk>"]
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self.min_freq = min_freq
        self.max_size = max_size

    def build(self, corpus: Iterable[List[str]]):
        from collections import Counter
        cnt = Counter()
        for toks in corpus:
            cnt.update(toks)
        words = [w for w, f in cnt.items() if f >= self.min_freq]
        words.sort(key=lambda w: (-cnt[w], w))
        if self.max_size is not None:
            words = words[: max(0, self.max_size - len(self.itos))]
        for w in words:
            if w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in toks]

class TextMultiLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, vocab: Optional[Vocab] = None, use_bigrams: bool = True, max_len: int = 128, min_freq: int = 1, max_size: Optional[int] = 40000):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.use_bigrams = use_bigrams
        self.max_len = max_len
        tokenized = []
        for s in texts:
            toks = _tok(str(s))
            if use_bigrams:
                toks = toks + _make_bigrams(toks)
            tokenized.append(toks)
        if vocab is None:
            vocab = Vocab(min_freq=min_freq, max_size=max_size)
            vocab.build(tokenized)
        self.vocab = vocab
        self.seqs = [self.vocab.encode(t)[:max_len] for t in tokenized]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.seqs[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def collate(batch):
        xs, ys = zip(*batch)
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
        denom = mask.sum(dim=1).clamp_min(1.)
        return summed / denom

class TextConceptHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_out),
        )

    def forward(self, z):
        return self.net(z)

class TextConceptModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = MeanPoolEncoder(vocab_size, embed_dim)
        self.head = TextConceptHead(embed_dim, hidden_dim, n_out, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, lengths)
        logits = self.head(z)
        return logits

class TextConceptDetector(ConceptDetector):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 192, epochs: int = 6, batch_size: int = 64, lr: float = 2e-3, weight_decay: float = 1e-2, dropout: float = 0.1, use_bigrams: bool = True, max_len: int = 128, min_freq: int = 1, max_vocab: Optional[int] = 40000, device: Optional[str] = None, **kwargs) -> None:
        super().__init__(embedding_model=None, concept_layers=None)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_bigrams = use_bigrams
        self.max_len = max_len
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
        self.vocab: Optional[Vocab] = None
        self.model: Optional[TextConceptModel] = None
        self._n_concepts: Optional[int] = None

    def _to_text_list(self, X) -> List[str]:
        if isinstance(X, list) and (len(X) == 0 or isinstance(X[0], str)):
            return [str(x) for x in X]
        try:
            X_list = X.tolist() if hasattr(X, "tolist") else list(X)
        except Exception as e:
            raise AssertionError(f"TextConceptDetector expected an iterable of texts; got type={type(X)} with error: {e}")
        return [str(x) for x in X_list]

    def _build_datasets(self, train: ConceptDatasetSample, valid: ConceptDatasetSample):
        train_texts = self._to_text_list(train.X)
        valid_texts = self._to_text_list(valid.X)
        ds_train = TextMultiLabelDataset(texts=train_texts, labels=train.C.astype(np.float32), vocab=None, use_bigrams=self.use_bigrams, max_len=self.max_len, min_freq=self.min_freq, max_size=self.max_vocab)
        self.vocab = ds_train.vocab
        ds_valid = TextMultiLabelDataset(texts=valid_texts, labels=valid.C.astype(np.float32), vocab=self.vocab, use_bigrams=self.use_bigrams, max_len=self.max_len)
        return ds_train, ds_valid

    def _make_loaders(self, ds_train: Dataset, ds_valid: Dataset):
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=TextMultiLabelDataset.collate)
        valid_loader = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=TextMultiLabelDataset.collate)
        return train_loader, valid_loader

    def fit(self, train_dataset: ConceptDatasetSample, valid_dataset: ConceptDatasetSample, freeze: bool = True, embed_params: Optional[dict] = None, fit_params: Optional[dict] = None, l1_size: Optional[int] = None, n_jobs: Optional[int] = None, **kwargs) -> None:
        ds_train, ds_valid = self._build_datasets(train_dataset, valid_dataset)
        n_out = ds_train.labels.shape[1]
        self._n_concepts = int(n_out)
        self.model = TextConceptModel(vocab_size=len(self.vocab.itos), embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, n_out=n_out, dropout=self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.BCEWithLogitsLoss()
        train_loader, valid_loader = self._make_loaders(ds_train, ds_valid)

        def _epoch(loader, train: bool):
            self.model.train() if train else self.model.eval()
            total = 0.0
            with torch.set_grad_enabled(train):
                for x, lengths, y in loader:
                    x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                    logits = self.model(x, lengths)
                    loss = crit(logits, y)
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    total += float(loss.item()) * y.size(0)
            return total / max(1, len(loader.dataset))

        for _ in range(self.epochs):
            _epoch(train_loader, True)
            _ = _epoch(valid_loader, False)
        self.embedding_model = self.model.encoder

    def predict(self, dataset: ConceptDatasetSample, emebed_params: Optional[dict] = None, **kwargs) -> np.ndarray:
        if emebed_params is None:
            emebed_params = kwargs.get("embed_params")
        if self.model is None or self.vocab is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        n = len(dataset.X)
        k = self._n_concepts if self._n_concepts is not None else dataset.n_concepts
        dummy = np.zeros((n, k), dtype=np.float32)
        texts = self._to_text_list(dataset.X)
        tmp = TextMultiLabelDataset(texts=texts, labels=dummy, vocab=self.vocab, use_bigrams=self.use_bigrams,
                                    max_len=self.max_len)
        loader = DataLoader(tmp, batch_size=self.batch_size, shuffle=False, num_workers=0,
                            collate_fn=TextMultiLabelDataset.collate)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for x, lengths, _ in loader:
                x, lengths = x.to(self.device), lengths.to(self.device)
                logits = self.model(x, lengths)
                prob = torch.sigmoid(logits).cpu().numpy()
                outs.append(prob)
        return np.vstack(outs)

    @property
    def n_concepts(self) -> int:
        if self._n_concepts is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._n_concepts