import numpy as np
import pytest
import torch

from concept_benchmark.data import ConceptDatasetSample


# ---------- Basic behavior ----------
def test_len_repr_getitem_dtypes(tab_small):
    sample = tab_small.training  # ConceptDatasetSample
    assert isinstance(sample, ConceptDatasetSample)
    assert len(sample) == sample.n == tab_small.n
    repr(sample)  # smoke
    x, c, y = sample[0]

    assert isinstance(x, np.ndarray) and x.dtype == np.float32
    assert isinstance(c, np.ndarray) and c.dtype == np.int64
    assert isinstance(y, np.integer) and y.dtype == np.int64

    idx = [0, 2, 4]
    x, c, y = sample[idx]
    assert isinstance(x, np.ndarray) and x.dtype == np.float32
    assert isinstance(c, np.ndarray) and c.dtype == np.int64
    assert isinstance(y, np.ndarray) and y.dtype == np.int64
    assert x.shape[0] == c.shape[0] == y.shape[0] == len(idx)


# ---------- filter() ----------
def test_filter_slices_and_preserves_meta_and_transforms(tab_small):
    sample = tab_small.training
    n = sample.n
    idx = np.zeros(n, dtype=np.bool_)
    idx[: n // 2] = True
    sub = sample.filter(indices=idx)
    assert sub.n == idx.sum()
    np.testing.assert_array_equal(sub.X, sample.X[idx])
    np.testing.assert_array_equal(sub.C, sample.C[idx])
    np.testing.assert_array_equal(sub.y, sample.y[idx])
    assert sub.meta == sample.meta
    np.testing.assert_array_equal(sub.indices, idx)


def test_filter_rejects_bad_indices(tab_small):
    s = tab_small.training
    with pytest.raises(AssertionError):
        s.filter(indices=[True, False])  # not ndarray
    with pytest.raises(AssertionError):
        s.filter(indices=np.ones((s.n, 1), dtype=bool))  # wrong shape
    with pytest.raises(AssertionError):
        s.filter(indices=np.arange(s.n))  # ints not restricted to {0,1}


# ---------- DataLoader ----------
def test_loader_shapes_and_dtypes(tab_small):
    s = tab_small.training
    loader = s.loader(batch_size=4, shuffle=False, num_workers=0)
    x, c, y = next(iter(loader))
    # Collate converts numpy arrays to tensors
    assert isinstance(x, torch.Tensor) and x.dtype == torch.float32
    assert isinstance(c, torch.Tensor) and c.dtype == torch.int64
    assert isinstance(y, torch.Tensor) and y.dtype == torch.int64
    assert x.shape[0] == c.shape[0] == y.shape[0] == 4
    assert x.ndim == 2 and c.ndim == 2 and y.ndim == 1


# ---------- Transforms ----------
def test_transforms_applied_and_types_respected(tab_small):
    # Transforms return tensors to bypass numpy recast
    def tx(x):
        return torch.as_tensor(x, dtype=torch.float32) * 2

    def tc(c):
        return torch.as_tensor(c, dtype=torch.float32) + 1

    def ty(y):
        return torch.as_tensor(y, dtype=torch.int64)

    s = ConceptDatasetSample(
        parent=tab_small,
        X=tab_small.X,
        C=tab_small.C,
        y=tab_small.y,
        meta=tab_small._full.meta,
        transform_x=tx,
        transform_c=tc,
        transform_y=ty,
    )

    x, c, y = s[0]
    assert isinstance(x, torch.Tensor) and x.dtype == torch.float32
    assert isinstance(c, torch.Tensor) and c.dtype == torch.float32
    assert isinstance(y, torch.Tensor) and y.dtype == torch.int64
    np.testing.assert_allclose(x.numpy(), tab_small.X[0] * 2)
    np.testing.assert_allclose(c.numpy(), tab_small.C[0] + 1)
    np.testing.assert_equal(y.item(), int(tab_small.y[0]))


# ---------- Embed path on sample ----------
class MeanEmbedder(torch.nn.Module):
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            z = x.float().view(x.shape[0], -1).mean(dim=1, keepdim=True)
            return torch.cat([z, z], dim=1)
        arr = np.asarray(x)
        z = arr.reshape(arr.shape[0], -1).mean(axis=1, keepdims=True)
        return np.concatenate([z, z], axis=1)


def test_embed_returns_tabular_and_preserves_indices(tab_small):
    s = tab_small.training
    emb = s.embed(MeanEmbedder(), batch_size=8, shuffle=False, device="cpu")
    assert isinstance(emb, ConceptDatasetSample)
    assert emb.meta.get("data_type") == "tabular"
    assert isinstance(emb.X, np.ndarray) and emb.X.ndim == 2 and emb.X.shape[1] == 2
    np.testing.assert_array_equal(emb.C, s.C)
    np.testing.assert_array_equal(emb.y, s.y)
    np.testing.assert_array_equal(emb.indices, s.indices)