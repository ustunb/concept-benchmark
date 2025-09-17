import numpy as np
import pytest
import torch

from concept_benchmark.data import ConceptDataset


# ---------- Helpers ----------
class MeanEmbedder(torch.nn.Module):
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            feat = x.float().view(x.shape[0], -1).mean(dim=1, keepdim=True)
            return torch.cat([feat, feat], dim=1)  # [B, 2]
        arr = np.asarray(x)
        feat = arr.reshape(arr.shape[0], -1).mean(axis=1, keepdims=True)
        return np.concatenate([feat, feat], axis=1)


# ---------- Constructor and basics ----------
def test_constructor_tabular_basics(tab_small):
    ds = tab_small
    assert isinstance(ds, ConceptDataset)
    assert ds.n == len(ds.X)
    assert ds.n_concepts == ds.C.shape[1]
    assert ds.n_classes == len(ds.classes)
    assert ds.training.n == ds.n and ds.validation.n == 0 and ds.test.n == 0
    repr(ds)  # smoke

def test_copy_equality_and_independence(tab_small):
    ds = tab_small
    cpy = ds.__copy__()
    assert cpy == ds
    # Mutate copy's data; original must not change and equality should break
    before = ds.X.copy()
    if isinstance(cpy.X, np.ndarray):
        cpy.X[0, 0] += 1.0
    else:
        # If X is not ndarray in future, replace with an equivalent mutation
        raise AssertionError("Unexpected X type in tabular dataset")
    assert cpy != ds
    np.testing.assert_array_equal(ds.X, before)


# ---------- CV indices and splitting ----------
def test_set_cvindices_and_split(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    n = ds.n
    tr = set(np.where(ds.training.indices)[0])
    va = set(np.where(ds.validation.indices)[0])
    te = set(np.where(ds.test.indices)[0])
    assert tr and va and te
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr | va | te) == n

def test_split_rejects_same_fold(tab_small_cv):
    ds, fid = tab_small_cv
    with pytest.raises(AssertionError):
        ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=1)


# ---------- Equality robustness ----------
def test_equality_changes_with_meta_change(tab_small):
    ds = tab_small
    cpy = ds.__copy__()
    cpy._full.meta = dict(cpy._full.meta)
    cpy._full.meta["classes"] = cpy._full.meta["classes"] + ["extra"]
    assert cpy != ds

def test_equality_changes_with_cvindices_change(tab_small_cv):
    ds, fid = tab_small_cv
    cpy = ds.__copy__()
    arr = cpy.cvindices[fid].copy()
    arr[0] = (arr[0] % max(arr)) + 1
    cpy._cvindices = {fid: arr}
    assert cpy != ds


# ---------- Embed path and split reapplication ----------
def test_embed_updates_full_and_preserves_splits(tab_medium_cv):
    ds, fid = tab_medium_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    n_tr, n_va, n_te = ds.training.n, ds.validation.n, ds.test.n
    model = MeanEmbedder()
    embed_ds = ds.embed(model, batch_size=4, shuffle=False, device="cpu", num_workers=0)
    assert ds._full.meta.get("data_type") == "tabular"
    assert (embed_ds.training.n, embed_ds.validation.n, embed_ds.test.n) == (n_tr, n_va, n_te)
    assert isinstance(embed_ds.X, np.ndarray) and embed_ds.X.ndim == 2 and embed_ds.X.shape[1] == 2


# ---------- Reset behavior ----------
def test_reset_restores_full_and_clears_splits(tab_medium_cv):
    ds, fid = tab_medium_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    assert ds.validation.n > 0 and ds.test.n > 0
    ds.reset()
    assert ds.training.n == ds.n and ds.validation.n == 0 and ds.test.n == 0


# ---------- Negative paths ----------
def test_split_requires_known_fold_id(tab_small):
    ds = tab_small
    with pytest.raises(AssertionError):
        ds.split(fold_id=None, fold_num_validation=1, fold_num_test=2)


# ---------- Missingness masking ----------
def test_mask_mcar_reproducible_and_shape(tab_small):
    ds = tab_small
    original = ds.training.C.copy()

    masks_a = ds.mask(p=0.2, mechanism="mcar", rng=123)
    masks_b = ds.mask(p=0.2, mechanism="mcar", rng=123)

    assert set(masks_a.keys()) == {"training", "validation", "test"}
    assert masks_a["training"].shape == ds.training.C.shape
    assert masks_a["validation"].shape[0] == 0
    assert masks_a["test"].shape[0] == 0
    np.testing.assert_array_equal(masks_a["training"], masks_b["training"])
    assert np.isclose(masks_a["training"].mean(), 0.2, atol=0.1)
    np.testing.assert_array_equal(ds.training.C, original)


def test_mask_apply_in_place_fill(tab_small):
    ds = tab_small
    original = ds.training.C.copy()

    masks = ds.mask(p=0.3, mechanism="mcar", rng=999, apply=True, fill_value=-1.0)
    train_mask = masks["training"]

    assert train_mask.shape == ds.training.C.shape
    assert np.all(ds.training.C[train_mask] == -1.0)
    np.testing.assert_array_equal(ds.training.C[~train_mask], original[~train_mask])


def test_mask_mnar_respects_probabilities(tab_medium_cv):
    ds, _ = tab_medium_cv
    original = ds.training.C.copy()

    masks = ds.mask(
        p=0.4,
        mechanism="mnar",
        rng=2024,
        mnar_config={"present_prob": 0.8, "absent_prob": 0.1},
    )

    train_mask = masks["training"]
    concepts = ds.training.C

    present_mask = train_mask[concepts == 1]
    absent_mask = train_mask[concepts == 0]

    assert present_mask.size > 0 and absent_mask.size > 0
    assert present_mask.mean() > absent_mask.mean()
    np.testing.assert_array_equal(ds.training.C, original)
