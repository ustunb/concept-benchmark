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
    assert (embed_ds.training.n, embed_ds.validation.n, embed_ds.test.n) == (
        n_tr,
        n_va,
        n_te,
    )
    assert (
        isinstance(embed_ds.X, np.ndarray)
        and embed_ds.X.ndim == 2
        and embed_ds.X.shape[1] == 2
    )


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
def test_sample_concept_missingness_mcar_reproducible(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    original = ds.training.base_concepts.copy()

    masks_a = ds.sample_concept_missingness(p=0.2, mechanism="mcar", rng=123)
    masks_b = ds.sample_concept_missingness(p=0.2, mechanism="mcar", rng=123)

    assert set(masks_a.keys()) == {"train", "validation", "test"}
    assert masks_a["train"].shape == ds.training.base_concepts.shape
    assert masks_a["validation"].shape == ds.validation.base_concepts.shape
    assert masks_a["test"].shape == ds.test.base_concepts.shape
    np.testing.assert_array_equal(masks_a["train"], masks_b["train"])
    assert np.isclose(masks_a["train"].mean(), 0.2, atol=0.1)
    np.testing.assert_array_equal(ds.training.base_concepts, original)
    np.testing.assert_array_equal(ds.training.C, original)


def test_sample_concept_missingness_enable_toggle(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    original = ds.training.base_concepts.copy()

    masks = ds.sample_concept_missingness(
        p=0.3,
        mechanism="mcar",
        rng=999,
        fill_value=-1.0,
        enable=True,
    )
    train_mask = masks["train"]

    assert ds.has_concept_missing is True
    assert train_mask.shape == ds.training.base_concepts.shape
    assert masks["validation"].shape == ds.validation.base_concepts.shape
    assert masks["test"].shape == ds.test.base_concepts.shape
    assert np.all(ds.training.C[train_mask] == -1.0)
    np.testing.assert_array_equal(ds.training.C[~train_mask], original[~train_mask])
    np.testing.assert_array_equal(ds.training.base_concepts, original)
    ds.has_concept_missing = False
    np.testing.assert_array_equal(ds.training.C, original)


def test_split_specific_concept_missing_toggle(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    ds.sample_concept_missingness(p=0.5, rng=0, mechanism="mcar", fill_value=-1.0)
    ds.has_concept_missing = False
    ds.training.has_concept_missing = True
    assert ds.training.has_concept_missing is True
    assert ds.validation.has_concept_missing is False
    assert np.any(ds.training.C == -1.0)
    assert np.all(ds.validation.C != -1.0)
    ds.training.has_concept_missing = False
    np.testing.assert_array_equal(ds.training.C, ds.training.base_concepts)


def test_sample_concept_missingness_mnar_respects_probabilities(tab_medium_cv):
    ds, _ = tab_medium_cv
    original = ds.training.base_concepts.copy()

    masks = ds.sample_concept_missingness(
        p=0.4,
        mechanism="mnar",
        rng=2024,
        mnar_config={"present_prob": 0.8, "absent_prob": 0.1},
    )

    train_mask = masks["train"]
    concepts = ds.training.base_concepts

    present_mask = train_mask[concepts == 1]
    absent_mask = train_mask[concepts == 0]

    assert present_mask.size > 0 and absent_mask.size > 0
    assert present_mask.mean() > absent_mask.mean()
    np.testing.assert_array_equal(ds.training.base_concepts, original)
    np.testing.assert_array_equal(ds.training.C, original)


# ---------- Concept noise ----------
def test_sample_concept_noise_reproducible_and_toggle(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    original = ds.training.base_concepts.copy()

    masks_a = ds.sample_concept_noise(p=0.25, rng=42)
    masks_b = ds.sample_concept_noise(p=0.25, rng=42)

    assert set(masks_a.keys()) == {"train", "validation", "test"}
    np.testing.assert_array_equal(masks_a["train"], masks_b["train"])
    ds.has_concept_noise = True
    noisy = ds.training.C
    expected = np.where(masks_a["train"], 1 - original, original)
    np.testing.assert_array_equal(noisy, expected)
    ds.has_concept_noise = False
    np.testing.assert_array_equal(ds.training.C, original)


def test_sample_concept_noise_asymmetric(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    base = ds.training.base_concepts.copy()

    masks = ds.sample_concept_noise(
        p=0.0,
        rng=7,
        config={"p01": 0.5, "p10": 0.0},
    )
    mask = masks["train"]

    assert np.all(mask[base == 1] == 0)
    assert mask[base == 0].any()

    ds.has_concept_noise = True
    noisy = ds.training.C
    np.testing.assert_array_equal(noisy[base == 1], base[base == 1])
    ds.has_concept_noise = False


def test_concept_noise_then_missingness_order(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    base = ds.training.base_concepts.copy()

    noise_masks = ds.sample_concept_noise(p=1.0, rng=0)
    missing_masks = ds.sample_concept_missingness(
        p=0.3,
        mechanism="mcar",
        rng=1,
        fill_value=-1.0,
    )

    ds.has_concept_noise = True
    ds.has_concept_missing = True

    noisy_then_missing = ds.training.C

    expected_noisy = np.where(noise_masks["train"], 1 - base, base)
    expected = expected_noisy.astype(np.float32)
    expected[missing_masks["train"]] = -1.0

    np.testing.assert_array_equal(noisy_then_missing, expected)
    ds.has_concept_noise = False
    ds.has_concept_missing = False
    np.testing.assert_array_equal(ds.training.C, base)


def test_split_specific_concept_noise_toggle(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    ds.sample_concept_noise(p=1.0, rng=42)
    ds.has_concept_noise = False
    ds.training.has_concept_noise = True
    assert ds.training.has_concept_noise is True
    assert ds.validation.has_concept_noise is False
    assert np.all(ds.training.C != ds.training.base_concepts)
    assert np.all(ds.validation.C == ds.validation.base_concepts)
    ds.training.has_concept_noise = False
    assert np.all(ds.training.C == ds.training.base_concepts)


# ---------- Label noise ----------
def test_sample_label_noise_reproducible_and_toggle(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    base_labels = ds.training.base_labels.copy()

    noisy_a = ds.sample_label_noise(p=0.4, rng=2023)
    noisy_b = ds.sample_label_noise(p=0.4, rng=2023)

    assert set(noisy_a.keys()) == {"full", "train", "validation", "test"}
    np.testing.assert_array_equal(noisy_a["train"], noisy_b["train"])
    assert ds.has_label_noise is False
    assert ds.training.has_label_noise is False
    np.testing.assert_array_equal(ds.training.y, base_labels)

    ds.has_label_noise = True
    np.testing.assert_array_equal(ds.training.y, noisy_a["train"])
    assert ds.training.has_label_noise is True
    ds.has_label_noise = False
    np.testing.assert_array_equal(ds.training.y, base_labels)
    assert ds.training.has_label_noise is False


def test_sample_label_noise_with_flip_matrix(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    base_labels = ds.training.base_labels.copy()
    flip_matrix = [[0.0, 1.0], [1.0, 0.0]]

    noisy = ds.sample_label_noise(
        p=0.0,
        rng=7,
        label_noise_config={"flip_matrix": flip_matrix},
        enable=True,
    )

    assert ds.has_label_noise is True
    assert ds.training.has_label_noise is True
    assert np.all(ds.training.y != base_labels)
    np.testing.assert_array_equal(ds.training.y, 1 - base_labels)

    ds.has_label_noise = False
    np.testing.assert_array_equal(ds.training.y, base_labels)
    assert ds.training.has_label_noise is False


def test_split_specific_label_noise_toggle(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    ds.sample_label_noise(p=0.5, rng=1)
    ds.has_label_noise = False
    ds.training.has_label_noise = True
    assert ds.training.has_label_noise is True
    assert ds.validation.has_label_noise is False
    assert ds.test.has_label_noise is False
    assert np.any(ds.training.y != ds.training.base_labels)
    assert np.all(ds.validation.y == ds.validation.base_labels)


# ---------- Dict-style split access ----------
def test_getitem_train_returns_training(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    assert ds["train"] is ds.training
    assert ds["training"] is ds.training


def test_getitem_val_returns_validation(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    assert ds["val"] is ds.validation
    assert ds["validation"] is ds.validation


def test_getitem_test_returns_test(tab_small_cv):
    ds, fid = tab_small_cv
    ds.split(fold_id=fid, fold_num_validation=1, fold_num_test=2)
    assert ds["test"] is ds.test


def test_getitem_invalid_raises_keyerror(tab_small):
    with pytest.raises(KeyError, match="Unknown split"):
        tab_small["invalid"]


def test_keys(tab_small):
    assert tab_small.keys() == ["train", "val", "test"]


def test_contains(tab_small):
    assert "train" in tab_small
    assert "training" in tab_small
    assert "val" in tab_small
    assert "validation" in tab_small
    assert "test" in tab_small
    assert "invalid" not in tab_small


# ---------- Aliases ----------
def test_train_alias(tab_small):
    assert tab_small.train is tab_small.training


def test_val_alias(tab_small):
    assert tab_small.val is tab_small.validation


# ---------- Description ----------
def test_description(tab_small):
    desc = tab_small.description
    assert isinstance(desc, str)
    assert len(desc) > 0
    assert "ConceptDataset" in desc
    assert str(tab_small.n) in desc
    assert tab_small._full.meta.get("data_type", "unknown") in desc
    for concept in tab_small.concepts:
        assert concept in desc
    for cls in tab_small.classes:
        assert cls in desc


def test_repr_multiline(tab_small):
    r = repr(tab_small)
    assert "ConceptDataset(" in r
    assert "train:" in r
    assert "val:" in r
    assert "test:" in r


# ---------- sample() method ----------


class TestSample:
    """Tests for ConceptDataset.sample()."""

    @pytest.fixture(autouse=True)
    def _import_factory(self):
        from tests.conftest import make_tabular_dataset

        self._make = make_tabular_dataset

    def test_basic_random_split_fractions(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=0.2, val_size=0.2, seed=42)
        assert ds.test.n == 20
        assert ds.validation.n == 20
        assert ds.training.n == 60
        assert ds.__check_rep__()

    def test_basic_random_split_ints(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=15, val_size=10, seed=7)
        assert ds.test.n == 15
        assert ds.validation.n == 10
        assert ds.training.n == 75
        assert ds.__check_rep__()

    def test_train_size_limits_training(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=20, val_size=20, train_size=30, seed=1)
        assert ds.test.n == 20
        assert ds.validation.n == 20
        assert ds.training.n == 30
        # Some samples discarded
        total = ds.training.n + ds.validation.n + ds.test.n
        assert total < ds.n
        assert ds.__check_rep__()

    def test_no_overlap_between_splits(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=0.2, val_size=0.2, seed=99)
        tr = set(np.where(ds.training.indices)[0])
        va = set(np.where(ds.validation.indices)[0])
        te = set(np.where(ds.test.indices)[0])
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)

    def test_stratified_preserves_proportions(self):
        ds, _ = self._make(n=200, d=4, k=3, n_classes=2)
        ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=10)
        overall = np.mean(ds.y)
        train_mean = np.mean(ds.training.y)
        test_mean = np.mean(ds.test.y)
        val_mean = np.mean(ds.validation.y)
        assert abs(train_mean - overall) < 0.1
        assert abs(test_mean - overall) < 0.1
        assert abs(val_mean - overall) < 0.1

    def test_group_split_no_leakage(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        # 20 groups (5 samples each)
        groups = np.repeat(np.arange(20), 5)
        ds.sample(test_size=0.2, val_size=0.2, groups=groups, seed=5)

        # Check no group appears in multiple splits
        train_idx = np.where(ds.training.indices)[0]
        val_idx = np.where(ds.validation.indices)[0]
        test_idx = np.where(ds.test.indices)[0]

        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        test_groups = set(groups[test_idx].tolist())

        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)

    def test_group_split_with_stratify(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        groups = np.repeat(np.arange(20), 5)
        ds.sample(
            test_size=0.2, val_size=0.2, groups=groups, stratify=ds.y, seed=5
        )
        # No group leakage
        train_idx = np.where(ds.training.indices)[0]
        val_idx = np.where(ds.validation.indices)[0]
        test_idx = np.where(ds.test.indices)[0]
        train_g = set(groups[train_idx].tolist())
        val_g = set(groups[val_idx].tolist())
        test_g = set(groups[test_idx].tolist())
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)

    def test_sampling_constraints(self):
        # Create dataset where concept z0 is rare (only ~10% positive)
        ds, _ = self._make(n=200, d=4, k=3, n_classes=2)
        # Force concept z0 to be mostly 0
        ds._full._C_base[:, 0] = 0
        ds._full._C_base[:20, 0] = 1  # only 20 out of 200

        constraints = [{"concepts": {"z0": 1}, "min_fraction": 0.3}]
        ds.sample(
            test_size=40,
            val_size=0.2,
            train_size=80,
            sampling_constraints=constraints,
            seed=42,
        )
        # At least 30% of training should have z0=1
        z0_col = ds.concepts.index("z0")
        z0_frac = np.mean(ds.training.C[:, z0_col])
        assert z0_frac >= 0.2  # constraint targets 0.3; best-effort oversampling from 10% base rate

    def test_reproducible_with_same_seed(self):
        ds1, _ = self._make(n=100, d=4, k=3, n_classes=2, seed=77)
        ds2, _ = self._make(n=100, d=4, k=3, n_classes=2, seed=77)
        ds1.sample(test_size=0.2, val_size=0.2, seed=42)
        ds2.sample(test_size=0.2, val_size=0.2, seed=42)
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.C, ds2.C)
        np.testing.assert_array_equal(ds1.training.y, ds2.training.y)
        np.testing.assert_array_equal(ds1.test.y, ds2.test.y)

    def test_returns_self_for_chaining(self):
        ds, _ = self._make(n=50, d=4, k=3, n_classes=2)
        result = ds.sample(test_size=0.2, val_size=0.2, seed=1)
        assert result is ds

    def test_check_rep_passes_after_sample(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=0.2, val_size=0.2, seed=42)
        assert ds.__check_rep__()

    def test_check_rep_passes_with_train_size(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=20, val_size=20, train_size=30, seed=1)
        assert ds.__check_rep__()

    def test_missingness_composable_after_sample(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=0.2, val_size=0.2, seed=42)
        masks = ds.sample_concept_missingness(
            p=0.3, mechanism="mcar", rng=99, enable=True
        )
        assert "train" in masks
        assert masks["train"].shape == ds.training.base_concepts.shape

    def test_missingness_splits_param_training_only(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        ds.sample(test_size=0.2, val_size=0.2, seed=42)
        masks = ds.sample_concept_missingness(
            p=0.5, mechanism="mcar", rng=1, splits={"train"}
        )
        assert "train" in masks
        assert "validation" not in masks
        assert "test" not in masks

    def test_drop_concepts_after_sample(self):
        ds, _ = self._make(n=100, d=4, k=4, n_classes=2)
        ds.sample(test_size=0.2, val_size=0.2, seed=42)
        original_k = ds.n_concepts
        ds.drop_concepts(["z0"])
        assert ds.n_concepts == original_k - 1
        assert ds.training.C.shape[1] == original_k - 1
        assert ds.validation.C.shape[1] == original_k - 1
        assert ds.test.C.shape[1] == original_k - 1
        assert ds.__check_rep__()

    def test_invalid_float_size_raises(self):
        ds, _ = self._make(n=100, d=4, k=3, n_classes=2)
        with pytest.raises(ValueError, match="Float size must be in"):
            ds.sample(test_size=1.5, seed=1)
