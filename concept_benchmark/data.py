from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Set
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .cv import generate_cvindices, validate_cvindices
from .helper.data_utils import (
    coerce_rng,
    sample_concept_noise_mask,
    sample_label_noise,
    sample_mcar_mask,
    sample_mnar_mask,
)


def _deep_equal(a, b) -> bool:
    """Type-aware deep equality for nested structures and array-like values.

    Handles dicts/lists/tuples/sets, numpy arrays/scalars, pandas objects,
    torch tensors, Paths, and falls back to safe equality (including
    array-like results via .all()).
    """
    if a is b:
        return True
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a.keys())

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))

    if isinstance(a, Set) and isinstance(b, Set):
        try:
            return a == b
        except Exception:
            return sorted(map(repr, a)) == sorted(map(repr, b))

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, np.generic) and isinstance(b, np.generic):
        return bool(a == b)

    if isinstance(a, (pd.DataFrame, pd.Series, pd.Index)) and isinstance(
        b, (pd.DataFrame, pd.Series, pd.Index)
    ):
        return a.equals(b)

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)

    if isinstance(a, Path) and isinstance(b, Path):
        try:
            return a.resolve() == b.resolve()
        except Exception:
            return str(a) == str(b)

    if callable(a) or callable(b):
        return a is b

    try:
        eq = a == b
    except Exception:
        return repr(a) == repr(b)
    else:
        if isinstance(eq, (bool, np.bool_)):
            return bool(eq)
        if hasattr(eq, "all"):
            try:
                return bool(eq.all())
            except Exception:
                pass
        return repr(a) == repr(b)


class ConceptDataset(object):
    SAMPLE_TYPES = ("training", "validation", "test")

    def __init__(
        self,
        X: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        meta: dict,
        cvindices: dict | None = None,
        transform: Callable | None = None,
        concept_transform: Callable | None = None,
        target_transform: Callable | None = None,
        concept_noise: bool = False,
        concept_missing: bool = False,
        label_noise: bool = False,
        **kwargs,
    ) -> None:
        """ConceptDataset

        Args:
            X (np.ndarray): Feature matrix. \
                For image data, this should be an array of image file paths.
            C (np.ndarray): Concept matrix. \
                Should be of shape (n_samples, n_concepts) with binary values (0 or 1).
            y (np.ndarray): Label vector. \
                Should be of shape (n_samples,) with integer class labels.
            meta (dict): Metadata dictionary containing:
                - 'classes': List of class names (in order of labels in y).
                - 'concepts': List of concept names (in order of columns in C).
                - 'data_type': Type of data ('image', 'tabular', etc.).
            cvindices (dict, optional): Cross-validation indices. \
                Defaults to None.
            transform (Callable, optional): Transformation function for features. \
                Defaults to None.
            concept_transform (Callable, optional): Transformation function for concepts. \
                Defaults to None.
            target_transform (Callable, optional): Transformation function for labels. \
                Defaults to None.
            concept_noise (bool, optional): Whether concept noise is enabled by default.
            concept_missing (bool, optional): Whether concept missingness is enabled by default.
            label_noise (bool, optional): Whether label noise is enabled by default.
            **kwargs: Additional keyword arguments.
        """
        self._init_kwargs = dict(kwargs)
        self._concept_noise = bool(concept_noise)
        self._concept_missing = bool(concept_missing)
        self._label_noise = bool(label_noise)
        self._init_kwargs.update(
            concept_noise=self._concept_noise,
            concept_missing=self._concept_missing,
            label_noise=self._label_noise,
        )

        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except Exception as e:
                raise ValueError(f"cannot convert X to np.ndarray: {e}")
        
        if not isinstance(C, np.ndarray):
            try:
                C = np.asarray(C)
            except Exception as e:
                raise ValueError(f"cannot convert C to np.ndarray: {e}")
    
        if not isinstance(y, np.ndarray):
            try:
                y = np.asarray(y)
            except Exception as e:
                raise ValueError(f"cannot convert y to np.ndarray: {e}")

        if meta.get("data_type") == "image":
            # do not cast X
            SampleClass = ConceptImageDatasetSample
        elif meta.get("data_type") == "text":
            SampleClass = ConceptDatasetSample
            X = X.astype(object)
        else:
            SampleClass = ConceptDatasetSample
            X = X.astype(np.float32)

        C = C.astype(np.int8)
        y = y.astype(np.int32)

        self._full = SampleClass(
            parent=self,
            X=X,
            C=C,
            y=y,
            meta=meta,
            transform=transform,
            concept_transform=concept_transform,
            target_transform=target_transform,
            concept_noise=self._concept_noise,
            concept_missing=self._concept_missing,
            label_noise=self._label_noise,
            **kwargs,
        )

        self._cvindices = cvindices
        self.reset()

    def drop_concepts(self, concepts_to_drop):
        """
        Drop specified concepts from the dataset.

        Args:
            concepts_to_drop (list of str): List of concept names to drop.
        """
        if not isinstance(concepts_to_drop, (list, tuple, set)):
            raise ValueError("concepts_to_drop should be a list, tuple, or set of strings")
        concepts_to_drop = set(concepts_to_drop)
        existing_concepts = set(self.concepts)
        invalid_concepts = concepts_to_drop - existing_concepts
        if invalid_concepts:
            raise ValueError(f"Concepts not found in dataset: {invalid_concepts}")

        keep_indices = [i for i, c in enumerate(self.concepts) if c not in concepts_to_drop]
        if not keep_indices:
            raise ValueError("Cannot drop all concepts; at least one must remain.")

        # Update concept matrix and metadata
        self._full._C_base = self._full._C_base[:, keep_indices]
        self._full.meta["concepts"] = [self.concepts[i] for i in keep_indices]
        self._full.concepts = self._full.meta["concepts"]

        # Update all samples
        for sample in self._iter_samples():
            sample.concepts = sample.meta["concepts"]

        assert self.__check_rep__()

    def reset(self):
        """
        initialize data object to a state before CV
        :return:
        """
        self._fold_id = None
        self._fold_number_range = []
        self._fold_num_test = 0
        self._fold_num_validation = 0
        self._fold_num_range = 0
        self.training = self._full
        self.validation = self._full.filter(indices=np.zeros(self.n, dtype=np.bool_))
        self.test = self._full.filter(indices=np.zeros(self.n, dtype=np.bool_))
        self._apply_noise_settings()
        assert self.__check_rep__()

    def _iter_samples(self):
        seen = set()
        for sample in (
            getattr(self, "_full", None),
            getattr(self, "training", None),
            getattr(self, "validation", None),
            getattr(self, "test", None),
        ):
            if sample is None:
                continue
            sid = id(sample)
            if sid in seen:
                continue
            seen.add(sid)
            yield sample

    def _apply_noise_settings(self):
        for sample in self._iter_samples():
            sample.concept_noise = self._concept_noise
            sample.concept_missing = self._concept_missing
            sample.label_noise = self._label_noise

    #### built-ins ####

    def __check_rep__(self):
        # check complete dataset
        assert self._full.__check_rep__()

        # check folds
        if self._cvindices is not None:
            validate_cvindices(self._cvindices)

        if self._fold_id is not None:
            assert self._cvindices is not None

        # check subsamples
        n_total = 0
        for sample_name in self.SAMPLE_TYPES:
            if hasattr(self, sample_name):
                sample = getattr(self, sample_name)
                assert sample.__check_rep__()
                n_total += sample.n

        assert self.n == n_total

        return True

    def __eq__(self, other):
        def _cv_equal(a, b):
            if (a is None) != (b is None):
                return False
            if a is None and b is None:
                return True
            if set(a.keys()) != set(b.keys()):
                return False
            for k in a.keys():
                if not np.array_equal(a[k], b[k]):
                    return False
            return True

        chk = (
            (self._full == other._full)
            and _cv_equal(self.cvindices, other.cvindices)
            and (self._fold_id == other._fold_id)
            and (self._fold_num_validation == other._fold_num_validation)
            and (self._fold_num_test == other._fold_num_test)
        )

        return chk

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"ConceptDataset<n={self.n}, n_concepts={self.n_concepts}, n_classes={self.n_classes}, data_type={self._full.meta.get('data_type')}, splits={{train:{getattr(self, 'training', None).n if hasattr(self, 'training') else 0}, val:{getattr(self, 'validation', None).n if hasattr(self, 'validation') else 0}, test:{getattr(self, 'test', None).n if hasattr(self, 'test') else 0}}}>"

    def __copy__(self):
        cpy = ConceptDataset(
            X=self.X,
            C=self._full.base_concepts.copy(),
            y=self._full.base_labels.copy(),
            meta=self._full.meta,
            cvindices=self._cvindices,
            **self._init_kwargs,
        )
        cpy.concept_noise = self.concept_noise
        cpy.concept_missing = self.concept_missing
        cpy.label_noise = self.label_noise

        return cpy

    #### INSTANCE VARIABLES
    @property
    def classes(self):
        return self._full.classes

    @property
    def concepts(self):
        return self._full.concepts

    @property
    def n(self):
        """number of examples in full dataset"""
        return self._full.n

    @property
    def n_concepts(self):
        return self._full.n_concepts

    @property
    def n_classes(self):
        return self._full.n_classes

    @property
    def X(self):
        """feature matrix"""
        return self._full.X

    @property
    def C(self):
        return self._full.C

    @property
    def y(self):
        """label vector"""
        return self._full.y

    @property
    def meta(self):
        return self._full.meta

    @property
    def transform(self):
        return self._full.transform

    @property
    def concept_transform(self):
        return self._full.concept_transform

    @property
    def target_transform(self):
        return self._full.target_transform

    @transform.setter
    def transform(self, transform):
        self._full.transform = transform

    @concept_transform.setter
    def concept_transform(self, concept_transform):
        self._full.concept_transform = concept_transform

    @target_transform.setter
    def target_transform(self, target_transform):
        self._full.target_transform = target_transform

    @property
    def concept_noise(self) -> bool:
        return self._concept_noise

    @concept_noise.setter
    def concept_noise(self, value: bool) -> None:
        self._concept_noise = bool(value)
        self._apply_noise_settings()

    @property
    def concept_missing(self) -> bool:
        return self._concept_missing

    @concept_missing.setter
    def concept_missing(self, value: bool) -> None:
        self._concept_missing = bool(value)
        self._apply_noise_settings()

    @property
    def label_noise(self) -> bool:
        return self._label_noise

    @label_noise.setter
    def label_noise(self, value: bool) -> None:
        self._label_noise = bool(value)
        self._apply_noise_settings()

    #### cross validation ####
    @property
    def cvindices(self):
        return self._cvindices

    @cvindices.setter
    def cvindices(self, cvindices):
        self._cvindices = validate_cvindices(cvindices)

    @property
    def fold_id(self):
        """string representing the indices of cross-validation folds
        K05N01 = 5-fold CV – 1st replicate
        K05N02 = 5-fold CV – 2nd replicate (in case you want to run 5-fold CV one more time)
        K10N01 = 10-fold CV – 1st replicate
        """
        return self._fold_id

    @fold_id.setter
    def fold_id(self, fold_id):
        assert self._cvindices is not None, (
            "cannot set fold_id on a BinaryClassificationDataset without cvindices"
        )
        assert isinstance(fold_id, str), f"fold_id={fold_id} should be string"
        assert fold_id in self.cvindices, (
            f"cvindices does not contain fols for fold_id=`{fold_id}`"
        )
        self._fold_id = str(fold_id)
        self._fold_number_range = np.unique(self.folds).tolist()

    @property
    def folds(self):
        """integer array showing the fold number of each sample in the full dataset"""
        return self._cvindices.get(self._fold_id)

    @property
    def fold_number_range(self):
        """range of all possible training folds"""
        return self._fold_number_range

    @property
    def fold_num_validation(self):
        """integer from 1 to K representing the validation fold"""
        return self._fold_num_validation

    @property
    def fold_num_test(self):
        """integer from 1 to K representing the test fold"""
        return self._fold_num_test

    def split(self, fold_id, fold_num_validation=None, fold_num_test=None):
        """
        :param fold_id:
        :param fold_num_validation: fold to use as a validation set
        :param fold_num_test: fold to use as a hold-out test set
        :return:
        """

        if fold_id is not None:
            self.fold_id = fold_id
        else:
            assert self.fold_id is not None

        # parse fold numbers
        if fold_num_validation is not None and fold_num_test is not None:
            assert int(fold_num_test) != int(fold_num_validation)

        if fold_num_validation is not None:
            fold_num_validation = int(fold_num_validation)
            assert fold_num_validation in self._fold_number_range
            self._fold_num_validation = fold_num_validation

        if fold_num_test is not None:
            fold_num_test = int(fold_num_test)
            assert fold_num_test in self._fold_number_range
            self._fold_num_test = fold_num_test

        # update subsamples
        self.training = self._full.filter(
            indices=np.isin(
                self.folds, [self.fold_num_validation, self.fold_num_test], invert=True
            )
        )
        self.validation = self._full.filter(
            indices=np.isin(self.folds, self.fold_num_validation)
        )
        self.test = self._full.filter(indices=np.isin(self.folds, self.fold_num_test))
        self._apply_noise_settings()
        return

    def generate_cvindices(
        self,
        strata=None,
        total_folds_for_cv=[1, 3, 4, 5],
        total_folds_for_inner_cv=[],
        replicates=3,
        seed=None,
    ):
        """
        :param strata:
        :param total_folds_for_cv:
        :param total_folds_for_inner_cv:
        :param replicates:
        :param seed:
        :return:
        """
        indices = generate_cvindices(
            n_samples=self.n if strata is None else None,
            strata=strata,
            total_folds_for_cv=total_folds_for_cv,
            total_folds_for_inner_cv=total_folds_for_inner_cv,
            replicates=replicates,
            seed=seed,
        )
        self.cvindices = indices

    def embed(self, model, batch_size=32, shuffle=False, device="cpu", **kwargs):
        """
        Embed the dataset using a given model and return a new dataset
        instance without modifying the current one.

        Parameters:
        - model: A model that can embed the dataset.

        Returns:
        - ConceptDataset: a new dataset whose features are the embedded
          representations. Cross-validation splits are preserved if present.
        """
        # Compute embedded representation for the full dataset sample
        embedded_full = self._full.embed(
            model, batch_size=batch_size, shuffle=shuffle, device=device, **kwargs
        )

        # Create a new ConceptDataset using the embedded features while
        # preserving metadata and CV indices from the original dataset.
        new_ds = ConceptDataset(
            X=embedded_full.X,
            C=embedded_full.C,
            y=embedded_full.y,
            meta=embedded_full.meta,
            cvindices=self._cvindices,
            **self._init_kwargs,
        )

        # Re-apply existing split configuration on the new dataset, if any.
        if self.fold_id is not None:
            new_ds.split(
                fold_id=self.fold_id,
                fold_num_validation=self.fold_num_validation,
                fold_num_test=self.fold_num_test,
            )

        return new_ds

    def sample_concept_missingness(
        self,
        *,
        p: float = 0.1,
        mechanism: str = "mcar",
        rng: np.random.Generator | int | None = None,
        mnar_config: Mapping[str, object] | None = None,
        fill_value: float = np.nan,
        enable: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample concept-level missingness masks.

        Args:
            p: Baseline prevalence of missingness.
            mechanism: Missingness mechanism, either ``"mcar"`` or ``"mnar"``.
            rng: Optional ``np.random.Generator`` or int seed for reproducibility.
            mnar_config: Optional configuration dict used when ``mechanism="mnar"``.
                Accepted keys:
                    - ``present_prob`` / ``absent_prob``: scalar or per-concept
                      probabilities (length ``n_concepts``) applied when the
                      observed concept value is 1 or 0 respectively.
                    - ``prob_matrix``: full matrix of probabilities overriding the
                      per-concept values. Shape must match the concept matrix.
            fill_value: Value used when applying the mask (via the
                ``concept_missing`` toggle).
            enable: If provided, sets ``self.concept_missing`` to the boolean value
                after sampling. Defaults to ``None`` (no change).

        Returns:
            A dictionary mapping split name (``"training"``, ``"validation"``,
            ``"test"``) to the sampled boolean mask for that split.
        """

        mechanism_key = mechanism.lower()
        if mechanism_key not in {"mcar", "mnar"}:
            raise ValueError("mechanism must be either 'mcar' or 'mnar'")

        rng_generated = coerce_rng(rng)

        if enable is not None:
            self.concept_missing = bool(enable)

        masks: dict[str, np.ndarray] = {}
        splits = {
            "training": self.training,
            "validation": getattr(self, "validation", None),
            "test": getattr(self, "test", None),
        }

        for split_name, sample in splits.items():
            if sample is None or sample.base_concepts is None:
                continue

            if sample.base_concepts.size == 0:
                masks[split_name] = np.zeros_like(sample.base_concepts, dtype=bool)
                continue

            if mechanism_key == "mcar":
                mask = sample_mcar_mask(rng_generated, sample.base_concepts.shape, p)
            else:
                mask = sample_mnar_mask(
                    rng_generated, sample.base_concepts, base_p=p, config=mnar_config
                )

            sample.set_concept_missing_mask(mask, fill_value=fill_value)
            masks[split_name] = mask

        return masks

    def sample_concept_noise(
        self,
        *,
        p: float = 0.1,
        rng: np.random.Generator | int | None = None,
        config: Mapping[str, object] | None = None,
        enable: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample concept-level noise masks (bit flips).

        Args:
            p: Baseline probability of flipping each concept bit.
            rng: Optional ``np.random.Generator`` or int seed for reproducibility.
            config: Optional configuration dict with keys:
                - ``flip_prob``: symmetric flip probability per concept.
                - ``p01``: probability of flipping 0→1 (scalar or per-concept array).
                - ``p10``: probability of flipping 1→0 (scalar or per-concept array).
                - ``prob_matrix``: full matrix of flip probabilities overriding the
                  above options.
            enable: If provided, sets ``self.concept_noise`` to the boolean value
                after sampling.

        Returns:
            Dictionary mapping split name to the boolean flip mask for that split.
        """

        if enable is not None:
            self.concept_noise = bool(enable)

        rng_generated = coerce_rng(rng)
        masks: dict[str, np.ndarray] = {}
        splits = {
            "training": self.training,
            "validation": getattr(self, "validation", None),
            "test": getattr(self, "test", None),
        }

        for split_name, sample in splits.items():
            if sample is None or sample.base_concepts is None:
                continue

            if sample.base_concepts.size == 0:
                masks[split_name] = np.zeros_like(sample.base_concepts, dtype=bool)
                continue

            mask = sample_concept_noise_mask(
                rng_generated,
                sample.base_concepts,
                base_p=p,
                config=config,
            )
            sample.set_concept_noise_mask(mask)
            masks[split_name] = mask

        return masks

    def sample_label_noise(
        self,
        *,
        p: float = 0.1,
        rng: np.random.Generator | int | None = None,
        label_noise_config: Mapping[str, object] | None = None,
        enable: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample label noise for each split and optionally enable the view."""

        if enable is not None:
            self.label_noise = bool(enable)

        rng_generated = coerce_rng(rng)

        noisy_labels: dict[str, np.ndarray] = {}

        full_labels = sample_label_noise(
            rng_generated,
            self._full.base_labels,
            num_classes=self._full.n_classes,
            base_p=p,
            config=label_noise_config,
        )
        self._full.set_label_noise_labels(full_labels)
        noisy_labels["full"] = full_labels

        splits = {
            "training": self.training,
            "validation": getattr(self, "validation", None),
            "test": getattr(self, "test", None),
        }

        for split_name, sample in splits.items():
            if sample is None or sample.base_labels.size == 0:
                continue
            if sample is self._full:
                noisy_labels[split_name] = full_labels
                continue
            new_labels = sample_label_noise(
                rng_generated,
                sample.base_labels,
                num_classes=sample.n_classes,
                base_p=p,
                config=label_noise_config,
            )
            sample.set_label_noise_labels(new_labels)
            noisy_labels[split_name] = new_labels

        return noisy_labels


class ConceptDatasetSample(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        meta: dict,
        *,
        parent: "ConceptDataset" = None,
        indices: np.ndarray | None = None,
        transform: Callable | None = None,
        concept_transform: Callable | None = None,
        target_transform: Callable | None = None,
        concept_noise: bool = False,
        concept_missing: bool = False,
        label_noise: bool = False,
        **kwargs,
    ) -> None:
        if not {"classes", "concepts", "data_type"}.issubset(meta.keys()):
            raise ValueError(
                "metedata dict must contain keys 'classes', 'concepts', and 'data_type'"
            )

        self.parent = parent
        self.transform = transform
        self.concept_transform = concept_transform
        self.target_transform = target_transform
        self._extra_kwargs = dict(kwargs)

        self._X = X
        self._y_base = np.asarray(y, dtype=np.int32)
        self._label_noise_labels: np.ndarray | None = None
        self._meta = meta
        self.classes, self.concepts = meta["classes"], meta["concepts"]
        self.task = meta["data_type"]

        self._C_base = np.asarray(C, dtype=np.int8)
        self.n = len(self._X)
        if self._y_base.ndim != 1:
            self._y_base = self._y_base.reshape(-1)
        if self._y_base.shape[0] != self.n:
            raise ValueError("Label vector must match number of samples")

        if indices is None:
            self.indices = np.ones(self.n, dtype=np.bool_)
        else:
            self.indices = np.asarray(indices).flatten().astype(np.bool_)

        self._concept_noise_mask: np.ndarray | None = None
        self._concept_missing_mask: np.ndarray | None = None
        self._concept_missing_fill_value: float = np.nan
        self._concept_noise_enabled = bool(concept_noise)
        self._concept_missing_enabled = bool(concept_missing)
        self._label_noise_enabled = bool(label_noise)

        assert self.__check_rep__()

    @property
    def meta(self) -> dict:
        return self._meta

    @meta.setter
    def meta(self, value: dict) -> None:
        if not {"classes", "concepts", "data_type"}.issubset(value.keys()):
            raise ValueError(
                "metedata dict must contain keys 'classes', 'concepts', and 'data_type'"
            )
        self._meta = value
        self.classes, self.concepts = value["classes"], value["concepts"]
        self.task = value["data_type"]

    @property
    def X(self) -> np.ndarray:
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        self._X = value
        self.n = len(self._X)

    @property
    def y(self) -> np.ndarray:
        apply_noise = self._label_noise_enabled and (
            self._label_noise_labels is not None
        )
        return self._label_noise_labels if apply_noise else self._y_base

    @y.setter
    def y(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.int32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.shape[0] != self.n:
            raise ValueError("Label vector must match number of samples")
        self._y_base = arr
        self._label_noise_labels = None

    @property
    def C(self) -> np.ndarray:
        base = self._C_base
        noise_mask = self._concept_noise_mask
        missing_mask = self._concept_missing_mask
        fill_value = self._concept_missing_fill_value

        apply_noise = self._concept_noise_enabled and (noise_mask is not None)
        apply_missing = self._concept_missing_enabled and (missing_mask is not None)

        if not apply_noise and not apply_missing:
            return base

        if apply_noise:
            concepts = np.where(noise_mask, 1 - base, base)
        else:
            concepts = base.copy()

        if apply_missing:
            dtype = np.result_type(concepts.dtype, type(fill_value))
            concepts = concepts.astype(dtype)
            concepts[missing_mask] = fill_value

        return concepts

    @C.setter
    def C(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.int8)
        if arr.ndim != 2:
            raise ValueError("Concept matrix must be 2-dimensional")
        if arr.shape[0] != self.n:
            raise ValueError("Concept matrix must match number of samples")
        self._C_base = arr

    @property
    def base_concepts(self) -> np.ndarray:
        return self._C_base

    @property
    def base_labels(self) -> np.ndarray:
        return self._y_base

    def set_concept_noise_mask(self, mask: np.ndarray | None) -> None:
        if mask is None:
            self._concept_noise_mask = None
            return
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.shape != self.base_concepts.shape:
            raise ValueError("Noise mask must match concept shape")
        self._concept_noise_mask = mask_arr

    @property
    def concept_noise_mask(self) -> np.ndarray | None:
        return self._concept_noise_mask

    def set_concept_missing_mask(
        self, mask: np.ndarray | None, *, fill_value: float = np.nan
    ) -> None:
        if mask is None:
            self._concept_missing_mask = None
            self._concept_missing_fill_value = fill_value
            return
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.shape != self.base_concepts.shape:
            raise ValueError("Missingness mask must match concept shape")
        self._concept_missing_mask = mask_arr
        self._concept_missing_fill_value = fill_value

    @property
    def concept_missing_mask(self) -> np.ndarray | None:
        return self._concept_missing_mask

    @property
    def concept_missing_fill_value(self):
        return self._concept_missing_fill_value

    @property
    def concept_noise(self) -> bool:
        return self._concept_noise_enabled

    @concept_noise.setter
    def concept_noise(self, value: bool) -> None:
        self._concept_noise_enabled = bool(value)

    @property
    def concept_missing(self) -> bool:
        return self._concept_missing_enabled

    @concept_missing.setter
    def concept_missing(self, value: bool) -> None:
        self._concept_missing_enabled = bool(value)

    @property
    def label_noise(self) -> bool:
        return self._label_noise_enabled

    @label_noise.setter
    def label_noise(self, value: bool) -> None:
        self._label_noise_enabled = bool(value)

    def set_label_noise_labels(self, labels: np.ndarray | None) -> None:
        if labels is None:
            self._label_noise_labels = None
            return
        arr = np.asarray(labels, dtype=self._y_base.dtype)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.shape[0] != self.n:
            raise ValueError("Label noise array must match number of samples")
        self._label_noise_labels = arr

    @property
    def label_noise_labels(self) -> np.ndarray | None:
        return self._label_noise_labels

    def __len__(self):
        return self.n

    def __eq__(self, other):
        if not isinstance(other, ConceptDatasetSample):
            return False
        if not np.array_equal(self.base_labels, other.base_labels):
            return False
        if not np.array_equal(self.X, other.X):
            return False
        if not np.array_equal(self.base_concepts, other.base_concepts):
            return False
        if not _deep_equal(self.meta, other.meta):
            return False
        if self.transform is not other.transform:
            return False
        if self.concept_transform is not other.concept_transform:
            return False
        if self.target_transform is not other.target_transform:
            return False
        return True

    def __check_rep__(self):
        return True

    def __getitem__(self, idx):
        x = self.X[idx]
        c = self.C[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)
        if self.concept_transform is not None:
            c = self.concept_transform(c)
        if self.target_transform is not None:
            y = self.target_transform(y)

        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        if isinstance(c, np.ndarray):
            c = c.astype(np.float32)
        if isinstance(y, (np.ndarray, np.integer)):
            y = y.astype(np.float32)

        return x, c, y

    def __repr__(self):
        return (
            "ConceptDatasetSample<n={n}, n_concepts={nc}, n_classes={nl}, "
            "data_type={dt}>"
        ).format(
            n=self.n,
            nc=self.n_concepts,
            nl=self.n_classes,
            dt=self.meta.get("data_type"),
        )

    @property
    def n_concepts(self):
        return len(self.concepts)

    @property
    def n_classes(self):
        return len(self.classes)

    def filter(self, indices):
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()

        filtered_meta = self.meta.copy()
        if 'UC' in filtered_meta:
            filtered_meta['UC'] = filtered_meta['UC'][indices]
            filtered_meta["df_indices"] = filtered_meta["df_indices"][indices]
        if "robot_ids" in filtered_meta:
            filtered_meta["robot_ids"] = np.asarray(filtered_meta["robot_ids"])[indices]

        new_sample = self.__class__(
            parent=self.parent,
            X=self.X[indices],
            C=self.base_concepts[indices],
            y=self.base_labels[indices],
            meta=filtered_meta,
            indices=indices,
            concept_noise=self.concept_noise,
            concept_missing=self.concept_missing,
            label_noise=self.label_noise,
            transform=self.transform,
            concept_transform=self.concept_transform,
            target_transform=self.target_transform,
            **self._extra_kwargs,
        )
        if self.concept_noise_mask is not None:
            new_sample.set_concept_noise_mask(self.concept_noise_mask[indices])
        if self.concept_missing_mask is not None:
            new_sample.set_concept_missing_mask(
                self.concept_missing_mask[indices],
                fill_value=self.concept_missing_fill_value,
            )
        if self.label_noise_labels is not None:
            new_sample.set_label_noise_labels(self.label_noise_labels[indices])
        return new_sample

    def loader(self, batch_size=32, shuffle=False, **kwargs) -> DataLoader:
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return loader

    def embed(
        self, model, batch_size=32, shuffle=False, device="cpu", **kwargs
    ) -> "ConceptDatasetSample":
        model = model.to(device)
        model.eval()
        loader = self.loader(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", False),
        )

        embedded_X = []
        for x_batch, c_batch, y_batch in tqdm(loader):
            if isinstance(x_batch, (list, tuple)):
                x_batch = [
                    torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
                    for x in x_batch
                ]
                try:
                    x_batch = torch.stack(x_batch, dim=0)
                except Exception:
                    pass
            if isinstance(x_batch, torch.Tensor):
                x_batch = x_batch.to(device)
            with torch.no_grad():
                embedded_x = model(x_batch)
            if isinstance(embedded_x, torch.Tensor):
                embedded_X.append(embedded_x.detach().cpu().numpy())
            else:
                embedded_X.append(np.asarray(embedded_x))

        embedded_X = np.concatenate(embedded_X, axis=0)

        embed_meta = dict(self.meta).copy()
        embed_meta["data_type"] = "tabular"

        new_sample = ConceptDatasetSample(
            parent=self.parent,
            X=embedded_X,
            C=self.base_concepts,
            y=self.base_labels,
            meta=embed_meta,
            indices=self.indices,
            concept_noise=self.concept_noise,
            concept_missing=self.concept_missing,
            label_noise=self.label_noise,
        )
        if self.concept_noise_mask is not None:
            new_sample.set_concept_noise_mask(self.concept_noise_mask.copy())
        if self.concept_missing_mask is not None:
            new_sample.set_concept_missing_mask(
                self.concept_missing_mask.copy(),
                fill_value=self.concept_missing_fill_value,
            )
        if self.label_noise_labels is not None:
            new_sample.set_label_noise_labels(self.label_noise_labels.copy())
        return new_sample


class ConceptImageDatasetSample(ConceptDatasetSample):
    def __init__(
        self,
        X: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        meta: dict,
        *,
        parent: "ConceptDataset" = None,
        indices: np.ndarray | None = None,
        transform: Callable | None = None,
        concept_transform: Callable | None = None,
        target_transform: Callable | None = None,
        preprocess: Callable | None = None,
        base_dir: Path | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            X=X,
            C=C,
            y=y,
            meta=meta,
            parent=parent,
            indices=indices,
            transform=transform,
            concept_transform=concept_transform,
            target_transform=target_transform,
            **kwargs,
        )
        self.preprocess = preprocess
        if base_dir is None:
            self.base_dir = Path(".")
        else:
            self.base_dir = Path(base_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, c, y = self.X[idx], self.C[idx], self.y[idx]

        p = img_path if isinstance(img_path, Path) else Path(img_path)
        bd = self.base_dir if isinstance(self.base_dir, Path) else (Path(self.base_dir) if self.base_dir is not None else None)
        if bd is not None and not p.is_absolute():
            parts = p.parts
            # Strip "data/<base>" or "<base>" prefixes from relative paths
            if len(parts) >= 2 and parts[0] == bd.parent.name and parts[1] == bd.name:
                p = Path(*parts[2:])
            elif parts and parts[0] == bd.name:
                p = Path(*parts[1:])
            p = bd / p
        img_path = p
        try:
            image = Image.open(img_path).convert("RGB")
            if self.preprocess is not None:
                image = self.preprocess(image)
            if self.transform is not None:
                image = self.transform(image)
        except (AttributeError, FileNotFoundError, OSError) as e:
            warnings.warn(f"{e}; cannot open image, returning path", RuntimeWarning)
            image = img_path

        c = torch.from_numpy(np.array(c, dtype=np.int64))
        y = torch.from_numpy(np.array(y, dtype=np.int64))

        if self.concept_transform is not None:
            c = self.concept_transform(c)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return image, c, y

    def __eq__(self, other):
        chk = super().__eq__(other) and (self.base_dir == other.base_dir)
        return chk

    def __repr__(self):
        return (
            "ConceptImageDatasetSample<n={n}, n_concepts={nc}, n_classes={nl}, "
            "data_type={dt}, base_dir={bd}>"
        ).format(
            n=self.n,
            nc=self.n_concepts,
            nl=self.n_classes,
            dt=self.meta.get("data_type"),
            bd=self.base_dir,
        )

    def filter(self, indices):
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()

        filtered_meta = self.meta.copy()
        if 'UC' in filtered_meta:
            filtered_meta['UC'] = filtered_meta['UC'][indices]
            filtered_meta["df_indices"] = filtered_meta["df_indices"][indices]
        if "robot_ids" in filtered_meta:
            filtered_meta["robot_ids"] = np.asarray(filtered_meta["robot_ids"])[indices]

        new_sample = self.__class__(
            parent=self.parent,
            X=self.X[indices],
            C=self.base_concepts[indices],
            y=self.base_labels[indices],
            meta=filtered_meta,
            indices=indices,
            concept_noise=self.concept_noise,
            concept_missing=self.concept_missing,
            label_noise=self.label_noise,
            preprocess=self.preprocess,
            transform=self.transform,
            concept_transform=self.concept_transform,
            target_transform=self.target_transform,
            base_dir=self.base_dir,
            **self._extra_kwargs,
        )
        if self.concept_noise_mask is not None:
            new_sample.set_concept_noise_mask(self.concept_noise_mask[indices])
        if self.concept_missing_mask is not None:
            new_sample.set_concept_missing_mask(
                self.concept_missing_mask[indices],
                fill_value=self.concept_missing_fill_value,
            )
        if self.label_noise_labels is not None:
            new_sample.set_label_noise_labels(self.label_noise_labels[indices])
        return new_sample
