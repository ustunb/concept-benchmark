from __future__ import annotations

__all__ = [
    "ConceptDataset",
    "ConceptDatasetSample",
    "ConceptImageDatasetSample",
    "DataLoader",
    "InputType",
]

import io
import platform
import warnings
from collections.abc import Callable, Mapping, Set
from pathlib import Path
from typing import Literal

InputType = Literal["image", "tabular", "text"]

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader as _TorchDataLoader, Dataset
from tqdm import tqdm

# On macOS, forked workers + MPS can deadlock. Wrap DataLoader to force
# single-process loading so every call site is safe without remembering
# to pass num_workers=0.
if platform.system() == "Darwin":

    def DataLoader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return _TorchDataLoader(*args, **kwargs)
else:
    DataLoader = _TorchDataLoader

from .cv import generate_cvindices, validate_cvindices
from .helper.data_utils import (
    coerce_rng,
    sample_concept_noise_mask,
    sample_label_noise,
    sample_mcar_mask,
    sample_mnar_mask,
)


def _resolve_split_size(size: int | float, total: int) -> int:
    """Convert a float fraction or absolute count to an integer count."""
    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValueError(f"Float size must be in (0, 1), got {size}")
        return int(round(size * total))
    return min(int(size), total)


def _to_mask(indices: np.ndarray, n: int) -> np.ndarray:
    """Convert an array of integer indices to a boolean mask of length *n*."""
    mask = np.zeros(n, dtype=bool)
    if len(indices) > 0:
        mask[indices] = True
    return mask


def _stratified_split(
    n: int,
    labels: np.ndarray,
    n_test: int,
    n_val: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices with class-stratified proportions."""
    classes = np.unique(labels)
    train_parts, val_parts, test_parts = [], [], []
    for c in classes:
        c_idx = np.where(labels == c)[0].copy()
        rng.shuffle(c_idx)
        n_c = len(c_idx)
        n_c_test = int(round(n_test * n_c / n))
        n_c_val = int(round(n_val * n_c / n))
        test_parts.append(c_idx[:n_c_test])
        val_parts.append(c_idx[n_c_test : n_c_test + n_c_val])
        train_parts.append(c_idx[n_c_test + n_c_val :])
    return (
        np.concatenate(train_parts) if train_parts else np.array([], dtype=int),
        np.concatenate(val_parts) if val_parts else np.array([], dtype=int),
        np.concatenate(test_parts) if test_parts else np.array([], dtype=int),
    )


def _group_split(
    groups: np.ndarray,
    n_test: int,
    n_val: int,
    stratify: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices so no group appears in multiple splits."""
    n = len(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    g_test = max(1, int(round(n_test / n * n_groups)))
    g_val = max(1, int(round(n_val / n * n_groups)))

    if stratify is not None:
        # Majority label per group
        group_labels: dict[int, list] = {}
        for i in range(n):
            group_labels.setdefault(int(groups[i]), []).append(int(stratify[i]))
        group_majority = {g: int(round(np.mean(ls))) for g, ls in group_labels.items()}
        by_class: dict[int, list] = {}
        for g in unique_groups:
            by_class.setdefault(group_majority[int(g)], []).append(g)
        test_g, val_g = [], []
        for cls in sorted(by_class):
            arr = np.array(by_class[cls])
            rng.shuffle(arr)
            frac = len(arr) / n_groups
            nt = int(round(g_test * frac))
            nv = int(round(g_val * frac))
            test_g.extend(arr[:nt].tolist())
            val_g.extend(arr[nt : nt + nv].tolist())
    else:
        arr = unique_groups.copy()
        rng.shuffle(arr)
        test_g = arr[:g_test].tolist()
        val_g = arr[g_test : g_test + g_val].tolist()

    test_set = set(test_g)
    val_set = set(val_g)
    test_mask = np.isin(groups, list(test_set))
    val_mask = np.isin(groups, list(val_set))
    train_mask = ~(test_mask | val_mask)
    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]


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
        except TypeError:
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
        except OSError:
            return str(a) == str(b)

    if callable(a) or callable(b):
        return a is b

    try:
        eq = a == b
    except (TypeError, ValueError):
        return repr(a) == repr(b)
    else:
        if isinstance(eq, (bool, np.bool_)):
            return bool(eq)
        if hasattr(eq, "all"):
            try:
                return bool(eq.all())
            except (TypeError, ValueError):
                pass
        return repr(a) == repr(b)


def _data_preview(sample, max_rows: int = 3, indent: str = "  ") -> str:
    """Return a pandas-style data preview string for a dataset sample."""
    try:
        df = sample.to_dataframe()
        n_rows, n_cols = len(df), len(df.columns)
        preview = df.head(max_rows).to_string()
        # Indent each line
        preview = "\n".join(indent + line for line in preview.split("\n"))
        preview += f"\n{indent}[{n_rows} rows \u00d7 {n_cols} columns]"
        return preview
    except Exception:
        return f"{indent}(preview unavailable)"


class ConceptDataset:
    """Container for concept-annotated datasets with train/val/test splits.

    Wraps a feature matrix *X*, a binary concept matrix *C*, and a label
    vector *y* together with metadata and optional noise/missingness overlays.

    Create splits with :meth:`sample` and access them as ``dataset.train``,
    ``dataset.val``, ``dataset.test``::

        from concept_benchmark.robots import DatasetGenerator

        ds = DatasetGenerator(seed=1014, render_images=False).generate()
        ds.sample(test_size=0.2, val_size=0.2, seed=1014)
        print(ds.train.n, ds.val.n, ds.test.n)

    Each split is a :class:`ConceptDatasetSample` (a ``torch.utils.data.Dataset``)
    with ``.X``, ``.C``, ``.y``, ``.concepts``, ``.loader()``, and
    ``.to_dataframe()`` attributes.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.  For image data, an array of image file paths.
    C : np.ndarray
        Concept matrix of shape ``(n_samples, n_concepts)`` with binary
        values (0 or 1).
    y : np.ndarray
        Label vector of shape ``(n_samples,)`` with integer class labels.
    meta : dict
        Metadata dictionary.  Must contain keys ``'classes'`` (list of
        class names), ``'concepts'`` (list of concept names), and
        ``'data_type'`` (``'image'``, ``'tabular'``, or ``'text'``).
    """

    SAMPLE_TYPES = ("train", "validation", "test")

    def __init__(
        self,
        inputs: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        meta: dict,
        *,
        input_type: InputType,
        classes: tuple[int, ...],
        cvindices: dict | None = None,
        transform: Callable | None = None,
        concept_transform: Callable | None = None,
        target_transform: Callable | None = None,
        has_concept_noise: bool = False,
        has_concept_missing: bool = False,
        has_label_noise: bool = False,
        **kwargs,
    ) -> None:
        if input_type not in ("image", "tabular", "text"):
            raise ValueError(
                f"input_type must be 'image', 'tabular', or 'text'; got {input_type!r}"
            )
        self.input_type = input_type
        self._classes_tuple = tuple(classes)

        self._init_kwargs = dict(kwargs)
        self._has_concept_noise = bool(has_concept_noise)
        self._has_concept_missing = bool(has_concept_missing)
        self._has_label_noise = bool(has_label_noise)
        self._init_kwargs.update(
            has_concept_noise=self._has_concept_noise,
            has_concept_missing=self._has_concept_missing,
            has_label_noise=self._has_label_noise,
        )

        if not isinstance(inputs, np.ndarray):
            try:
                inputs = np.asarray(inputs)
            except (TypeError, ValueError) as e:
                raise ValueError(f"cannot convert inputs to np.ndarray: {e!r}")

        if not isinstance(C, np.ndarray):
            try:
                C = np.asarray(C)
            except (TypeError, ValueError) as e:
                raise ValueError(f"cannot convert C to np.ndarray: {e!r}")

        if not isinstance(y, np.ndarray):
            try:
                y = np.asarray(y)
            except (TypeError, ValueError) as e:
                raise ValueError(f"cannot convert y to np.ndarray: {e!r}")

        if input_type == "image":
            # do not cast inputs
            SampleClass = ConceptImageDatasetSample
        elif input_type == "text":
            SampleClass = ConceptDatasetSample
            inputs = inputs.astype(object)
        else:
            SampleClass = ConceptDatasetSample
            inputs = inputs.astype(np.float32)

        C = C.astype(np.int8)
        y = y.astype(np.int32)

        self._full = SampleClass(
            parent=self,
            inputs=inputs,
            C=C,
            y=y,
            meta=meta,
            input_type=input_type,
            classes=self._classes_tuple,
            transform=transform,
            concept_transform=concept_transform,
            target_transform=target_transform,
            has_concept_noise=self._has_concept_noise,
            has_concept_missing=self._has_concept_missing,
            has_label_noise=self._has_label_noise,
            **kwargs,
        )

        self._cvindices = cvindices
        self.reset()

    def drop_concepts(self, concepts_to_drop):
        """Remove concepts from the dataset by name.

        Parameters
        ----------
        concepts_to_drop : list of str
            Concept names to drop.  At least one concept must remain.
        """
        if not isinstance(concepts_to_drop, (list, tuple, set)):
            raise ValueError(
                "concepts_to_drop should be a list, tuple, or set of strings"
            )
        concepts_to_drop = set(concepts_to_drop)
        existing_concepts = set(self.concepts)
        invalid_concepts = concepts_to_drop - existing_concepts
        if invalid_concepts:
            raise ValueError(f"Concepts not found in dataset: {invalid_concepts}")

        keep_indices = [
            i for i, c in enumerate(self.concepts) if c not in concepts_to_drop
        ]
        if not keep_indices:
            raise ValueError("Cannot drop all concepts; at least one must remain.")

        new_concepts = [self.concepts[i] for i in keep_indices]

        # Update all samples (including _full and splits)
        for sample in self._iter_samples():
            sample._C_base = sample._C_base[:, keep_indices]
            new_meta = dict(sample.meta)
            new_meta["concepts"] = new_concepts
            sample.meta = new_meta
            if sample._concept_noise_mask is not None:
                sample._concept_noise_mask = sample._concept_noise_mask[:, keep_indices]
            if sample._concept_missing_mask is not None:
                sample._concept_missing_mask = sample._concept_missing_mask[
                    :, keep_indices
                ]

        assert self.__check_rep__()
        return self

    def reset(self):
        """Reset to the pre-split state (all data in training, empty val/test)."""
        self._fold_id = None
        self._fold_number_range = []
        self._fold_num_test = 0
        self._fold_num_validation = 0
        self._fold_num_range = 0
        self.train = self._full
        self.validation = self._full.filter(indices=np.zeros(self.n, dtype=np.bool_))
        self.test = self._full.filter(indices=np.zeros(self.n, dtype=np.bool_))
        self._apply_noise_settings()
        assert self.__check_rep__()

    def _iter_samples(self):
        seen = set()
        for sample in (
            getattr(self, "_full", None),
            getattr(self, "train", None),
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
            sample.has_concept_noise = self._has_concept_noise
            sample.has_concept_missing = self._has_concept_missing
            sample.has_label_noise = self._has_label_noise

    # -- Dict-style split access --

    _SPLIT_ALIASES = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }

    def __getitem__(self, key: str) -> "ConceptDatasetSample":
        """Access a split by name (``"train"``, ``"val"``, or ``"test"``)."""
        try:
            attr = self._SPLIT_ALIASES[key]
        except KeyError:
            raise KeyError(f"Unknown split '{key}'. Available: 'train', 'val', 'test'")
        return getattr(self, attr)

    def keys(self):
        """Available split names: ``['train', 'val', 'test']``."""
        return ["train", "val", "test"]

    def __contains__(self, key) -> bool:
        """Check whether *key* is a valid split name."""
        return key in self._SPLIT_ALIASES

    def __len__(self) -> int:
        """Total number of samples across all splits."""
        return self._full.n

    @property
    def val(self) -> "ConceptDatasetSample":
        """Alias for :attr:`validation`."""
        return self.validation

    # -- Description --

    @property
    def description(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [
            f"{self.__class__.__name__} ({self._full.input_type})",
            f"  Samples: {self.n} (train={self.train.n}, val={self.validation.n}, test={self.test.n})",
            f"  Concepts ({self.n_concepts}): {self.concepts}",
            f"  Classes ({self.n_classes}): {self.classes}",
        ]
        return "\n".join(lines)

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

        assert n_total <= self.n

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

    def __repr__(self):
        dt = self._full.input_type
        concepts = list(self.concepts)
        if len(concepts) > 5:
            concepts_str = str(concepts[:5])[:-1] + ", ...]"
        else:
            concepts_str = str(concepts)

        lines = [
            f"ConceptDataset({dt}, {self.n} samples, {self.n_concepts} concepts, {self.n_classes} classes)",
            f"  concepts: {concepts_str}",
        ]

        has_splits = self.train.n > 0 and self.test.n > 0
        if has_splits:
            lines.append(
                f"  splits:   train={self.train.n}, val={self.validation.n}, test={self.test.n}"
            )

        # Show data preview from the first available split (or _full)
        lines.append("")
        sample = self.train if has_splits else self._full
        lines.append(_data_preview(sample))
        return "\n".join(lines)

    def __copy__(self):
        cpy = ConceptDataset(
            inputs=self.inputs,
            C=self._full.base_concepts.copy(),
            y=self._full.base_labels.copy(),
            meta=self._full.meta,
            input_type=self._full.input_type,
            classes=self._full.classes,
            cvindices=self._cvindices,
            **self._init_kwargs,
        )
        cpy.has_concept_noise = self.has_concept_noise
        cpy.has_concept_missing = self.has_concept_missing
        cpy.has_label_noise = self.has_label_noise

        return cpy

    #### INSTANCE VARIABLES
    @property
    def classes(self):
        """List of class names, ordered by label index."""
        return self._full.classes

    @property
    def concepts(self):
        """List of concept names, ordered by column index in *C*."""
        return self._full.concepts

    @property
    def n(self):
        """Total number of samples in the full (unsplit) dataset."""
        return self._full.n

    @property
    def n_concepts(self):
        """Number of concepts."""
        return self._full.n_concepts

    @property
    def n_classes(self):
        """Number of classes."""
        return self._full.n_classes

    @property
    def inputs(self):
        """Raw inputs of the full dataset (feature matrix, paths, or text array)."""
        return self._full.inputs

    @property
    def C(self):
        """Concept matrix of the full dataset (with noise/missingness applied)."""
        return self._full.C

    @property
    def y(self):
        """Label vector of the full dataset (with label noise applied)."""
        return self._full.y

    @property
    def meta(self):
        """Metadata dictionary (classes, concepts, data_type, ...)."""
        return self._full.meta

    @property
    def transform(self):
        """Feature transform applied in ``__getitem__``."""
        return self._full.transform

    @property
    def concept_transform(self):
        """Concept transform applied in ``__getitem__``."""
        return self._full.concept_transform

    @property
    def target_transform(self):
        """Target transform applied in ``__getitem__``."""
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
    def has_concept_noise(self) -> bool:
        """Whether concept noise is applied when reading *C*."""
        return self._has_concept_noise

    @has_concept_noise.setter
    def has_concept_noise(self, value: bool) -> None:
        self._has_concept_noise = bool(value)
        self._apply_noise_settings()

    @property
    def has_concept_missing(self) -> bool:
        """Whether concept missingness is applied when reading *C*."""
        return self._has_concept_missing

    @has_concept_missing.setter
    def has_concept_missing(self, value: bool) -> None:
        self._has_concept_missing = bool(value)
        self._apply_noise_settings()

    @property
    def has_label_noise(self) -> bool:
        """Whether label noise is applied when reading *y*."""
        return self._has_label_noise

    @has_label_noise.setter
    def has_label_noise(self, value: bool) -> None:
        self._has_label_noise = bool(value)
        self._apply_noise_settings()

    #### cross validation ####
    @property
    def cvindices(self):
        """Cross-validation fold index dictionary, or ``None``."""
        return self._cvindices

    @cvindices.setter
    def cvindices(self, cvindices):
        self._cvindices = validate_cvindices(cvindices)

    @property
    def fold_id(self):
        """Active cross-validation fold identifier.

        Format: ``"K{folds}N{replicate}"`` — e.g. ``"K05N01"`` for 5-fold
        CV, 1st replicate.
        """
        return self._fold_id

    @fold_id.setter
    def fold_id(self, fold_id):
        assert self._cvindices is not None, (
            "cannot set fold_id on a ConceptDataset without cvindices"
        )
        assert isinstance(fold_id, str), f"fold_id={fold_id} should be string"
        assert fold_id in self.cvindices, (
            f"cvindices does not contain folds for fold_id=`{fold_id}`"
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
        """Split into training, validation, and test sets using CV folds.

        Parameters
        ----------
        fold_id : str
            Cross-validation fold identifier (e.g., ``"K05N01"``).
        fold_num_validation : int, optional
            Fold number to use as validation set.
        fold_num_test : int, optional
            Fold number to use as hold-out test set.
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
        self.train = self._full.filter(
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
        total_folds_for_cv=None,
        total_folds_for_inner_cv=None,
        replicates=3,
        seed=None,
    ):
        """Generate and store cross-validation fold indices.

        Parameters
        ----------
        strata : array-like, optional
            Stratification labels (typically ``y``) for balanced splits.
        total_folds_for_cv : list of int
            Number of folds for each CV scheme (e.g., ``[5]`` for 5-fold).
        total_folds_for_inner_cv : list of int
            Fold counts for nested (inner) CV.
        replicates : int
            Number of independent CV replicates.
        seed : int, optional
            Random seed for reproducibility.
        """
        if total_folds_for_cv is None:
            total_folds_for_cv = [1, 3, 4, 5]
        if total_folds_for_inner_cv is None:
            total_folds_for_inner_cv = []
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
        """Embed features with *model* and return a new tabular dataset.

        The original dataset is not modified.  Cross-validation splits and
        noise/missingness settings are preserved in the returned copy.

        Parameters
        ----------
        model : torch.nn.Module
            Encoder that maps input batches to feature vectors.
        batch_size : int
            Batch size for the embedding pass.
        shuffle : bool
            Whether to shuffle the data loader.
        device : str
            Device to run the model on (e.g. ``"cpu"``, ``"cuda"``).
        **kwargs
            Extra arguments forwarded to the data loader.

        Returns
        -------
        ConceptDataset
            New dataset with embedded features and ``data_type='tabular'``.
        """
        # Compute embedded representation for the full dataset sample
        embedded_full = self._full.embed(
            model, batch_size=batch_size, shuffle=shuffle, device=device, **kwargs
        )

        # Create a new ConceptDataset using the embedded features while
        # preserving metadata and CV indices from the original dataset.
        new_ds = ConceptDataset(
            inputs=embedded_full.inputs,
            C=embedded_full.C,
            y=embedded_full.y,
            meta=embedded_full.meta,
            input_type=embedded_full.input_type,
            classes=embedded_full.classes,
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

    def sample(
        self,
        *,
        test_size: int | float = 0.2,
        val_size: int | float = 0.2,
        train_size: int | None = None,
        groups: np.ndarray | None = None,
        stratify: np.ndarray | None = None,
        sampling_constraints: list[dict] | None = None,
        seed: int | None = None,
    ) -> "ConceptDataset":
        """Split into training/validation/test sets.

        Supports random, stratified, and group-based splitting, plus
        skewed training set resampling via *sampling_constraints*.

        Parameters
        ----------
        test_size : int or float
            Number of test samples (int) or fraction (float in (0, 1)).
        val_size : int or float
            Number of validation samples (int) or fraction (float in (0, 1)).
        train_size : int, optional
            Maximum number of training samples.  ``None`` uses all remaining.
        groups : np.ndarray, optional
            Group labels for group-based splitting (no group leakage).
        stratify : np.ndarray, optional
            Labels for stratified splitting (preserves class proportions).
        sampling_constraints : list of dict, optional
            Skewing constraints for the training set.  Each dict has
            ``"concepts"`` (mapping concept_name → value) and
            ``"min_fraction"`` (minimum fraction of training set).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ConceptDataset
            Self, with ``training``/``validation``/``test`` set.
        """
        from .utils import _create_skewed_training_set

        rng = np.random.default_rng(seed)
        n = self.n
        n_test = _resolve_split_size(test_size, n)
        n_val = _resolve_split_size(val_size, n)

        if groups is not None:
            train_idx, val_idx, test_idx = _group_split(
                groups, n_test, n_val, stratify, rng
            )
        elif stratify is not None:
            train_idx, val_idx, test_idx = _stratified_split(
                n, stratify, n_test, n_val, rng
            )
        else:
            all_idx = np.arange(n)
            rng.shuffle(all_idx)
            test_idx = all_idx[:n_test]
            remaining = all_idx[n_test:]

            if sampling_constraints:
                n_train = (
                    train_size if train_size is not None else len(remaining) - n_val
                )
                train_idx = _create_skewed_training_set(
                    self, sampling_constraints, remaining, n_train, rng
                )
                used = set(train_idx.tolist())
                val_pool = np.array([i for i in remaining if i not in used])
                rng.shuffle(val_pool)
                val_idx = val_pool[:n_val]
            else:
                val_idx = remaining[:n_val]
                train_idx = remaining[n_val:]

        # Apply train_size limit (when no sampling_constraints)
        if (
            not sampling_constraints
            and train_size is not None
            and len(train_idx) > train_size
        ):
            train_idx = train_idx[:train_size]

        self.train = self._full.filter(_to_mask(train_idx, n))
        self.validation = self._full.filter(_to_mask(val_idx, n))
        self.test = self._full.filter(_to_mask(test_idx, n))
        self._apply_noise_settings()
        return self

    def sample_concept_missingness(
        self,
        *,
        p: float = 0.1,
        mechanism: str = "mcar",
        rng: np.random.Generator | int | None = None,
        mnar_config: Mapping[str, object] | None = None,
        fill_value: float = np.nan,
        enable: bool | None = None,
        splits: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample concept-level missingness masks.

        Parameters
        ----------
        p : float
            Baseline prevalence of missingness.
        mechanism : ``"mcar"`` or ``"mnar"``
            Missingness mechanism.
        rng : np.random.Generator or int, optional
            Random generator or seed for reproducibility.
        mnar_config : dict, optional
            Configuration for MNAR missingness.  Accepted keys:

            - ``present_prob`` / ``absent_prob``: scalar or per-concept
              probabilities (length ``n_concepts``) applied when the
              observed concept value is 1 or 0 respectively.
            - ``prob_matrix``: full probability matrix overriding
              per-concept values (shape must match the concept matrix).
        fill_value : float
            Value used to replace missing concepts (default ``NaN``).
        enable : bool, optional
            If provided, sets :attr:`has_concept_missing` after sampling.
        splits : set of str, optional
            Which splits to apply missingness to (e.g. ``{"train"}``).
            ``None`` applies to all splits.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from split name (``"train"``, ``"validation"``,
            ``"test"``) to the sampled boolean mask.
        """

        mechanism_key = mechanism.lower()
        if mechanism_key not in {"mcar", "mnar"}:
            raise ValueError("mechanism must be either 'mcar' or 'mnar'")

        rng_generated = coerce_rng(rng)

        if enable is not None:
            self.has_concept_missing = bool(enable)

        masks: dict[str, np.ndarray] = {}
        all_splits = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }

        for split_name, sample in all_splits.items():
            if splits is not None and split_name not in splits:
                continue
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

        Parameters
        ----------
        p : float
            Baseline probability of flipping each concept bit.
        rng : np.random.Generator or int, optional
            Random generator or seed for reproducibility.
        config : dict, optional
            Fine-grained noise configuration.  Accepted keys:

            - ``flip_prob``: symmetric flip probability per concept.
            - ``p01``: probability of flipping 0 → 1 (scalar or per-concept).
            - ``p10``: probability of flipping 1 → 0 (scalar or per-concept).
            - ``prob_matrix``: full flip-probability matrix overriding the
              above options.
        enable : bool, optional
            If provided, sets :attr:`has_concept_noise` after sampling.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from split name to the boolean flip mask.
        """

        if enable is not None:
            self.has_concept_noise = bool(enable)

        rng_generated = coerce_rng(rng)
        masks: dict[str, np.ndarray] = {}
        splits = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
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
        """Sample label noise for each split.

        Parameters
        ----------
        p : float
            Probability of flipping each label to a random other class.
        rng : np.random.Generator or int, optional
            Random generator or seed for reproducibility.
        label_noise_config : dict, optional
            Fine-grained noise configuration forwarded to the sampler.
        enable : bool, optional
            If provided, sets :attr:`has_label_noise` after sampling.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from split name to the noisy label array.
        """

        if enable is not None:
            self.has_label_noise = bool(enable)

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
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
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
    """A single split (train/val/test) of a :class:`ConceptDataset`.

    Implements the PyTorch :class:`~torch.utils.data.Dataset` interface so it
    can be passed directly to a :class:`~torch.utils.data.DataLoader`.
    Concept noise, concept missingness, and label noise are applied lazily
    when accessing :attr:`C` or :attr:`y`.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (or array of file paths for image data).
    C : np.ndarray
        Binary concept matrix of shape ``(n_samples, n_concepts)``.
    y : np.ndarray
        Integer label vector of shape ``(n_samples,)``.
    meta : dict
        Metadata dict with keys ``'classes'``, ``'concepts'``, ``'data_type'``.
    parent : ConceptDataset, optional
        Back-reference to the owning :class:`ConceptDataset`.
    indices : np.ndarray, optional
        Boolean mask indicating which rows of the parent were selected.
    transform : callable, optional
        Feature transform applied in ``__getitem__``.
    concept_transform : callable, optional
        Concept transform applied in ``__getitem__``.
    target_transform : callable, optional
        Target transform applied in ``__getitem__``.
    has_concept_noise : bool
        Whether to apply the concept noise mask (default ``False``).
    has_concept_missing : bool
        Whether to apply the concept missingness mask (default ``False``).
    has_label_noise : bool
        Whether to return noisy labels (default ``False``).
    **kwargs
        Extra keyword arguments stored for :meth:`filter` round-tripping.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        meta: dict,
        *,
        input_type: InputType,
        classes: tuple[int, ...] | None = None,
        parent: "ConceptDataset" = None,
        indices: np.ndarray | None = None,
        transform: Callable | None = None,
        concept_transform: Callable | None = None,
        target_transform: Callable | None = None,
        has_concept_noise: bool = False,
        has_concept_missing: bool = False,
        has_label_noise: bool = False,
        **kwargs,
    ) -> None:
        if "concepts" not in meta:
            raise ValueError("metadata dict must contain key 'concepts'")
        if input_type not in ("image", "tabular", "text"):
            raise ValueError(
                f"input_type must be 'image', 'tabular', or 'text'; got {input_type!r}"
            )

        self.parent = parent
        self.transform = transform
        self.concept_transform = concept_transform
        self.target_transform = target_transform
        self._extra_kwargs = dict(kwargs)

        self._inputs = inputs
        self._y_base = np.asarray(y, dtype=np.int32)
        self._label_noise_labels: np.ndarray | None = None
        self._meta = meta
        self.input_type = input_type
        # `classes` is either passed explicitly (preferred) or read from meta for
        # construction sites still using the meta dict. The concepts list always
        # comes from meta because it's the canonical source today.
        if classes is None:
            classes = tuple(meta.get("classes", ()))
        self.classes = tuple(classes)
        self.concepts = meta["concepts"]

        self._C_base = np.asarray(C, dtype=np.int8)
        self.n = len(self._inputs)
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
        self._has_concept_noise = bool(has_concept_noise)
        self._has_concept_missing = bool(has_concept_missing)
        self._has_label_noise = bool(has_label_noise)

        assert self.__check_rep__()

    def __setstate__(self, state: dict) -> None:
        """Support unpickling objects saved with old attribute names."""
        renames = {
            "_concept_noise_enabled": "_has_concept_noise",
            "_concept_missing_enabled": "_has_concept_missing",
            "_label_noise_enabled": "_has_label_noise",
        }
        for old, new in renames.items():
            if old in state and new not in state:
                state[new] = state.pop(old)
        if "task" in state and "input_type" not in state:
            state["input_type"] = state.pop("task")
        if "data_type" in state and "input_type" not in state:
            state["input_type"] = state.pop("data_type")
        self.__dict__.update(state)

    @property
    def meta(self) -> dict:
        """Metadata dictionary (classes, concepts, data_type, ...)."""
        return self._meta

    @meta.setter
    def meta(self, value: dict) -> None:
        if "concepts" not in value:
            raise ValueError("metadata dict must contain key 'concepts'")
        self._meta = value
        self.concepts = value["concepts"]
        if "classes" in value:
            self.classes = tuple(value["classes"])

    @property
    def inputs(self) -> np.ndarray:
        """Raw inputs: feature matrix for tabular, path array for image, text array for text."""
        return self._inputs

    @inputs.setter
    def inputs(self, value: np.ndarray) -> None:
        self._inputs = value
        self.n = len(self._inputs)

    @property
    def y(self) -> np.ndarray:
        """Label vector, with label noise applied when enabled."""
        apply_noise = self._has_label_noise and (self._label_noise_labels is not None)
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
        """Concept matrix, with noise and missingness applied when enabled."""
        base = self._C_base
        noise_mask = self._concept_noise_mask
        missing_mask = self._concept_missing_mask
        fill_value = self._concept_missing_fill_value

        apply_noise = self._has_concept_noise and (noise_mask is not None)
        apply_missing = self._has_concept_missing and (missing_mask is not None)

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
        """Clean concept matrix before noise/missingness."""
        return self._C_base

    @property
    def base_labels(self) -> np.ndarray:
        """Clean label vector before noise."""
        return self._y_base

    def set_concept_noise_mask(self, mask: np.ndarray | None) -> None:
        """Set (or clear) the boolean concept noise mask.

        Parameters
        ----------
        mask : np.ndarray or None
            Boolean array matching ``base_concepts.shape``.  ``True``
            entries will have their concept value flipped.  Pass ``None``
            to clear.
        """
        if mask is None:
            self._concept_noise_mask = None
            return
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.shape != self.base_concepts.shape:
            raise ValueError("Noise mask must match concept shape")
        self._concept_noise_mask = mask_arr

    @property
    def concept_noise_mask(self) -> np.ndarray | None:
        """Boolean mask of concept noise flips, or ``None``."""
        return self._concept_noise_mask

    def set_concept_missing_mask(
        self, mask: np.ndarray | None, *, fill_value: float = np.nan
    ) -> None:
        """Set (or clear) the concept missingness mask.

        Parameters
        ----------
        mask : np.ndarray or None
            Boolean array matching ``base_concepts.shape``.  ``True``
            entries will be replaced with *fill_value*.  Pass ``None``
            to clear.
        fill_value : float
            Value to substitute for missing concepts (default ``NaN``).
        """
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
        """Boolean mask of missing concepts, or ``None``."""
        return self._concept_missing_mask

    @property
    def concept_missing_fill_value(self):
        """Fill value used for missing concepts (default ``NaN``)."""
        return self._concept_missing_fill_value

    @property
    def has_concept_noise(self) -> bool:
        """Whether concept noise is applied when reading :attr:`C`."""
        return self._has_concept_noise

    @has_concept_noise.setter
    def has_concept_noise(self, value: bool) -> None:
        self._has_concept_noise = bool(value)

    @property
    def has_concept_missing(self) -> bool:
        """Whether concept missingness is applied when reading :attr:`C`."""
        return self._has_concept_missing

    @has_concept_missing.setter
    def has_concept_missing(self, value: bool) -> None:
        self._has_concept_missing = bool(value)

    @property
    def has_label_noise(self) -> bool:
        """Whether label noise is applied when reading :attr:`y`."""
        return self._has_label_noise

    @has_label_noise.setter
    def has_label_noise(self, value: bool) -> None:
        self._has_label_noise = bool(value)

    def set_label_noise_labels(self, labels: np.ndarray | None) -> None:
        """Set (or clear) the noisy label vector.

        Parameters
        ----------
        labels : np.ndarray or None
            Integer array of the same length as the sample, or ``None``
            to clear.
        """
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
        """Noisy label vector, or ``None`` if not sampled."""
        return self._label_noise_labels

    def __len__(self):
        return self.n

    def __eq__(self, other):
        if not isinstance(other, ConceptDatasetSample):
            return False
        if not np.array_equal(self.base_labels, other.base_labels):
            return False
        if not np.array_equal(self.inputs, other.inputs):
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
        """Return ``(x, c, y)`` for the given index, with transforms applied."""
        x = self.inputs[idx]
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
        lines = [
            f"ConceptDatasetSample({self.input_type}, {self.n} samples, {self.n_concepts} concepts)",
            "",
            _data_preview(self),
        ]
        return "\n".join(lines)

    @property
    def n_concepts(self):
        """Number of concepts."""
        return len(self.concepts)

    @property
    def n_classes(self):
        """Number of classes."""
        return len(self.classes)

    def filter(self, indices):
        """Return a new sample containing only the selected rows.

        Parameters
        ----------
        indices : np.ndarray
            Boolean mask of length ``n``.  Noise/missingness masks are
            sliced accordingly.

        Returns
        -------
        ConceptDatasetSample
            Filtered copy (same class as ``self``).
        """
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()

        filtered_meta = self.meta.copy()
        if "UC" in filtered_meta:
            filtered_meta["UC"] = filtered_meta["UC"][indices]
            filtered_meta["df_indices"] = filtered_meta["df_indices"][indices]
        if "robot_ids" in filtered_meta:
            filtered_meta["robot_ids"] = np.asarray(filtered_meta["robot_ids"])[indices]

        new_sample = self.__class__(
            parent=self.parent,
            inputs=self.inputs[indices],
            C=self.base_concepts[indices],
            y=self.base_labels[indices],
            meta=filtered_meta,
            input_type=self.input_type,
            classes=self.classes,
            indices=indices,
            has_concept_noise=self.has_concept_noise,
            has_concept_missing=self.has_concept_missing,
            has_label_noise=self.has_label_noise,
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
        """Create a PyTorch :class:`DataLoader` for this sample.

        Parameters
        ----------
        batch_size : int
            Batch size (default 32).
        shuffle : bool
            Whether to shuffle (default ``False``).
        **kwargs
            Extra arguments forwarded to :class:`DataLoader`.

        Returns
        -------
        DataLoader
        """
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return loader

    def to_dataframe(self, include_X: bool = False) -> "pd.DataFrame":
        """Convert concepts and labels to a pandas DataFrame.

        Parameters
        ----------
        include_X : bool
            If ``True``, prepend feature columns.  For tabular data these are
            named ``x_0, x_1, …``; for text data a single ``text`` column is
            used.  Image data is handled by the
            :class:`ConceptImageDatasetSample` override.

        Returns
        -------
        pd.DataFrame
            DataFrame with concept columns (named by ``self.concepts``),
            a ``label`` column, and a ``class`` column.
        """
        parts: list[pd.DataFrame] = []
        if include_X:
            if self.input_type == "text":
                parts.append(pd.DataFrame({"text": list(self.inputs)}))
            else:
                # Tabular / fallback: one column per feature dimension
                X_arr = np.asarray(self.inputs)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                x_cols = {f"x_{j}": X_arr[:, j] for j in range(X_arr.shape[1])}
                parts.append(pd.DataFrame(x_cols))
        parts.append(pd.DataFrame(self.C, columns=self.concepts))
        df = pd.concat(parts, axis=1)
        df["label"] = self.y
        df["class"] = [self.classes[int(i)] for i in self.y]
        return df

    def explore(self, **kwargs):
        """Open an interactive data browser with `Renumics Spotlight`_.

        Requires the ``explore`` extra::

            pip install concept-benchmark[explore]
            # or: uv sync --group explore

        All keyword arguments are forwarded to :func:`spotlight.show`.

        .. _Renumics Spotlight: https://github.com/Renumics/spotlight
        """
        try:
            from renumics import spotlight
        except ImportError:
            raise ImportError(
                "The explore() method requires Renumics Spotlight.\n"
                "Install it with:  pip install concept-benchmark[explore]\n"
                "            or:  uv sync --group explore"
            ) from None

        df = self.to_dataframe(include_X=True)
        dtype = {}
        if "image" in df.columns:
            dtype["image"] = spotlight.Image
        spotlight.show(df, dtype=dtype, **kwargs)

    def embed(
        self, model, batch_size=32, shuffle=False, device="cpu", **kwargs
    ) -> "ConceptDatasetSample":
        """Embed features with *model* and return a new tabular sample.

        Parameters
        ----------
        model : torch.nn.Module
            Encoder that maps input batches to feature vectors.
        batch_size : int
            Batch size for the embedding pass.
        shuffle : bool
            Whether to shuffle the data loader.
        device : str
            Device to run the model on.
        **kwargs
            Extra arguments (``num_workers``, ``pin_memory``).

        Returns
        -------
        ConceptDatasetSample
            New sample with embedded features and ``data_type='tabular'``.
        """
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
                except (RuntimeError, TypeError):
                    pass  # heterogeneous shapes/types — keep as list
            if isinstance(x_batch, torch.Tensor):
                x_batch = x_batch.to(device)
            with torch.no_grad():
                embedded_x = model(x_batch)
            if isinstance(embedded_x, torch.Tensor):
                embedded_X.append(embedded_x.detach().cpu().numpy())
            else:
                embedded_X.append(np.asarray(embedded_x))

        embedded_X = np.concatenate(embedded_X, axis=0)

        embed_meta = dict(self.meta)

        new_sample = ConceptDatasetSample(
            parent=self.parent,
            inputs=embedded_X,
            C=self.base_concepts,
            y=self.base_labels,
            meta=embed_meta,
            input_type="tabular",
            classes=self.classes,
            indices=self.indices,
            has_concept_noise=self.has_concept_noise,
            has_concept_missing=self.has_concept_missing,
            has_label_noise=self.has_label_noise,
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
    """Image-backed variant of :class:`ConceptDatasetSample`.

    ``X`` stores file paths rather than pixel arrays.  ``__getitem__``
    loads each image from disk, applies *preprocess* (e.g. resize/normalize),
    then *transform*.

    Parameters
    ----------
    X : np.ndarray
        Array of image file paths (relative to *base_dir*).
    C : np.ndarray
        Binary concept matrix.
    y : np.ndarray
        Integer label vector.
    meta : dict
        Metadata dictionary.
    parent : ConceptDataset, optional
        Back-reference to the owning dataset.
    indices : np.ndarray, optional
        Boolean selection mask.
    transform : callable, optional
        Transform applied after *preprocess*.
    concept_transform : callable, optional
        Concept transform applied in ``__getitem__``.
    target_transform : callable, optional
        Target transform applied in ``__getitem__``.
    preprocess : callable, optional
        Image preprocessing (e.g. ``torchvision.transforms``).
    base_dir : Path or str, optional
        Root directory for resolving image paths (default ``"."``).
    **kwargs
        Extra keyword arguments forwarded to the parent class.
    """

    def __init__(
        self,
        inputs: np.ndarray,
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
            inputs=inputs,
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
        """Load image from disk and return ``(image, c, y)``."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, c, y = self.inputs[idx], self.C[idx], self.y[idx]

        if self.base_dir is not None:
            img_path = self.base_dir / img_path
        try:
            image = Image.open(io.BytesIO(Path(img_path).read_bytes())).convert("RGB")
            if self.preprocess is not None:
                image = self.preprocess(image)
            if self.transform is not None:
                image = self.transform(image)
        except (AttributeError, FileNotFoundError, OSError) as e:
            warnings.warn(f"{e!r}; cannot open image, returning path", RuntimeWarning)
            image = img_path

        c = torch.as_tensor(c, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.int64)

        if self.concept_transform is not None:
            c = self.concept_transform(c)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return image, c, y

    def to_dataframe(self, include_X: bool = False) -> "pd.DataFrame":
        """Convert to DataFrame, resolving image paths when *include_X* is set.

        When ``include_X=True``, an ``image`` column with absolute paths is
        prepended (resolving filenames via ``self.base_dir``).
        """
        parts: list[pd.DataFrame] = []
        if include_X:
            resolved = [str(self.base_dir / p) for p in self.inputs]
            parts.append(pd.DataFrame({"image": resolved}))
        parts.append(pd.DataFrame(self.C, columns=self.concepts))
        df = pd.concat(parts, axis=1)
        df["label"] = self.y
        df["class"] = [self.classes[int(i)] for i in self.y]
        return df

    def __eq__(self, other):
        chk = super().__eq__(other) and (self.base_dir == other.base_dir)
        return chk

    def __repr__(self):
        lines = [
            f"ConceptImageDatasetSample(image, {self.n} samples, {self.n_concepts} concepts)",
            "",
            _data_preview(self),
        ]
        return "\n".join(lines)

    def filter(self, indices):
        """Return a filtered copy, preserving *preprocess* and *base_dir*.

        Parameters
        ----------
        indices : np.ndarray
            Boolean mask of length ``n``.

        Returns
        -------
        ConceptImageDatasetSample
        """
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()

        filtered_meta = self.meta.copy()
        if "UC" in filtered_meta:
            filtered_meta["UC"] = filtered_meta["UC"][indices]
            filtered_meta["df_indices"] = filtered_meta["df_indices"][indices]
        if "robot_ids" in filtered_meta:
            filtered_meta["robot_ids"] = np.asarray(filtered_meta["robot_ids"])[indices]

        new_sample = self.__class__(
            parent=self.parent,
            inputs=self.inputs[indices],
            C=self.base_concepts[indices],
            y=self.base_labels[indices],
            meta=filtered_meta,
            input_type=self.input_type,
            classes=self.classes,
            indices=indices,
            has_concept_noise=self.has_concept_noise,
            has_concept_missing=self.has_concept_missing,
            has_label_noise=self.has_label_noise,
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
