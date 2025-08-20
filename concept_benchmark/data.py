import warnings
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .cv import generate_cvindices, validate_cvindices


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
            **kwargs: Additional keyword arguments.
        """
        self._init_kwargs = dict(kwargs)

        if meta.get("data_type") == "image":
            SampleClass = ConceptImageDatasetSample
            # do not cast X
            C = C.astype(np.int8)
            y = y.astype(np.int32)
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
            **kwargs,
        )

        self._cvindices = cvindices
        self.reset()

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
        assert self.__check_rep__()

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
            C=self.C,
            y=self.y,
            meta=self._full.meta,
            cvindices=self._cvindices,
            **self._init_kwargs,
        )

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
        Embed the dataset using a given model.

        Parameters:
        - model: A model that can embed the dataset.

        Returns:
        - An embedded version of the dataset.
        """
        self._full = self._full.embed(
            model, batch_size=batch_size, shuffle=shuffle, device=device, **kwargs
        )

        # apply cv indices to the embedded dataset
        if self.fold_id is not None:
            self.split(
                fold_id=self.fold_id,
                fold_num_validation=self.fold_num_validation,
                fold_num_test=self.fold_num_test,
            )


@dataclass
class ConceptDatasetSample(Dataset):
    X: np.ndarray
    C: np.ndarray
    y: np.ndarray
    meta: dict
    parent: "ConceptDataset" = None
    indices: np.ndarray = None
    transform: Callable | None = None
    concept_transform: Callable | None = None
    target_transform: Callable | None = None

    def __post_init__(self):
        assert {"classes", "concepts", "data_type"}.issubset(self.meta.keys()), (
            "metedata dict must contain keys 'classes', 'concepts', and 'data_type'"
        )

        self.classes, self.concepts = self.meta["classes"], self.meta["concepts"]
        self.task = self.meta["data_type"]  # image, tabular, etc...

        self.n = len(self.X)

        if self.indices is None:
            self.indices = np.ones(self.n, dtype=np.bool_)
        else:
            self.indices = self.indices.flatten().astype(np.bool_)

        assert self.__check_rep__()

    def __len__(self):
        return self.n

    def __eq__(self, other):
        if not isinstance(other, ConceptDatasetSample):
            return False
        if not np.array_equal(self.y, other.y):
            return False
        if not np.array_equal(self.X, other.X):
            return False
        if not np.array_equal(self.C, other.C):
            return False
        # Use parent's deep comparator for meta
        if not _deep_equal(self.meta, other.meta):
            return False
        # Compare transforms by identity to avoid ambiguous equality
        if self.transform is not other.transform:
            return False
        if self.concept_transform is not other.concept_transform:
            return False
        if self.target_transform is not other.target_transform:
            return False
        return True

    def __check_rep__(self):
        """returns True is object satisfies representation invariants"""
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
        return f"ConceptDatasetSample<n={self.n}, n_concepts={self.n_concepts}, n_classes={self.n_classes}, data_type={self.meta.get('data_type')}>"

    @property
    def n_concepts(self):
        return len(self.concepts)

    @property
    def n_classes(self):
        return len(self.classes)

    #### methods #####
    def filter(self, indices):
        """filters samples based on indices"""
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()
        return self.__class__(
            parent=self.parent,
            X=self.X[indices],
            C=self.C[indices],
            y=self.y[indices],
            meta=self.meta,
            indices=indices,
            transform=self.transform,
            concept_transform=self.concept_transform,
            target_transform=self.target_transform,
        )

    def loader(self, batch_size=32, shuffle=False, **kwargs) -> DataLoader:
        """
        Returns a DataLoader for the dataset.

        Parameters:
        - batch_size (int): Size of each batch.
        - shuffle (bool): Whether to shuffle the data.
        - **kwargs: Additional keyword arguments for DataLoader.
        """
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return loader

    def embed(
        self, model, batch_size=32, shuffle=False, device="cpu", **kwargs
    ) -> "ConceptDatasetSample":
        """
        Embed the dataset using a given model.

        Parameters:
        - model: A model that can embed the dataset.

        Returns:
        - An embedded version of the dataset.
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
            # Normalize x_batch to a tensor batch if possible
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

        embed_meta = dict(self.meta)
        embed_meta["data_type"] = "tabular"

        return ConceptDatasetSample(
            parent=self.parent,
            X=embedded_X,
            C=self.C,
            y=self.y,
            meta=embed_meta,
            indices=self.indices,
        )


@dataclass
class ConceptImageDatasetSample(ConceptDatasetSample):
    """
    A sample of a ConceptDataset that contains image data.
    Inherits from ConceptDatasetSample.
    """

    base_dir: Path = field(default_factory=lambda: Path("."))

    def __post_init__(self):
        super().__post_init__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, c, y = self.X[idx], self.C[idx], self.y[idx]

        if self.base_dir is not None:
            img_path = self.base_dir / img_path
        try:
            image = Image.open(img_path).convert("RGB")
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
        return f"ConceptImageDatasetSample<n={self.n}, n_concepts={self.n_concepts}, n_classes={self.n_classes}, data_type={self.meta.get('data_type')}, base_dir={self.base_dir}>"

    def filter(self, indices):
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()
        return self.__class__(
            parent=self.parent,
            X=self.X[indices],
            C=self.C[indices],
            y=self.y[indices],
            meta=self.meta,
            indices=indices,
            transform=self.transform,
            concept_transform=self.concept_transform,
            target_transform=self.target_transform,
            base_dir=self.base_dir,
        )