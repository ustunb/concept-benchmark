import numpy as np
import pandas as pd
from mlcroissant import Dataset

from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from dataclasses import dataclass, field
from .cv import validate_cvindices, generate_cvindices

class ConceptDataset(object):

    SAMPLE_TYPES = ("training", "validation", "test")
    
    def __init__(
        self, 
        X: np.ndarray, 
        C: np.ndarray, 
        y: np.ndarray,
        meta: dict,
        **kwargs
    ) -> None:

        # example meta dict
        # ex_meta = {
        #    "classes": ["label_a", "label_b", "label_c"],
        #    "concepts": ["concept_a", "concept_b", "concept_c"]
        # }
        # map one/multi-hot encoded to string values


        self._full = ConceptDatasetSample(parent=self, X=X, C=C, y=y, meta=meta)
        self._cvindices = kwargs.get("cvindices")


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
        return (self._full == other._full) and all(
            np.array_equal(self.cvindices[k], other.cvindices[k])
            for k in self.cvindices.keys()
        )

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"ConceptDataset<n={self.n}, n_concepts={self.n_concepts}, n_classes={self.n_classes}>"

    def __copy__(self):

        cpy = ConceptDataset(
            X=self.X,
            C=self.C,
            y=self.y,
            cvindices=self.cvindices,
        )

        return cpy

    @staticmethod
    def from_croissant(croissant_dataset):
        """
        Initialize the dataset from a Croissant dataset.
        
        Parameters:
        - croissant_dataset: An instance of a Croissant dataset.
        """
        pass
    
    @staticmethod
    def to_croissant():
        """
        Convert the dataset to a Croissant dataset.
        
        Returns:
        - An instance of a Croissant dataset.
        """
        pass

    #### INSTANCE VARIABLES
    @property
    def classes(self):
        return self._classes
    
    @property
    def concepts(self):
        return self._concepts

    @property
    def n(self):
        """number of examples in full dataset"""
        return self._full.n

    @property
    def n_concepts(self):
        return len(self.concepts)
    
    @property
    def n_classes(self):
        return len(self.classes)

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
        assert (
            self._cvindices is not None
        ), "cannot set fold_id on a BinaryClassificationDataset without cvindices"
        assert isinstance(fold_id, str), f"fold_id={fold_id} should be string"
        assert (
            fold_id in self.cvindices
        ), f"cvindices does not contain fols for fold_id=`{fold_id}`"
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
            strata=strata,
            total_folds_for_cv=total_folds_for_cv,
            total_folds_for_inner_cv=total_folds_for_inner_cv,
            replicates=replicates,
            seed=seed,
        )
        self.cvindices = indices

@dataclass
class ConceptDatasetSample(Dataset):
    parent: ConceptDataset
    X: np.ndarray
    C: np.ndarray
    y: np.ndarray
    meta: dict
    indices: np.ndarray = None

    def __post_init__(self):

        assert ("classes", "concepts", "data_type") in self.meta.keys(), \
            "metedata dict must contain keys 'classes', 'concepts', and 'data_type'"

        self.classes, self.concepts = self.meta["classes"], self.meta["concepts"]
        self.task = self.meta["data_type"]   # image, tabular, etc...

        assert len(self.X.shape) >= 2, \
            "X must have at least 2 dimensions"

        assert self.C.shape[0] == self.X.shape[0], \
            "number of concepts does not match number of samples in X"

        assert self.y.max() <= self.classes, \
            "number of classes in y exceeds number of classes in metadata"

        self.n = self.X.shape[0]

        if self.indices is None:
            self.indices = np.ones(self.n, dtype=np.bool_)
        else:
            self.indices = self.indices.flatten().astype(np.bool_)

        self.loader = None

        assert self.__check_rep__()

    def __len__(self):
        return self.n

    def __eq__(self, other):
        chk = (
            isinstance(other, ConceptDatasetSample)
            and np.array_equal(self.y, other.y)
            and np.array_equal(self.X, other.X)
        )
        return chk

    def __check_rep__(self):
        """returns True is object satisfies representation invariants"""
        assert isinstance(self.X, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        assert self.n == len(self.y)
        assert np.sum(self.indices) == self.n
        assert np.isfinite(self.X).all()
        assert np.isin(
            self.y, self.classes
        ).all(), "y values must be stored as {}".format(self.classes)
        return True

    def __getitem__(self, idx):
        x = self.X[idx]
        c = self.C[idx]
        y = self.y[idx]
        
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        if isinstance(c, np.ndarray):
            c = c.astype(np.float32)
        if isinstance(y, np.ndarray):
            y = y.astype(np.int64)
        
        return x, c, y

    @property
    def n_concepts(self):
        return len(self.concepts)
    
    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def loader(self, batch_size=32, shuffle=False):
        """returns a DataLoader for this sample"""
        if self._loader is None or \
            self._loader.batch_size != batch_size or \
                self._loader.shuffle != shuffle:
            self._loader = DataLoader(
                self, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=0
            )

        return self._loader

    #### methods #####
    def filter(self, indices):
        """filters samples based on indices"""
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()
        return ConceptDatasetSample(
            parent=self.parent, X=self.X[indices], y=self.y[indices], indices=indices
        )