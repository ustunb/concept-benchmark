"""
Helper classes to represent and manipulate datasets for a binary classification task
"""

import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from .concept_benchmark.cv import validate_cvindices, generate_cvindices
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


class BinaryClassificationDataset(object):
    """class to represent/manipulate a dataset for a binary classification task"""

    SAMPLE_TYPES = ("training", "validation", "test")

    def __init__(self, X, y, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        """
        # complete dataset
        self._full = BinaryClassificationSample(parent=self, X=X, y=y)

        # variable names
        self._names = BinaryClassificationVariableNames(
            parent=self,
            y=kwargs.get("y_name", "y"),
            X=kwargs.get("X_names", [f"x{j:02d}" for j in range(1, self.d + 1)]),
        )

        # cvindices
        self._cvindices = kwargs.get("cvindices")
        self.n_pos = sum(y == 1)
        self.n_neg = sum(y == -1)

        # indicator to check if we have split into train, test, splits
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

        # check names
        assert self.names.__check_rep__()

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
        return f"ClassificationDataset<n={self.n}, d={self.d}>"

    def __copy__(self):

        cpy = BinaryClassificationDataset(
            X=self.X,
            y=self.y,
            X_names=self.names.X,
            y_name=self.names.y,
            cvindices=self.cvindices,
        )

        return cpy

    #### io functions ####
    @staticmethod
    def from_df(df, outcome_id=0, drop_na=True):

        assert isinstance(df, pd.DataFrame), "df must be a dataframe"
        column_ids = range(df.shape[1])
        assert (
            isinstance(outcome_id, int) and outcome_id in column_ids
        ), "invalid outcome_id"

        if drop_na:
            missing_idx = np.any(df.isna().values, axis=1)
            if np.any(missing_idx):
                warnings.warn(
                    f"dropping {np.sum(missing_idx)}/{len(missing_idx)} rows with NA values"
                )
                df = df.iloc[~missing_idx, :]

        names = df.columns.tolist()
        feature_indices = [j for j in column_ids if j != outcome_id]
        X = df.values[:, feature_indices]
        y = df.iloc[:, outcome_id].replace(0, -1).values

        return BinaryClassificationDataset(X=X, y=y, X_names=names[1:], y_name=names[0])

    @staticmethod
    def read_csv(data_file, **kwargs):
        """
        loads raw data from CSV
        :param data_file: Path to the data_file
        :param helper_file: Path to the helper_file or None.
        :return:
        """
        # extract common file header from dataset file
        file_header = str(data_file).rsplit("_data.csv")[0]

        # convert file names into path objects with the correct extension
        files = {
            "data": f"{file_header}_data",
            "helper": kwargs.get("helper_file", f"{file_header}_helper"),
            # 'weights': kwargs.get('weights_file', '{}_weight'.format(file_header)),
        }
        files = {k: Path(v).with_suffix(".csv") for k, v in files.items()}
        assert files["data"].is_file(), f"could not find data file: {files['data']}"
        if not files["helper"].is_file():
            out = BinaryClassificationDataset.from_df(df = pd.read_csv(files["data"], sep=","))
        else:
            # read helper file
            hf = pd.read_csv(files["helper"], sep=",")
            hf["is_variable"] = ~(hf["is_outcome"])
            hf_headers = ["is_outcome", "is_variable"]
            assert all(hf[hf_headers].isin([0, 1]))
            assert sum(hf["is_outcome"]) == 1, "helper file should specify 1 outcome"

            # parse names
            names = {
                "y": hf.query("is_outcome")["header"][0],
                "X": hf.query("is_variable")["header"].tolist(),
            }

            # read raw data from disk with expected datatypes
            dtypes = {names["y"]: int} | {n: float for n in names["X"]}
            df = pd.read_csv(files["data"], sep=",", dtype=dtypes)

            # cehck match
            df_names = df.columns.to_list()
            header_names = hf["header"].to_list()
            assert set(df_names) == set(header_names), \
                f"columns names must match names in helper file {files['helper']}"

            out = BinaryClassificationDataset.from_df(
                df, outcome_id=df_names.index(names["y"])
            )

        return out

    #### variable names ####
    @property
    def names(self):
        """pointer to names of X, y"""
        return self._names

    #### properties of the full dataset ####
    @property
    def n(self):
        """number of examples in full dataset"""
        return self._full.n

    @property
    def d(self):
        """number of features in full dataset"""
        return self._full.d

    @property
    def df(self):
        return self._full.df

    @property
    def X_df(self):
        return pd.DataFrame(self._full.X, columns=self._names.X)

    @property
    def X(self):
        """feature matrix"""
        return self._full.X

    @property
    def y(self):
        """label vector"""
        return self._full.y

    @property
    def U(self):
        """unique feature matrix"""
        return self._full.U

    @property
    def u_idx(self):
        """unique feature matrix"""
        return self._full.u_idx
    
    @property
    def inv(self):
        """inverse to map from unique to full feature matrix"""
        return self._full.inv
    
    @property
    def cnt(self):
        """counts of each unique data point in the full feature matrix"""
        return self._full.cnt

    @property
    def classes(self):
        return self._full.classes

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
class BinaryClassificationSample:
    """class to store and manipulate a subsample of points in a survival dataset"""

    parent: BinaryClassificationDataset
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray = None

    def __post_init__(self):

        self.classes = (-1, 1)
        self.X = np.atleast_2d(np.array(self.X, float))
        self.n = self.X.shape[0]
        self.n_pos = sum(self.y == 1)
        self.n_neg = sum(self.y == -1)
        self.d = self.X.shape[1]

        self.U, self.u_idx, self.inv, self.cnt = np.unique(
            self.X, 
            axis=0, 
            return_inverse=True, 
            return_counts=True,
            return_index=True
            )

        if self.indices is None:
            self.indices = np.ones(self.n, dtype=np.bool_)
        else:
            self.indices = self.indices.flatten().astype(np.bool_)

        self.update_classes(self.classes)
        assert self.__check_rep__()

    def __len__(self):
        return self.n

    def __eq__(self, other):
        chk = (
            isinstance(other, BinaryClassificationSample)
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

    def update_classes(self, values):
        assert len(values) == 2
        assert values[0] < values[1]
        assert isinstance(values, (np.ndarray, list, tuple))

        # change y encoding using new classes
        if self.n > 0:
            y = np.array(self.y, dtype=float).flatten()
            neg_idx = np.equal(y, self.classes[0])
            y[neg_idx] = values[0]
            y[~neg_idx] = values[1]
            self.classes = tuple(np.array(values, dtype=int))
            self.y = y

    @property
    def df(self):
        """
        pandas data.frame containing y, G, X for this sample
        """
        df = pd.DataFrame(self.X, columns=self.parent.names.X)
        df.insert(column=self.parent.names.y, value=self.y, loc=0)
        return df

    #### methods #####
    def filter(self, indices):
        """filters samples based on indices"""
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()
        return BinaryClassificationSample(
            parent=self.parent, X=self.X[indices], y=self.y[indices], indices=indices
        )


@dataclass
class BinaryClassificationVariableNames:
    """class to represent the names of features, group attributes, and the label in a classification task"""

    parent: BinaryClassificationDataset
    X: List[str] = field(repr=True)
    y: str = field(repr=True, default="y")

    def __post_init__(self):
        assert self.__check_rep__()

    @staticmethod
    def check_name_str(s):
        """check variable name"""
        return isinstance(s, str) and len(s.strip()) > 0

    def __check_rep__(self):
        """check if this object satisfies representation invariants"""

        assert isinstance(self.X, list) and all(
            [self.check_name_str(n) for n in self.X]
        ), "X must be a list of strings"
        assert len(self.X) == len(set(self.X)), "X must be a list of unique strings"
        assert self.check_name_str(self.y), "y must be at least 1 character long"
        return True


def undersample_by_label(data, random_state=None):
    """
    oversample dataset to equalize number of positive and negative labels in each group
    :param data:
    :param kwargs:
    :return:
    """
    rus = RandomUnderSampler(random_state=random_state)

    # generate resampled data
    Xr, yr = rus.fit_resample(data.X, data.y)

    rus_data = BinaryClassificationDataset(
        X=Xr,
        y=yr,
        X_names=data.names.X,
        y_name=data.names.y,
    )

    return rus_data


def oversample_by_label(data, random_state=None):
    """
    oversample dataset to equalize number of positive and negative labels in each group
    :param data:
    :param kwargs:
    :return:
    """
    ros = RandomOverSampler(random_state=random_state)

    # generate resampled data
    Xr, yr = ros.fit_resample(data.X, data.y)

    ros_data = BinaryClassificationDataset(
        X=Xr,
        y=yr,
        X_names=data.names.X,
        y_name=data.names.y,
    )

    return ros_data
