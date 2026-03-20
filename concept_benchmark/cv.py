"""
Helper functions to generate cross-validation indices for binary classification
"""

from __future__ import annotations

__all__ = [
    "parse_fold_id",
    "validate_fold_id",
    "is_inner_fold_id",
    "to_fold_id",
    "generate_folds",
    "generate_cvindices",
    "validate_folds",
    "validate_cvindices",
    "check_strata",
]

import logging
import re
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

#### fold id parsing / validation ####

TRIVIAL_FOLD_ID = "K01N01"
FOLD_ID_FORMAT = "K%02dN%02d"
INNER_ID_FORMAT = "F%02dK%02d"
INNER_CV_SEPARATOR = "_"
INNER_FOLD_ID_FORMAT = f"{FOLD_ID_FORMAT}{INNER_CV_SEPARATOR}{INNER_ID_FORMAT}"
OUTER_CV_PATTERN = "^K[0-9]{2}N[0-9]{2}$"
INNER_CV_PATTERN = "^K[0-9]{2}N[0-9]{2}_F[0-9]{2}K[0-9]{2}$"
OUTER_CV_PARSER = re.compile(OUTER_CV_PATTERN)
INNER_CV_PARSER = re.compile(INNER_CV_PATTERN)


def parse_fold_id(fold_id):
    """Parse a fold ID string into its components.

    Parameters
    ----------
    fold_id : str
        Fold identifier, e.g. ``"K05N01"`` (outer) or
        ``"K05N01_F02K03"`` (inner).

    Returns
    -------
    tuple
        ``(total_folds, replicate_idx, inner_fold_idx, inner_total_folds)``
        where the inner fields are ``None`` for outer fold IDs.
    """
    fold_id_elements = fold_id.split(INNER_CV_SEPARATOR)
    outer_fold_id = fold_id_elements[0]
    total_folds = int(outer_fold_id[1:3])
    replicate_idx = 1
    fold_idx_inner_cv = None
    total_folds_inner_cv = None

    if len(outer_fold_id) >= 4:
        replicate_idx = int(outer_fold_id[4:6])

    if len(fold_id_elements) > 1:
        inner_fold_id = fold_id_elements[1]
        fold_idx_inner_cv = int(inner_fold_id[1:3])
        total_folds_inner_cv = int(inner_fold_id[4:6])
        if fold_idx_inner_cv > total_folds:
            raise ValueError(
                f"inner_fold_id {fold_id} is for fold #{fold_idx_inner_cv}, which does not exist for outer fold_id {outer_fold_id}"
            )

    return total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv


def validate_fold_id(fold_id):
    """Validate and normalize a fold ID string.

    Parameters
    ----------
    fold_id : str
        Fold identifier to validate (case-insensitive).

    Returns
    -------
    str
        Uppercase, validated fold ID.

    Raises
    ------
    AssertionError
        If ``fold_id`` does not match outer or inner CV patterns.
    """

    fold_id = fold_id.strip().upper()
    parsed = INNER_CV_PARSER.match(fold_id)

    if parsed is not None:
        return parsed.string

    # must be outer-cv
    parsed = OUTER_CV_PARSER.match(fold_id)
    if parsed is None:
        raise ValueError(f"invalid fold_id: {fold_id}")

    return parsed.string


def is_inner_fold_id(fold_id):
    """Return ``True`` if *fold_id* represents an inner cross-validation fold.

    Parameters
    ----------
    fold_id : str
        Fold identifier to check.
    """
    parsed = INNER_CV_PARSER.match(fold_id)
    return parsed is not None


def to_fold_id(
    total_folds, replicate_idx=1, fold_idx_inner_cv=None, total_folds_inner_cv=None
):
    total_folds = int(total_folds)
    replicate_idx = int(replicate_idx)

    if total_folds < 1:
        raise ValueError("total_folds must be >= 1")
    if replicate_idx < 1:
        raise ValueError("replicate_idx must be >= 1")

    if fold_idx_inner_cv is None:
        fold_id = FOLD_ID_FORMAT % (total_folds, replicate_idx)
    else:
        fold_idx_inner_cv = int(fold_idx_inner_cv)
        total_folds_inner_cv = int(total_folds_inner_cv)
        if total_folds_inner_cv < 1:
            raise ValueError("total_folds_inner_cv must be >= 1")
        if fold_idx_inner_cv < 1:
            raise ValueError("fold_idx_inner_cv must be >= 1")
        if total_folds < fold_idx_inner_cv:
            raise ValueError("total_folds must be >= fold_idx_inner_cv")
        fold_id = INNER_FOLD_ID_FORMAT % (
            total_folds,
            replicate_idx,
            fold_idx_inner_cv,
            total_folds_inner_cv,
        )

    fold_id = validate_fold_id(fold_id)
    return fold_id


#### fold generation ####


def generate_folds(n_folds=5, n_samples=None, strata=None, random_state=None):
    """
    generate fold indices for standard or stratified K-fold CV

    :param n_folds: number of folds (i.e. K in K-fold CV)
                    must be a positive integer >= 2

    :param n_samples: size of the indices
                      must be a positive integer >= 2
                      n_samples is only used for standard CV

    :param strata: vector of indices that will be used for stratified CV,
                   must contain at least 2 distinct elements
                   n_samples is only used for standard CV

    :param random_state: random state for reproducibility

    :return: 1D of indices that can be used  indices that can be used for CV
    """
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    stratified = strata is not None
    if stratified:
        check_strata(strata)
        n_samples = len(strata)
        fold_generator = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        if not isinstance(n_samples, int) or n_samples < 2:
            raise ValueError("n_samples must be an integer >= 2")
        strata = np.empty(n_samples)
        fold_generator = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    folds = np.zeros(n_samples, dtype=int)
    for k, (train_idx, test_idx) in enumerate(fold_generator.split(X=strata, y=strata)):
        folds[test_idx] = k + 1

    # check folds
    folds = validate_folds(folds=folds, n_samples=n_samples, stratified=stratified)

    return folds


def generate_cvindices(
    n_samples=None,
    strata=None,
    total_folds_for_cv=None,
    total_folds_for_inner_cv=None,
    replicates=3,
    seed=None,
):
    """
    :param n_samples:
    :param strata:
    :param total_folds_for_cv:
    :param total_folds_for_inner_cv:
    :param replicates:
    :param seed:
    :return:
    """
    if total_folds_for_cv is None:
        total_folds_for_cv = [1, 2, 3, 5, 10]
    if total_folds_for_inner_cv is None:
        total_folds_for_inner_cv = [2, 3, 5]

    # type checks
    if not (
        isinstance(total_folds_for_cv, list)
        and len(total_folds_for_cv) > 0
        and (len(total_folds_for_cv) == len(set(total_folds_for_cv)))
    ):
        raise ValueError("total_folds_for_cv must be a non-empty list of unique integers")
    if total_folds_for_inner_cv is not None and not (
        isinstance(total_folds_for_inner_cv, list)
        and len(total_folds_for_inner_cv) == len(set(total_folds_for_inner_cv))
    ):
        raise ValueError("total_folds_for_inner_cv must be a list of unique integers")
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates should be a positive integer")

    # determine type of CV generation
    stratified = strata is not None
    if stratified:
        check_strata(strata)
        n_samples = len(strata)

    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # generate CV indices
    total_folds_for_cv = list(total_folds_for_cv)  # avoid mutating caller's list
    cvindices = dict()
    if 1 in total_folds_for_cv:
        cvindices[TRIVIAL_FOLD_ID] = np.ones(n_samples, dtype=int)
        total_folds_for_cv.remove(1)

    for k in total_folds_for_cv:
        for n in range(1, replicates + 1):
            fold_id = to_fold_id(total_folds=k, replicate_idx=n)
            if stratified:
                cvindices[fold_id] = generate_folds(n_folds=k, strata=strata, random_state=rng)
            else:
                cvindices[fold_id] = generate_folds(n_folds=k, n_samples=n_samples, random_state=rng)

            # generate inner folds for k-fold cv
            for f in range(1, k + 1):
                fold_idx = np.not_equal(cvindices[fold_id], f)
                n_samples_fold = np.sum(fold_idx)
                for num_inner_folds in total_folds_for_inner_cv:
                    inner_fold_id = to_fold_id(
                        total_folds=k,
                        replicate_idx=n,
                        fold_idx_inner_cv=f,
                        total_folds_inner_cv=num_inner_folds,
                    )
                    if stratified:
                        cvindices[inner_fold_id] = generate_folds(
                            n_folds=num_inner_folds, strata=strata[fold_idx], random_state=rng
                        )
                    else:
                        cvindices[inner_fold_id] = generate_folds(
                            n_folds=num_inner_folds, n_samples=n_samples_fold, random_state=rng
                        )

    cvindices = validate_cvindices(cvindices, stratified=stratified)
    return cvindices


#### checks and validation ####


def validate_folds(folds, fold_id=None, n_samples=None, stratified=True):
    """
    check folds used for cross-validation
    :param folds:
    :param fold_id:
    :param n_samples:
    :param stratified:
    :return: True
    """

    # reshape folds
    if not isinstance(folds, np.ndarray):
        raise ValueError("folds should be array-like")
    if folds.ndim != 1 or len(folds) < 1:
        raise ValueError("folds must be a non-empty 1D array")

    # check length
    if n_samples is not None and len(folds) != n_samples:
        raise ValueError(f"folds length {len(folds)} != n_samples {n_samples}")

    # check fold values
    fold_values, fold_counts = np.unique(folds, return_counts=True)
    fold_values_min = np.min(fold_values)
    fold_values_max = np.max(fold_values)
    if fold_values_min != 1:
        raise ValueError("fold values must start at 1")
    if fold_values_max > len(folds):
        raise ValueError("fold values must not exceed number of samples")

    if not np.array_equal(fold_values, np.arange(1, fold_values_max + 1)):
        raise ValueError(f"fold indices {fold_values} are not consecutive")

    if not stratified:
        if np.min(fold_counts) < np.max(fold_counts) - 1:
            raise ValueError(
                "imbalanced folds: max (points/fold) must be within min (points/fold)"
            )

    # check that fold id matches fold content
    if fold_id is not None:
        (total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv) = (
            parse_fold_id(fold_id)
        )
        if is_inner_fold_id(fold_id):
            if total_folds < 1:
                raise ValueError("total_folds must be >= 1")
            if total_folds < fold_idx_inner_cv:
                raise ValueError("total_folds must be >= fold_idx_inner_cv")
            if not np.isin(fold_idx_inner_cv, np.arange(1, total_folds + 1)):
                raise ValueError(f"fold_idx_inner_cv {fold_idx_inner_cv} not in range [1, {total_folds}]")
            if total_folds_inner_cv != fold_values_max:
                raise ValueError(f"total_folds_inner_cv {total_folds_inner_cv} != fold_values_max {fold_values_max}")
        else:
            if total_folds != fold_values_max:
                raise ValueError(f"total_folds {total_folds} != fold_values_max {fold_values_max}")
            if replicate_idx < 1:
                raise ValueError("replicate_idx must be >= 1")
            if fold_idx_inner_cv is not None:
                raise ValueError("fold_idx_inner_cv must be None for outer folds")
            if total_folds_inner_cv is not None:
                raise ValueError("total_folds_inner_cv must be None for outer folds")

    return folds


def validate_cvindices(cvindices, stratified=True):
    """
    will drop fold_ids for inner cv if the corresponding outer_cv fold_id does not exist
    :param cvindices:
    :return:
    """

    # check that fold_ids are valid
    all_fold_ids = list(cvindices.keys())
    for fold_id in all_fold_ids:
        try:
            validated_id = validate_fold_id(fold_id)
            if validated_id != fold_id:
                cvindices[validated_id] = cvindices.pop(fold_id)
        except ValueError:
            cvindices.pop(fold_id)

    all_fold_ids = list(cvindices.keys())
    outer_ids = list(filter(OUTER_CV_PARSER.match, all_fold_ids))
    inner_ids = list(filter(INNER_CV_PARSER.match, all_fold_ids))

    if len(outer_ids) == 0:
        if len(inner_ids) != 0:
            raise ValueError("inner fold IDs found without outer fold IDs")
        return cvindices

    # at this point cvindices must have at least one outer id
    validated_indices = dict()
    n_samples = len(cvindices[outer_ids[0]])
    for fold_id in outer_ids:
        try:
            validated_indices[fold_id] = validate_folds(
                cvindices[fold_id], fold_id, n_samples, stratified
            )
        except ValueError:
            logging.getLogger(__name__).warning("could not validate fold: %s", fold_id)

    for fold_id in inner_ids:
        outer_id, _ = fold_id.split(INNER_CV_SEPARATOR)
        if outer_id in outer_ids:
            try:
                validated_indices[fold_id] = validate_folds(
                    cvindices[fold_id], fold_id, stratified=stratified
                )
            except ValueError:
                logging.getLogger(__name__).warning(
                    "could not validate fold: %s", fold_id
                )

    return validated_indices


def check_strata(strata):
    """
    check vector used for stratified CV
    :param strata:
    :return:
    """
    if not isinstance(strata, np.ndarray):
        raise ValueError("strata should be array-like")
    if strata.ndim != 1:
        raise ValueError("strata should be 1 dimensional")
    if np.issubdtype(strata.dtype, np.number):
        if not np.isfinite(strata).all():
            raise ValueError("strata should be finite")
    if len(np.unique(strata)) < 2:
        raise ValueError("strata should contain at least 2 distinct classes")
    return True
