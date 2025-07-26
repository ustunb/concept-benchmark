"""
Helper functions to generate cross-validation indices for binary classification
"""

import re
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

#### fold id parsing / validation ####

TRIVIAL_FOLD_ID = "K01N01"
FOLD_ID_FORMAT = 'K%02dN%02d'
INNER_ID_FORMAT = 'F%02dK%02d'
INNER_CV_SEPARATOR = "_"
INNER_FOLD_ID_FORMAT = '%s%s%s' % (FOLD_ID_FORMAT, INNER_CV_SEPARATOR, INNER_ID_FORMAT)
OUTER_CV_PATTERN = "^K[0-9]{2}N[0-9]{2}$"
INNER_CV_PATTERN = "^K[0-9]{2}N[0-9]{2}_F[0-9]{2}K[0-9]{2}$"
OUTER_CV_PARSER = re.compile(OUTER_CV_PATTERN)
INNER_CV_PARSER = re.compile(INNER_CV_PATTERN)

def parse_fold_id(fold_id):
    """
    #todo add spec
    :param fold_id:
    :return:
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
        assert fold_idx_inner_cv <= total_folds, f'inner_fold_id {fold_id} is for fold #{fold_idx_inner_cv}, which does not exist for outer fold_id {outer_fold_id}'

    return total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv

def validate_fold_id(fold_id):
    """
    #todo add spec
    :param fold_id:
    :return:
    """

    fold_id = fold_id.strip().upper()
    parsed = INNER_CV_PARSER.match(fold_id)

    if parsed is not None:
        return parsed.string

    # must be outer-cv
    parsed = OUTER_CV_PARSER.match(fold_id)
    assert parsed is not None, \
        'invalid fold_id: %s' % fold_id

    return parsed.string

def is_inner_fold_id(fold_id):
    """
    #todo add spec
    :param fold_id:
    :return:
    """
    parsed = INNER_CV_PARSER.match(fold_id)
    return parsed is not None

def to_fold_id(total_folds, replicate_idx = 1, fold_idx_inner_cv = None, total_folds_inner_cv = None):

    total_folds = int(total_folds)
    replicate_idx = int(replicate_idx)

    assert total_folds >= 1
    assert replicate_idx >= 1

    if fold_idx_inner_cv is None:
        fold_id = FOLD_ID_FORMAT % (total_folds, replicate_idx)
    else:
        fold_idx_inner_cv = int(fold_idx_inner_cv)
        total_folds_inner_cv = int(total_folds_inner_cv)
        assert total_folds_inner_cv >= 1
        assert fold_idx_inner_cv >= 1
        assert total_folds >= fold_idx_inner_cv
        fold_id = INNER_FOLD_ID_FORMAT % (total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv)

    fold_id = validate_fold_id(fold_id)
    return fold_id

#### fold generation ####

def generate_folds(n_folds = 5, n_samples = None, strata = None):
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

    :return: 1D of indices that can be used  indices that can be used for CV
    """
    assert isinstance(n_folds, int) and n_folds >= 2
    stratified = strata is not None
    if stratified:
        assert check_strata(strata)
        n_samples = len(strata)
        fold_generator = StratifiedKFold(n_splits = n_folds, shuffle = True)
    else:
        assert isinstance(n_samples, int) and n_samples >= 2
        strata = np.empty(n_samples)
        fold_generator = KFold(n_splits = n_folds, shuffle = True)

    folds = np.zeros(n_samples, dtype=int)
    for k, (train_idx, test_idx) in enumerate(fold_generator.split(X = strata, y = strata)):
        folds[test_idx] = k + 1

    # check folds
    folds = validate_folds(folds = folds, n_samples = n_samples, stratified = stratified)

    return folds

def generate_cvindices(n_samples = None, strata = None, total_folds_for_cv = [1, 2, 3, 5, 10], total_folds_for_inner_cv = [2, 3, 5], replicates = 3, seed = None):
    """
    :param n_samples:
    :param strata:
    :param total_folds_for_cv:
    :param total_folds_for_inner_cv:
    :param replicates:
    :param seed:
    :return:
    """

    # type checks
    assert isinstance(total_folds_for_cv, list) and len(total_folds_for_cv) > 0 and (len(total_folds_for_cv) == len(set(total_folds_for_cv)))
    assert (total_folds_for_inner_cv is None) or (isinstance(total_folds_for_inner_cv, list) and len(total_folds_for_inner_cv) == len(set(total_folds_for_inner_cv)))
    assert isinstance(replicates, int) and replicates >= 1, 'replicates should be a positive integer'

    # determine type of CV generation
    stratified = strata is not None
    if stratified:
        assert check_strata(strata)
        n_samples = len(strata)

    assert isinstance(n_samples, int) and n_samples >= 1

    if seed is not None:
        np.random.seed(seed)

    # generate CV indices
    cvindices = dict()
    if 1 in total_folds_for_cv:
        cvindices[TRIVIAL_FOLD_ID] = np.ones(n_samples, dtype=int)
        total_folds_for_cv.remove(1)

    for k in total_folds_for_cv:
        for n in range(1, replicates + 1):
            fold_id = to_fold_id(total_folds = k, replicate_idx = n)
            if stratified:
                cvindices[fold_id] = generate_folds(n_folds = k, strata = strata)
            else:
                cvindices[fold_id] = generate_folds(n_folds = k, n_samples = n_samples)

            # generate inner folds for k-fold cv
            for f in range(1, k + 1):
                fold_idx = np.not_equal(cvindices[fold_id], f)
                n_samples_fold = np.sum(fold_idx)
                for l in total_folds_for_inner_cv:
                    inner_fold_id = to_fold_id(
                            total_folds = k,
                            replicate_idx = n,
                            fold_idx_inner_cv = f,
                            total_folds_inner_cv = l
                            )
                    if stratified:
                        cvindices[inner_fold_id] = generate_folds(n_folds = l, strata = strata[fold_idx])
                    else:
                        cvindices[inner_fold_id] = generate_folds(n_folds = l, n_samples = n_samples_fold)

    cvindices = validate_cvindices(cvindices, stratified = stratified)
    return cvindices

#### checks and validation ####

def validate_folds(folds, fold_id = None, n_samples = None, stratified = True):
    """
    check folds used for cross-validation
    :param folds:
    :param fold_id:
    :param n_samples:
    :param stratified:
    :return: True
    """

    # reshape folds
    assert isinstance(folds, np.ndarray), 'folds should be array-like'
    assert folds.ndim == 1 and len(folds) >= 1

    # check length
    if n_samples is not None:
        assert len(folds) == n_samples

    # check fold values
    fold_values, fold_counts = np.unique(folds, return_counts = True)
    fold_values_min = np.min(fold_values)
    fold_values_max = np.max(fold_values)
    assert fold_values_min == 1
    assert fold_values_max <= len(folds)

    assert np.array_equal(fold_values, np.arange(1, fold_values_max + 1)), f'fold indices {fold_values} are not consecutive'

    if not stratified:
        assert np.min(fold_counts) >= np.max(fold_counts) - 1, 'imbalanced folds: max (points/fold) must be within min (points/fold)'

    # check that fold id matches fold content
    if fold_id is not None:
        (total_folds, replicate_idx, fold_idx_inner_cv, total_folds_inner_cv) = parse_fold_id(fold_id)
        if is_inner_fold_id(fold_id):
            assert total_folds >= 1
            assert total_folds >= fold_idx_inner_cv
            assert np.isin(fold_idx_inner_cv, np.arange(1, total_folds + 1))
            assert total_folds_inner_cv == fold_values_max
        else:
            assert total_folds == fold_values_max
            assert replicate_idx >= 1
            assert fold_idx_inner_cv is None
            assert total_folds_inner_cv is None

    return folds

def validate_cvindices(cvindices, stratified = True):
    """
    will drop fold_ids for inner cv if the corresponding outer_cv fold_id does not exist
    :param cvindices:
    :return:
    """

    #check that fold_ids are valid
    all_fold_ids = list(cvindices.keys())
    for fold_id in all_fold_ids:
        try:
            validated_id = validate_fold_id(fold_id)
            if validated_id != fold_id:
                cvindices[validated_id] = cvindices.pop(fold_id)
        except AssertionError:
            cvindices.pop(fold_id)

    all_fold_ids = list(cvindices.keys())
    outer_ids = list(filter(OUTER_CV_PARSER.match, all_fold_ids))
    inner_ids = list(filter(INNER_CV_PARSER.match, all_fold_ids))

    if len(outer_ids) == 0:
        assert len(inner_ids) == 0
        return cvindices

    #at this point cvindices must have at least one outer id
    validated_indices = dict()
    n_samples = len(cvindices[outer_ids[0]])
    for fold_id in outer_ids:
        try:
            validated_indices[fold_id] = validate_folds(cvindices[fold_id], fold_id, n_samples, stratified)
        except AssertionError:
            print('could not validate fold: %s' % fold_id)
            pass


    for fold_id in inner_ids:
        outer_id, _ = fold_id.split(INNER_CV_SEPARATOR)
        if outer_id in outer_ids:
            try:
                validated_indices[fold_id] = validate_folds(cvindices[fold_id], fold_id, stratified = stratified)
            except AssertionError:
                print('could not validate fold: %s' % fold_id)
                pass


    return validated_indices

def check_strata(strata):
    """
    check vector used for stratified CV
    :param strata:
    :return:
    """
    assert isinstance(strata, np.ndarray), 'strata should be array-like'
    assert strata.ndim == 1, 'strata should be 1 dimensional'
    if np.issubdtype(strata.dtype, np.number):
        assert np.isfinite(strata).all(), 'strata should be finite'
    assert len(np.unique(strata)) >= 2, 'strata should contain at least 2 distinct classes'
    return True