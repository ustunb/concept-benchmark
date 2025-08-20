"""
This file contains helper functions that we use throughout the project
"""
import os
import dill
import re
import paths
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product, combinations
from itertools import chain, repeat, islice
from colorir import StackPalette, simplified_dist as colordist
from pero import Color

# general helper functions
def split_into_chunks(iterable, n, pad_value = None):
    """
    splits iterable into n evenly sized 'chunks', filling the remainder
    :param iterable: iterable
    :param n: chunks
    :param pad_value: value to pad the last chunk - set as None to include no padding - in this case the last chunk will have mod(len(iterable), n) chunks
    :return:
    """
    return list(zip(*[chain(iterable, repeat(pad_value, n - 1))]*n))

def to_list(el):
    if isinstance(el, list):
        return el
    elif isinstance(el, tuple):
        return list(el)
    elif el is None:
        return []
    else:
        return [el]

is_between = lambda x, a, b: np.logical_and(np.greater_equal(x, a), np.less_equal(x, b))

compute_error = lambda h, X, y: np.not_equal(h.predict(X), y).mean()

# file IO
def save_to_disk(session, **kwargs):
    """
    :param session: Flask session object
    :param path: directory for where to save file
    :param name: name of the file
    :param exist_ok: OK
    :return: True if saved
    """

    # parse file name
    path = kwargs.get('path', paths.results_dir)
    name = kwargs.get('name', '{}_session'.format(session.get('pid', 'unknown')))
    suffix = kwargs.get('suffix', '.pickle')
    filename = (Path(path) / Path(name)).with_suffix(suffix)

    overwrite = kwargs.get('overwrite', True)
    if not overwrite and filename.exists():
        if 'logger' in kwargs:
            kwargs['logger'].warning(
                    'failed to save session file\nfile already exists: {}'.format(filename)
                    )
    else:
        # pull objects to save
        contents = pull_experiment_state(session)
        with open(filename, 'wb') as f:
            dill.dump(contents, file = f, protocol = dill.HIGHEST_PROTOCOL)
        if 'logger' in kwargs:
            kwargs['logger'].warning(
                    "saved session content in file: {}".format(filename)
                    )

    return filename

def open_file(file_name):
    """opens a file on the operating system"""
    f = Path(file_name)
    if not f.is_file():
        raise IOError('file {} does not exist'.format(file_name))
    cmd = 'open "%s"' % str(file_name)
    os.system(cmd)

def pull_experiment_state(session):
    """
    pulls experimental parameters and results into a dictionary object that can be saved
    :param session: Flask session object
    :return:
    """
    state = {
        'pid': session['pid'],
        'params': session['params'],
        'results': session['results'],
        }
    return state

# Color Schemes
def generate_color_schemes(shuffle = True, random_seed = 123456, include_flipped = True):
    """
    generate color schemes used to color distinct Robots
    each color scheme is a tuple that contains the fill color for the (left, right) halves of a Robot
    :param shuffle: set to True to shuffle the schemes
    :param random_seed: random seed used to shuffle
    :param include_flipped: set to True to include (right, left) color scheme in output
    :return:
    """

    # use spectral since we can easily drop similar colors
    pal = StackPalette.load("spectral")

    # add in "dark" colors from paired for extra colors
    qal = StackPalette.load("paired")
    qal = StackPalette([qal[i] for i in range(1, len(qal), 2)])

    schemes = []
    for i, a in enumerate(pal):
        drop_idx = np.arange(i-2, i + 3) # similar colors in spectral are nearby
        keep_idx = np.setdiff1d(np.arange(len(pal)), drop_idx).tolist()
        pool = StackPalette(pal[keep_idx]) & qal
        pairs = [(a, b) for b in pool if 250 < colordist(a, b)]
        schemes += pairs

    # add flipped pairs
    if include_flipped:
        schemes += [(b, a) for a, b in schemes]

    # convert to Pero colors
    schemes = [(Color(a.hex()), Color(b.hex())) for a, b in schemes]

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(schemes)

    return schemes


# Rule Comparisons
def check_comparability(rule1, rule2):
    # Split the input rules into a list of features
    conds1 = rule1.split('OR')
    conds2 = rule2.split('OR')
    prev = None
    # Check if any OR term in rule1 is in rule2
    for cond1 in conds1:
        if any(cond1.strip() == cond2.strip() for cond2 in conds2):
            continue
        if any(cond1.strip() in cond2.strip() for cond2 in conds2):
            if prev == -1:
                return 0
            prev = 1
        if any(cond2.strip() in cond1.strip() for cond2 in conds2):
            if prev == 1:
                return 0
            prev = -1
    return 1


def calculate_scores(rule, compare_rule):
    # Split each rule into a list of OR conditions
    or_conditions = rule.split('OR')
    compare_or_conditions = compare_rule.split('OR')

    if not(check_comparability(rule, compare_rule)):
        compare_and_conditions = [[f.strip() for f in c.split('AND')] for c in compare_or_conditions]
        return len(compare_or_conditions), sum([len(c) for c in compare_and_conditions])

    # Check if any OR condition in rule is in compare_rule or vice versa
    shared_conditions, compare_shared_conditions = [], []
    for condition in or_conditions:
        for compare_condition in compare_or_conditions:
            if condition.strip() == compare_condition.strip() or condition.strip() in compare_condition.strip() or compare_condition.strip() in condition.strip():
                shared_conditions.append(condition)
                compare_shared_conditions.append(compare_condition)
    # Calculate generality score (number of additional OR conditions in compare_rule)
    generality_score = len(compare_or_conditions) - len(compare_shared_conditions)
    # Split shared conditions into a list of AND conditions
    and_conditions = [[f.strip() for f in c.split('AND')] for c in shared_conditions]
    compare_and_conditions = [[f.strip() for f in c.split('AND')] for c in compare_shared_conditions]
    # Calculate specificity score (number of additional AND conditions in compare_rule's AND conditions)
    scores = [len(set(cac).difference(set(ac))) for cac, ac in zip(compare_and_conditions, and_conditions)]
    generality_score += sum([abs(s) for s in scores if s < 0])
    additional_compare_and_conditions = [[f.strip() for f in c.split('AND')]
                                         for c in compare_or_conditions if c not in compare_shared_conditions]
    specificity_score = sum([s for s in scores if s >= 0])
    specificity_score += sum([len(c) for c in additional_compare_and_conditions])
    return generality_score, specificity_score


def levenshtein_distance(rule1, rule2, signed=False):
    rule1 = rule1.split(' OR ')
    rule2 = rule2.split(' OR ')
    m = len(rule1)
    n = len(rule2)
    # Create a matrix of distances
    dist = [[0 for j in range(n)] for i in range(m)]
    signed_dist = [[0 for j in range(n)] for i in range(m)]
    # Fill in the rest of the matrix
    for i in range(m):
        for j in range(n):
            # Compute the cost of each operation
            cost = len(set(rule1[i].split(' AND ')).symmetric_difference(set(rule2[j].split(' AND '))))
            num_to_remove = len(set(rule1[i].split(' AND ')).difference(set(rule2[j].split(' AND '))))
            num_to_add = len(set(rule2[j].split(' AND ')).difference(set(rule1[i].split(' AND '))))
            signed_cost = num_to_add - num_to_remove
            signed_dist[i][j] = signed_cost
            dist[i][j] = cost
    from itertools import permutations

    if n >= m:
        all_idx = list(permutations(list(range(n)), m))
        costs = []
        signed_costs = []
        for idx in all_idx:
            cost = 0
            signed_cost = 0
            for i in range(m):
                cost += dist[i][idx[i]]
                signed_cost += signed_dist[i][idx[i]]
            costs.append(cost)
            signed_costs.append(signed_cost)
    else:
        all_idx = list(permutations(list(range(m)), n))
        costs = []
        signed_costs = []
        for idx in all_idx:
            cost = 0
            signed_cost = 0
            for i in range(n):
                cost += dist[idx[i]][i]
                signed_cost += signed_dist[idx[i]][i]
            costs.append(cost)
            signed_costs.append(signed_cost)

    min_cost = min(costs)
    min_idx = all_idx[np.argmin(costs)]
    min_signed_cost = signed_costs[np.argmin(costs)]
    # compute the sum of and conditions in the terms not chosen in min_idx
    if n >= m:
        for i in range(n):
            if i not in min_idx:
                min_cost += len(rule2[i].split(' AND '))
                min_signed_cost += len(rule2[i].split(' AND '))
    else:
        for i in range(m):
            if i not in min_idx:
                min_cost += len(rule1[i].split(' AND '))
                min_signed_cost -= len(rule1[i].split(' AND '))

    # Return the final distance
    if signed:
        return min_signed_cost
    return min_cost


def l1_vector_distance(v1, v2, signed=False):
    """
    Computes the L1 distance between two vectors.
    """
    if signed:
        return np.sum(v1 - v2)
    else:
        return np.sum(np.abs(v1 - v2))


def rule_similarity(rule1, rule2):
    """
    Computes the similarity between two rules in a string format.
    The rules are a mix of ANDs and ORs with terms of the form VALUE FEATURE.
    """
    # Split each rule into its component parts
    rule1_parts = list(chain.from_iterable(islice(rule1.split(), i, i + 2) for i in range(0, len(rule1.split()), 3)))
    rule2_parts = list(chain.from_iterable(islice(rule2.split(), i, i + 2) for i in range(0, len(rule2.split()), 3)))

    # Compute the intersection and union of the features in the two rules
    intersection = set(rule1_parts).intersection(set(rule2_parts))
    union = set(rule1_parts).union(set(rule2_parts))

    # Compute the Jaccard similarity between the two sets
    similarity = float(len(intersection)) / len(union)

    return similarity


def sort_terms(rule, terms):
    # Split the input rule into a list of features
    features = rule.split('OR')
    # Create a dictionary to store the similarity scores for each term
    scores = {term: 0 for term in terms}
    # Loop over each feature in the input rule
    for feature in features:
        # Split the feature into a list of sub-features
        sub_features = feature.split('AND')
        # Calculate the similarity score between each sub-feature and each term
        for sub_feature in sub_features:
            for term in terms:
                value, char = sub_feature.split()
                score = 2 if sub_feature in term else 1 if char in term else 0
                scores[term] += score
    # Sort the terms by their total similarity score in descending order
    sorted_terms = sorted(terms, key=lambda term: scores[term], reverse=True)
    return sorted_terms

def unlist0(obj):
    if type(obj) in [list, tuple]:
        return obj[0]
    else:
        return obj

def linear_model_to_shap(model_info, features, random_state):
    """
    Converts a linear model to a SHAP explanation
    :param model_info: the model information
    :param features: the features
    :return: list of triples (feature, value, shap value), mean value
    """
    conditions=model_info['conditions']
    shap_feature_importance = []
    mean_val = 0
    max_neg = 0
    random_noises = list(random_state.uniform(0, 0.5, size=len(conditions)).round(3))
    for feature, value, coeff in conditions:
        is_present = features.get(feature, None) == value
        rval = features.get(feature, None)
        # 0.5 is the mean value of every condition on a feature in the roboto catalog, since all features are binary
        if is_present:
            shap_val = int(coeff) * (1 - 0.5)
        else:
            shap_val = int(coeff) * (0 - 0.5)
        random_noise = random_noises.pop()
        max_neg -= int(coeff)
        # we can add to positive shap values to even strengthen the prediction
        # or subtract from negative shap values to weaken the prediction
        if shap_val >= 0:
            shap_val += random_noise
        else:
            shap_val -= random_noise
        shap_feature_importance.append((feature, rval, shap_val))
        mean_val += 0.5 * int(coeff)

    mean_val -= int(model_info['threshold'])

    return {'feature_importance': shap_feature_importance, "mean_val": mean_val, "max_negative_shap_sum": max_neg}


def add_unobserved_rows_to_df(df):
    """
    adds rows to a classification dataset DataFrame
    :param df: pandas DataFrame with d + 2 columns corresponding to
               (x_1,..,x_d, n_neg, n_pos) and at most 2^d rows

    :return: pandas.DataFrame with d + 2 columns corresponding to and exactly 2^d rows.
             rows are sorted based on the value of x1, x2,..., xd
    """

    d = df.shape[1] - 2
    X_names = df.columns[0:d].tolist()
    n_names = df.columns[d:d+2].tolist()

    # fill out data frame with missing values
    X_full = ordered_feature_matrix(d, add_intercept = False)
    df_full = pd.DataFrame(np.hstack((X_full, np.zeros((2 ** d, 2)))))
    df_full.columns = df.columns
    df = pd.concat([df, df_full])
    df = df.groupby(X_names).sum(n_names).reset_index()

    # sort and cast
    df = df.sort_values(by = X_names).astype(int)

    return df


#### Helper Functions

def ordered_feature_matrix(d, false_value = -1, add_intercept = True):
    """
    :param d: # of dimensions, not including the intercept
    :param false_value: value assigned x when x is false - must be 0 or -1
    :param add_intercept: if true, adds a column of 1

    :return: matrix of distinct feature vectors (2^d rows and d columns)
             rows correspond to distinct feature vectors and are ordered according to their x_type

             X[0]      = [0,..,0,0]
             X[1]      = [0,..,0,1]
             ...
             X[2**d-1] = [1,...1,1]

             if add_intercept == true, X has (d + 1) columns so it looks like:

             X[0]      = [1, 0,..,0,0]
             X[1]      = [1, 0,..,0,1]
             ...
             X[2**d-1] = [1, 1,...1,1]

    """
    assert isinstance(d, int) and d >= 1
    X = np.array(list(product([false_value, 1], repeat = d)), dtype = np.int8)
    if add_intercept:
        X = np.insert(X, obj = 0, values = 1, axis = 1)
    return X


def logistic(x):
    return 1 / (1 + np.exp(-x))


def model_to_logistic(model: str):
    """
    Extracts the arithmetic expression inside the first (...) before a comparison
    and returns it wrapped in the logistic function.
    """
    # find whatever is inside (...) before a comparison operator
    match = re.search(r'\((.+?)\s*>=', model)
    if not match:
        raise ValueError("Model string not in expected format.")

    expr = match.group(1).strip()
    #print(f"Returning: expit({expr})")
    return f"expit({expr})"