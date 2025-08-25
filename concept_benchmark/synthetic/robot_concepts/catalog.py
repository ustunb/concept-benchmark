"""
This file contains classes to represent and manipulate a set of all possible robots
"""
import copy

import numpy as np
import pandas as pd
from .robots import ALL_ROBOT_FEATURES, COLOR_SCHEMES, ROBOT_TYPES
from .dataset import ClassificationDataset, DatasetBinarizer
from .robots import draw_robot
from PIL import Image
from pathlib import Path

pd.options.mode.chained_assignment = None

OUTCOME_NAME = 'robot_type'
OUTCOME_MISSING = '?'

drop_constant_cols = lambda df: df.loc[:, (df != df.iloc[0]).any()]


def get_robot_catalog_df(concepts, repetitions=1):
    """
    create a dataframe containing all possible combinations of robot features
    this dataframe defines a unique ID for each robot that is used to call files - therefore it must contain all possible robots
    each column shows the value of a specific feature – e.g., head_shape, body_shape
    each row shows a distinct combination of features – i.e., a unique robot
    :return: pandas DataFrame
    """
    #todo: generate multiple catalogs given a parameter i that defines the number of repetitions
    index = pd.MultiIndex.from_product(concepts.values(), names=concepts.keys())
    for i in range(1, repetitions):
        new_index = pd.MultiIndex.from_product(concepts.values(), names=concepts.keys())
        index = index.append(new_index)

    df = pd.DataFrame(index = index).reset_index()
    df['color_scheme'] = np.mod(df.index, len(COLOR_SCHEMES))
    df['id'] = df.index
    df[OUTCOME_NAME] = OUTCOME_MISSING
    return df


def collapse_robot_subtypes(df, robot_features = ALL_ROBOT_FEATURES, subtype_separator = '_'):
    """
    collapses feature values with subtypes into feature_types
    """
    df_feature_names = [k for k in df.columns if k in robot_features]
    separate = lambda x: pd.Series(str(x).split(subtype_separator))
    for name in df_feature_names:
        sf = df[name].apply(separate)
        if sf.shape[1] == 2: #has subtypes
            df[name] = sf.loc[:, 0].values
            df[name+'_subtype'] = sf.loc[:, 1].values
            try:
                sf.loc[:, 0].empty or sf.loc[:, 1].empty is False
            except:
                Exception()
    return df


def add_irrelevant_feature(df, feature_name = 'has_elbows', values = [True, False]):
    """
    :param df:
    :param feature_name:
    :param values:
    :return:
    """

    raise NotImplementedError()
    # todo berk: only implement this if you have a way of ensuring that the feature does not change
    assert isinstance(df, pd.DataFrame)
    assert isinstance(feature_name, str) and len(feature_name) >= 1
    assert feature_name not in df.columns
    df[feature_name] = values[0]
    df_list = [df]
    for v in values[1:]:
        df_new = df.copy()
        df_new[feature_name] = v
        df_list.append(df)

    return pd.concat(df_list)


class RobotDistribution(object):
    """class to represent, manipulate, and sample from probability distributions over robots"""
    STAGE_NAMES = ('anchoring', 'probing', 'deployment')
    def __init__(self, df, outcome_name = OUTCOME_NAME, outcome_values = ROBOT_TYPES):

        assert isinstance(df, pd.DataFrame)
        assert isinstance(outcome_name, str) and len(outcome_name) >= 1
        assert isinstance(outcome_values, (list, tuple, set)) and len(set(outcome_values)) >= 1

        # properties
        self._outcome_name = str(outcome_name)
        self._outcome_values = [str(k) for k in outcome_values]
        self._feature_names = [str(k) for k in df.columns if k in ALL_ROBOT_FEATURES.keys()]
        self._target_outcome = str(ROBOT_TYPES[0])

        # data frame
        self._df = pd.DataFrame(df)
        self._df[self._outcome_name] = OUTCOME_MISSING
        self._df[list('k_%s' % s for s in self.STAGE_NAMES)] = 0

        # binarizer
        # collapse features values
        robot_feats = {k: [v.split('_')[0] for v in l] for k, l in ALL_ROBOT_FEATURES.items()}
        self._binarizer = DatasetBinarizer(outcome_name = self._outcome_name, outcome_values = self._outcome_values, feature_names = self._feature_names, feature_values = robot_feats)

    @property
    def df(self):
        return self._df

    @property
    def target_outcome(self):
        return self._target_outcome

    @property
    def outcome_name(self):
        """outcome column"""
        return self._outcome_name

    @property
    def feature_names(self):
        """feature columns"""
        return self._feature_names

    @property
    def binarizer(self):
        """binarizer"""
        return self._binarizer

    def to_dataset(self, df = None, count_column = None, **kwargs):
        """
        produces a df with training data for a binary classification task using
        df with information about the joint distribution P(A, Y)

        :param count_column:

        :return: df of training data for a classification task with binary input
                 variables and binary outcome variable.

        """
        if df is None:
            df = self._df

        # create df with categorical X, y values
        label_column = kwargs.get('label_column', self._outcome_name)
        df_cat = df[self._feature_names + [label_column]].rename(columns = {label_column: self._outcome_name})

        # repeat df rows based on count column
        if count_column is not None:
            n_reps = df['k_%s' % count_column].astype(int)
            idx = df.index.repeat(n_reps)
            df_cat = df_cat.loc[idx]

        # convert df into classification dataset
        out = ClassificationDataset(df_cat = df_cat, binarizer = self.binarizer.copy())

        return out


def convert_to_grayscale(image_path):
    """Convert saved image to grayscale"""
    img = Image.open(image_path).convert('L')  # L = grayscale
    img.save(image_path)

def generate_robot_catalog(params, drop_irrelevant=True):
    """
    :params: Dictionary containing robot generation parameters including:
            - num_robots: Number of robots to generate
            - concepts: Dictionary of feature names and possible values
            - model: String defining the ground truth labeling function
    :return:
    """
    # get the number of generated robot images to determine the number of repetitions for the robot catalog
    print("Starting robot generation...")
    print(params)
    num_robots = params['num_robots'] if 'num_robots' in params else 96
    num_unique_robots = np.prod([len(v) for v in params['concepts'].values()])
    catalog_df = get_robot_catalog_df(concepts=params['concepts'],
                                      repetitions=int(np.ceil(float(num_robots) / num_unique_robots)))

    # filter robot catalog so that you only see differences in selected features
    for name, values in params['concepts'].items():
        if len(values) == 1:
            query_cmd = "{}=='{}'".format(name, values[0])
            catalog_df = catalog_df.query(query_cmd)

    init_catalog_df = copy.deepcopy(catalog_df)

    if 'irrelevant_features' in params and len(params['irrelevant_features']) > 0 and drop_irrelevant:
        # check if irrelevant featuers are in the catalog
        existing_irrelevant_features = [f for f in params['irrelevant_features'] if f in catalog_df.columns]
        if existing_irrelevant_features:
            catalog_df.drop(columns = existing_irrelevant_features, inplace = True)

    catalog_df = collapse_robot_subtypes(df=catalog_df, robot_features=list(params['concepts'].keys()))
    catalog_df = drop_constant_cols(catalog_df)

    # create local directories
    output_path = Path(params['output_directory'])
    output_path.mkdir(parents=True, exist_ok=True)

    # delete old image files if they exist
    if params.get('draw', False):
        for file in output_path.glob('robot_[0-9][0-9][0-9].*'):
            file.unlink()

        print(f"Generating {len(catalog_df)} robot images...")
    png_filenames = []

    for k, features in init_catalog_df.iterrows():
        png_filename = f'robot_{k:03d}.png'

        png_file = output_path / png_filename

        if params.get('draw', False):
            # Generate robot images
            png_robot = draw_robot(filetype='png', width=params['resolution'], height=params['resolution'], **features)

            # Save images
            png_robot.export(str(png_file))

            if params.get('color_mode') in ['grayscale', 'greyscale']:
                convert_to_grayscale(str(png_file))

        png_filenames.append(png_filename)

    # Add filename columns
    catalog_df['png_filename'] = png_filenames

    return catalog_df