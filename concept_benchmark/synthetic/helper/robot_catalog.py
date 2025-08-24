"""
This file contains classes to represent and manipulate a set of all possible robots
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .robot_draw import ALL_ROBOT_FEATURES, COLOR_SCHEMES, ROBOT_TYPES, draw_robot

pd.options.mode.chained_assignment = None

OUTCOME_NAME = "robot_type"
OUTCOME_MISSING = "?"


def get_robot_catalog_df(concepts, repetitions=1):
    """
    create a dataframe containing all possible combinations of robot features
    this dataframe defines a unique ID for each robot that is used to call files - therefore it must contain all possible robots
    each column shows the value of a specific feature – e.g., head_shape, body_shape
    each row shows a distinct combination of features – i.e., a unique robot
    :return: pandas DataFrame
    """
    # todo: generate multiple catalogs given a parameter i that defines the number of repetitions
    index = pd.MultiIndex.from_product(concepts.values(), names=concepts.keys())
    for i in range(1, repetitions):
        new_index = pd.MultiIndex.from_product(concepts.values(), names=concepts.keys())
        index = index.append(new_index)

    df = pd.DataFrame(index=index).reset_index()
    df["color_scheme"] = np.mod(df.index, len(COLOR_SCHEMES))
    df["id"] = df.index
    df[OUTCOME_NAME] = OUTCOME_MISSING
    return df


def collapse_robot_subtypes(
    df, robot_features=ALL_ROBOT_FEATURES, subtype_separator="_"
):
    """
    collapses feature values with subtypes into feature_types
    """
    df_feature_names = [k for k in df.columns if k in robot_features]
    separate = lambda x: pd.Series(str(x).split(subtype_separator))
    for name in df_feature_names:
        sf = df[name].apply(separate)
        if sf.shape[1] == 2:  # has subtypes
            df[name] = sf.loc[:, 0].values
            df[name + "_subtype"] = sf.loc[:, 1].values
            try:
                sf.loc[:, 0].empty or sf.loc[:, 1].empty is False
            except:
                Exception()
    return df


def add_irrelevant_feature(df, feature_name="has_elbows", values=[True, False]):
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

    STAGE_NAMES = ("anchoring", "probing", "deployment")

    def __init__(self, df, outcome_name=OUTCOME_NAME, outcome_values=ROBOT_TYPES):
        assert isinstance(df, pd.DataFrame)
        assert isinstance(outcome_name, str) and len(outcome_name) >= 1
        assert (
            isinstance(outcome_values, (list, tuple, set))
            and len(set(outcome_values)) >= 1
        )

        # properties
        self._outcome_name = str(outcome_name)
        self._outcome_values = [str(k) for k in outcome_values]
        self._feature_names = [
            str(k) for k in df.columns if k in ALL_ROBOT_FEATURES.keys()
        ]
        self._target_outcome = str(ROBOT_TYPES[0])

        # data frame
        self._df = pd.DataFrame(df)
        self._df[self._outcome_name] = OUTCOME_MISSING
        self._df[list("k_%s" % s for s in self.STAGE_NAMES)] = 0

        # Define positive (indicator) value per feature using collapsed base values
        robot_feats = {
            k: [v.split("_")[0] for v in l] for k, l in ALL_ROBOT_FEATURES.items()
        }
        self._positive_value_by_feature = {
            k: robot_feats[k][0] for k in self._feature_names if k in robot_feats
        }

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
    def positive_value_by_feature(self):
        """Mapping from feature name to the value treated as 1 in binary encoding."""
        return dict(self._positive_value_by_feature)

    # Note: Binary concept encoding is computed where needed (e.g., in main.py)


def convert_to_grayscale(image_path):
    """Convert saved image to grayscale"""
    img = Image.open(image_path).convert("L")  # L = grayscale
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
    num_robots = params["num_robots"] if "num_robots" in params else 96
    num_unique_robots = np.prod([len(v) for v in params["concepts"].values()])
    catalog_df = get_robot_catalog_df(
        concepts=params["concepts"],
        repetitions=int(np.ceil(float(num_robots) / num_unique_robots)),
    )

    # filter robot catalog so that you only see differences in selected features
    for name, values in params["concepts"].items():
        if len(values) == 1:
            query_cmd = "{}=='{}'".format(name, values[0])
            catalog_df = catalog_df.query(query_cmd)

    init_catalog_df = copy.deepcopy(catalog_df)

    if (
        "irrelevant_features" in params
        and len(params["irrelevant_features"]) > 0
        and drop_irrelevant
    ):
        # check if irrelevant featuers are in the catalog
        existing_irrelevant_features = [
            f for f in params["irrelevant_features"] if f in catalog_df.columns
        ]
        if existing_irrelevant_features:
            catalog_df.drop(columns=existing_irrelevant_features, inplace=True)

    catalog_df = collapse_robot_subtypes(
        df=catalog_df, robot_features=list(params["concepts"].keys())
    )
    constant_cols = [
        col
        for col in catalog_df.columns
        if catalog_df[col].nunique() == 1 and col not in ["id", "png_filename"]
    ]
    catalog_df = catalog_df.drop(columns=constant_cols)

    # create local directories
    output_path = Path(params["output_directory"])
    output_path.mkdir(parents=True, exist_ok=True)

    # delete old image files if they exist
    if params.get("draw", False):
        for file in output_path.glob("robot_[0-9][0-9][0-9].*"):
            file.unlink()

        print(f"Generating {len(catalog_df)} robot images...")
    png_filenames = []

    for k, features in init_catalog_df.iterrows():
        png_filename = f"robot_{k:03d}.png"
        color_lefts, color_rights = [], []

        png_file = output_path / png_filename

        if params.get("draw", False):
            # Generate robot images
            png_robot = draw_robot(
                filetype="png",
                width=params["resolution"],
                height=params["resolution"],
                **features,
            )

            # Save images
            png_robot.export(str(png_file))

            if params.get("color_mode") in ["grayscale", "greyscale"]:
                convert_to_grayscale(str(png_file))

        png_filenames.append(png_filename)
        color_scheme_id = np.mod(
            features["color_scheme"], len(COLOR_SCHEMES)
        )
        color_left, color_right = COLOR_SCHEMES[color_scheme_id]
        color_lefts.append(color_left)
        color_rights.append(color_right)

    # Add filename columns
    catalog_df["png_filename"] = png_filenames

    # Add color columns
    catalog_df["color_left"] = color_lefts
    catalog_df["color_right"] = color_rights


    return catalog_df
