"""
This file contains classes to represent and manipulate a set of all possible robots
"""
from __future__ import annotations

import copy
from pathlib import Path
from collections.abc import Sequence
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .robot_draw import (
    ALL_ROBOT_FEATURES,
    COLOR_SCHEMES,
    draw_robot,
    blur_parts,
)

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
    df, robot_features=ALL_ROBOT_FEATURES, subtype_separator="_", collapse_as_new_feature=[]
):
    """
    collapses feature values with subtypes into feature_types
    """
    df_feature_names = [k for k in df.columns if k in robot_features]
    separate = lambda x: pd.Series(str(x).split(subtype_separator))
    new_features = {}
    for name in df_feature_names:
        sf = df[name].apply(separate)
        if sf.shape[1] == 2:  # has subtypes
            df[name] = sf.loc[:, 0].values
            df[name + "_subtype"] = sf.loc[:, 1].values
            if collapse_as_new_feature:
                # create as many new features as there are subtypes per type
                subtype_values = sf.loc[:, 1].unique()
                types = sf.loc[:, 0].unique()
                for t in types:
                    if f"{name}_subtype" in collapse_as_new_feature:
                        for sv in subtype_values:
                            new_feature_name = f"{name}_{t}_{sv}"
                            print(f"Creating new feature {new_feature_name}")
                            if new_feature_name not in df.columns:
                                df[new_feature_name] = False
                                new_features[new_feature_name] = [False, True]
                            df.loc[df[name] == t, new_feature_name] = (sf.loc[:, 1] == sv) & (sf.loc[:, 0] == t) # this call ensures that you only get True if the type matches
                            df[new_feature_name] = df[new_feature_name].astype(str)
    return df, new_features


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

def convert_to_grayscale(image_path):
    """Convert saved image to grayscale"""
    img = Image.open(image_path).convert("L")  # L = grayscale
    img.save(image_path)


def generate_robot_catalog(
    *,
    concepts: dict,
    num_robots: int | None = None,
    resolution: int = 224,
    output_directory: str | Path = ".static/images",
    draw: bool = False,
    color_mode: str = "color",
    blur: dict | None = None,
    additional_features: Sequence[str] | None = None,
    verbose: bool = False,
    **unused,
):
    """Generate the tabular robot catalog and optionally draw robot images."""

    if not concepts:
        raise ValueError("concepts dictionary must be provided and non-empty")

    if verbose:
        print("Starting robot generation...")

    num_unique_robots = int(np.prod([len(v) for v in concepts.values()]))
    total_robots = num_robots or num_unique_robots
    catalog_df = get_robot_catalog_df(
        concepts=concepts,
        repetitions=int(np.ceil(float(total_robots) / num_unique_robots)),
    )

    # filter robot catalog so that you only see differences in selected features
    for name, values in concepts.items():
        if len(values) == 1:
            query_cmd = "{}=='{}'".format(name, values[0])
            catalog_df = catalog_df.query(query_cmd)

    init_catalog_df = copy.deepcopy(catalog_df)

    ids = init_catalog_df["id"].astype(str)
    hb = ids.map(lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") & 1).astype(int)
    init_catalog_df["foot_orientation"] = np.where(hb.values == 1, "vertex", "side")
    if verbose:
        u, c = np.unique(init_catalog_df["foot_orientation"], return_counts=True)
        print("foot_orientation_counts:", dict(zip(u, c)))

    catalog_df, new_features = collapse_robot_subtypes(
        df=catalog_df, robot_features=list(concepts.keys()),
        collapse_as_new_feature=additional_features or [],
    )
    print(f"After collapse: {new_features}")

    constant_cols = [
        col
        for col in catalog_df.columns
        if catalog_df[col].nunique() == 1 and col not in ["id", "png_filename"]
    ]
    catalog_df = catalog_df.drop(columns=constant_cols)
    new_features = {k: v for k, v in new_features.items() if k not in constant_cols}

    # create local directories
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    png_filenames = []
    n_skipped = 0
    n_generated = 0

    if draw and verbose:
        print(f"Drawing up to {len(catalog_df)} robot images (skipping existing)...")

    color_lefts, color_rights = [], []
    for k, features in tqdm(init_catalog_df.iterrows(), total=len(catalog_df)):
        png_filename = f"robot_{k:03d}.png"

        png_file = output_path / png_filename

        if draw and not png_file.exists():
            # Generate robot images
            png_robot = draw_robot(
                filetype="png",
                width=resolution,
                height=resolution,
                **features,
            )

            # Apply optional blur (body/hands/feet, etc.) and save
            if blur:
                parts = tuple(blur.get("parts", ("hands",)))
                radius = float(blur.get("radius", 2.0))
                expand = blur.get("expand_mask_px", None)
                feather = float(blur.get("feather_mask_px", 0.0))
                mode = blur.get("mask_mode", "uniform_rect")
                # features ensures mask matches geometry
                blurred = blur_parts(
                    png_robot,
                    parts=parts,
                    radius=radius,
                    expand_mask_px=expand,
                    feather_mask_px=feather,
                    mask_mode=mode,
                    **features,
                )
                blurred.save(str(png_file))
            else:
                # Save original image
                png_robot.export(str(png_file))

            if color_mode in ["grayscale", "greyscale"]:
                convert_to_grayscale(str(png_file))
            n_generated += 1
        elif draw:
            n_skipped += 1

        png_filenames.append(png_filename)
        color_scheme_id = np.mod(
            features["color_scheme"], len(COLOR_SCHEMES)
        )
        color_left, color_right = COLOR_SCHEMES[color_scheme_id]
        color_lefts.append(color_left)
        color_rights.append(color_right)

    if draw and verbose:
        print(f"Images: {n_generated} generated, {n_skipped} skipped (already existed)")

    # Add filename columns
    catalog_df["png_filename"] = png_filenames

    # Add color columns
    catalog_df["color_left"] = color_lefts
    catalog_df["color_right"] = color_rights

    ids2 = catalog_df["id"].astype(str)
    hb2 = ids2.map(lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") & 1).astype(int)
    catalog_df["foot_orientation"] = np.where(hb2.values == 1, "vertex", "side")

    return catalog_df, new_features
