"""Robot catalog construction for semantic identities and render instances."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from concept_benchmark.config import (
    RobotRenderNuisanceConfig,
    RobotValidationChecksConfig,
)

from . import textgen as text_helper
from .robot_draw import (
    ALL_ROBOT_FEATURES,
    COLOR_SCHEMES,
    _apply_color_jitter,
    _color_to_hex,
    _resolve_color_scheme,
    blur_parts,
    compute_robot_geometry,
    draw_robot,
    render_state_from_metadata,
    sample_robot_render_state,
    validate_robot_render,
)

OUTCOME_NAME = "robot_type"
OUTCOME_MISSING = "?"


def _stable_hash(payload: Any, *, prefix: str, truncate: int = 16) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"{prefix}_{hashlib.sha256(blob).hexdigest()[:truncate]}"


def _stable_int(payload: Any) -> int:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def _concept_product_df(concepts: dict[str, Sequence[Any]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(concepts.values(), names=concepts.keys())
    return pd.DataFrame(index=index).reset_index()


def get_robot_catalog_df(concepts, repetitions=1):
    """Create the legacy cartesian robot catalog."""

    indices = [
        pd.MultiIndex.from_product(concepts.values(), names=concepts.keys())
        for _ in range(repetitions)
    ]
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    df = pd.DataFrame(index=index).reset_index()
    df["color_scheme"] = np.mod(df.index, len(COLOR_SCHEMES))
    df["id"] = df.index
    df[OUTCOME_NAME] = OUTCOME_MISSING
    return df


def collapse_robot_subtypes(
    df,
    robot_features=ALL_ROBOT_FEATURES,
    subtype_separator="_",
    collapse_as_new_feature=None,
):
    """Collapse feature values with subtypes into coarse feature types."""

    if collapse_as_new_feature is None:
        collapse_as_new_feature = []
    df_feature_names = [k for k in df.columns if k in robot_features]
    new_features = {}
    for name in df_feature_names:
        str_vals = df[name].astype(str)
        split_df = str_vals.str.split(subtype_separator, n=1, expand=True)
        if split_df.shape[1] == 2:
            types_col = split_df[0].values
            subtypes_col = split_df[1].values
            df[name] = types_col
            df[name + "_subtype"] = subtypes_col
            if collapse_as_new_feature and name in collapse_as_new_feature:
                subtype_values = pd.Series(subtypes_col).unique()
                types = pd.Series(types_col).unique()
                for t in types:
                    type_mask = types_col == t
                    for sv in subtype_values:
                        new_feature_name = f"{name}_{t}_{sv}"
                        if new_feature_name not in df.columns:
                            new_features[new_feature_name] = [False, True]
                        df[new_feature_name] = (
                            (subtypes_col == sv) & type_mask
                        ).astype(str)
    return df, new_features


def convert_to_grayscale(image_path):
    """Convert a saved image to grayscale."""

    img = Image.open(image_path).convert("L")
    img.save(image_path)


def _semantic_payload(row: pd.Series | dict[str, Any], concept_names: Sequence[str]) -> dict[str, str]:
    source = row.to_dict() if isinstance(row, pd.Series) else row
    return {name: str(source[name]) for name in concept_names}


def _semantic_id(row: pd.Series | dict[str, Any], concept_names: Sequence[str]) -> str:
    return _stable_hash(_semantic_payload(row, concept_names), prefix="sem")


def _foot_orientation_for_key(payload: Any) -> str:
    return "vertex" if (_stable_int(payload) & 1) else "side"


def _render_state_metadata(
    render_state,
    *,
    requested_mode: str,
    color_scheme_id: int,
    validation_passed: bool,
    validation_attempts: int,
    validation_fail_reason: str | None,
    validation_used_fallback: bool,
    validation_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(render_state.to_metadata())
    metadata["render_space_mode"] = str(requested_mode)
    metadata["color_scheme"] = int(color_scheme_id)
    metadata["render_id"] = _stable_hash(
        {"render": metadata, "color_scheme": int(color_scheme_id)},
        prefix="rnd",
    )
    metadata["validation_passed"] = bool(validation_passed)
    metadata["validation_attempts"] = int(validation_attempts)
    metadata["validation_fail_reason"] = validation_fail_reason or ""
    metadata["validation_used_fallback"] = bool(validation_used_fallback)
    metadata["validation_failed_check"] = ""
    metadata["validation_enabled_checks"] = ""
    metadata["validation_min_clearance_pair"] = ""
    metadata["validation_min_clearance_px"] = np.nan
    if validation_stats:
        metadata["validation_failed_check"] = str(
            validation_stats.get("failed_check", "") or ""
        )
        enabled_checks = validation_stats.get("enabled_checks", ())
        metadata["validation_enabled_checks"] = "|".join(
            str(item) for item in enabled_checks
        )
        metadata["validation_min_clearance_pair"] = str(
            validation_stats.get("min_clearance_pair", "") or ""
        )
        min_clearance_px = validation_stats.get("min_clearance_px", np.nan)
        metadata["validation_min_clearance_px"] = (
            float(min_clearance_px)
            if pd.notna(min_clearance_px)
            else np.nan
        )
        metadata["validation_foreground_fraction"] = float(
            validation_stats.get("foreground_fraction", 0.0)
        )
        areas = validation_stats.get("part_areas", {})
        metadata["validation_body_area"] = int(areas.get("body", 0))
        metadata["validation_head_area"] = int(areas.get("head", 0))
        metadata["validation_mouth_area"] = int(areas.get("mouth", 0))
        metadata["validation_feet_area"] = int(areas.get("feet", 0))
    else:
        metadata["validation_foreground_fraction"] = np.nan
        metadata["validation_body_area"] = 0
        metadata["validation_head_area"] = 0
        metadata["validation_mouth_area"] = 0
        metadata["validation_feet_area"] = 0
    return metadata


def _resolve_colors_for_row(row: pd.Series) -> tuple[str, str]:
    left, right = _resolve_color_scheme(int(row["color_scheme"]))
    left = _apply_color_jitter(left, float(row.get("color_jitter", 0.0)))
    right = _apply_color_jitter(right, -float(row.get("color_jitter", 0.0)))
    return _color_to_hex(left), _color_to_hex(right)


def build_robot_instance_catalog(
    *,
    concepts: dict[str, Sequence[Any]],
    num_robots: int | None = None,
    resolution: int = 224,
    seed: int = 0,
    render_space_mode: str = "legacy",
    render_nuisance: RobotRenderNuisanceConfig | dict[str, Any] | None = None,
    validation_checks: RobotValidationChecksConfig | dict[str, Any] | None = None,
    validate_renders: bool = True,
    max_render_validation_attempts: int = 8,
) -> pd.DataFrame:
    """Construct the full per-instance robot catalog before subtype collapse."""

    if not concepts:
        raise ValueError("concepts dictionary must be provided and non-empty")

    concept_names = list(concepts.keys())
    num_unique_robots = int(np.prod([len(v) for v in concepts.values()]))
    total_robots = int(num_robots or num_unique_robots)

    if render_space_mode == "legacy":
        repetitions = int(np.ceil(float(total_robots) / num_unique_robots))
        df = get_robot_catalog_df(concepts=concepts, repetitions=repetitions).iloc[
            :total_robots
        ].copy()
        for name, values in concepts.items():
            if len(values) == 1:
                df = df.query(f"{name}=='{values[0]}'")
        df = df.reset_index(drop=True)
        df["id"] = df.index
        df["semantic_id"] = df.apply(
            lambda row: _semantic_id(row, concept_names), axis=1
        )
        df["semantic_index"] = df.groupby("semantic_id").ngroup()
        df["render_index_within_semantic"] = df.groupby("semantic_id").cumcount()
        df["foot_orientation"] = df["id"].map(
            lambda value: _foot_orientation_for_key({"legacy_id": int(value)})
        )
        legacy_rows = []
        for _, row in df.iterrows():
            render_state = sample_robot_render_state(
                render_space_mode="legacy",
                foot_orientation=str(row["foot_orientation"]),
            )
            meta = _render_state_metadata(
                render_state,
                requested_mode="legacy",
                color_scheme_id=int(row["color_scheme"]),
                validation_passed=True,
                validation_attempts=0,
                validation_fail_reason=None,
                validation_used_fallback=False,
                validation_stats=None,
            )
            payload = row.to_dict()
            payload.update(meta)
            payload["instance_id"] = f"legacy_{int(row['id']):06d}"
            payload.update(text_helper.pose_metadata_from_row(payload))
            legacy_rows.append(payload)
        catalog_df = pd.DataFrame(legacy_rows)
        catalog_df["render_id"] = catalog_df["render_id"].astype(str)
        catalog_df["instance_id"] = catalog_df["instance_id"].astype(str)
        colors = catalog_df.apply(_resolve_colors_for_row, axis=1, result_type="expand")
        catalog_df["color_left"] = colors[0]
        catalog_df["color_right"] = colors[1]
        catalog_df[OUTCOME_NAME] = OUTCOME_MISSING
        return catalog_df

    nuisance = render_nuisance or RobotRenderNuisanceConfig.for_mode(render_space_mode)
    if isinstance(nuisance, dict):
        nuisance = RobotRenderNuisanceConfig(**nuisance)
    nuisance.validate()
    if isinstance(validation_checks, dict):
        validation_checks = RobotValidationChecksConfig(**validation_checks)
    validation_checks = validation_checks or RobotValidationChecksConfig()
    validation_checks.validate()

    semantic_df = _concept_product_df(concepts)
    semantic_df["semantic_id"] = semantic_df.apply(
        lambda row: _semantic_id(row, concept_names), axis=1
    )
    semantic_df["semantic_index"] = np.arange(len(semantic_df), dtype=int)

    order: list[int] = []
    cycle = 0
    while len(order) < total_robots:
        rng = np.random.default_rng(seed + cycle)
        perm = rng.permutation(len(semantic_df)).tolist()
        order.extend(perm)
        cycle += 1
    order = order[:total_robots]

    counts_by_semantic: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for instance_index, semantic_row_index in enumerate(order):
        semantic_row = semantic_df.iloc[int(semantic_row_index)]
        semantic_payload = semantic_row.to_dict()
        semantic_id = str(semantic_payload["semantic_id"])
        repeat_idx = counts_by_semantic.get(semantic_id, 0)
        counts_by_semantic[semantic_id] = repeat_idx + 1

        base_seed_payload = {
            "seed": int(seed),
            "semantic_id": semantic_id,
            "repeat_idx": int(repeat_idx),
            "instance_index": int(instance_index),
        }
        render_seed = _stable_int(base_seed_payload)
        color_scheme_id = render_seed % len(COLOR_SCHEMES)
        foot_orientation = _foot_orientation_for_key(base_seed_payload)

        chosen_state = None
        chosen_stats = None
        chosen_reason = None
        validation_passed = not validate_renders
        validation_attempts = 0
        validation_used_fallback = False

        for attempt in range(max_render_validation_attempts):
            validation_attempts = attempt + 1
            render_state = sample_robot_render_state(
                seed=render_seed + attempt,
                render_space_mode=render_space_mode,
                render_nuisance=nuisance,
                foot_orientation=foot_orientation,
            )
            if not validate_renders:
                chosen_state = render_state
                chosen_stats = None
                chosen_reason = None
                validation_passed = True
                break
            geometry = compute_robot_geometry(
                render_state,
                width=resolution,
                height=resolution,
                color_scheme=color_scheme_id,
                **_semantic_payload(semantic_row, concept_names),
            )
            passed, reason, stats = validate_robot_render(
                geometry=geometry,
                validation_checks=validation_checks,
            )
            if passed:
                chosen_state = render_state
                chosen_stats = stats
                chosen_reason = None
                validation_passed = True
                break
            chosen_reason = reason
            chosen_stats = stats

        if chosen_state is None:
            validation_used_fallback = True
            chosen_state = sample_robot_render_state(
                render_space_mode="legacy",
                foot_orientation=foot_orientation,
            )
            geometry = compute_robot_geometry(
                chosen_state,
                width=resolution,
                height=resolution,
                color_scheme=color_scheme_id,
                **_semantic_payload(semantic_row, concept_names),
            )
            passed, _, stats = validate_robot_render(
                geometry=geometry,
                validation_checks=validation_checks,
            )
            validation_passed = bool(passed)
            chosen_stats = stats

        meta = _render_state_metadata(
            chosen_state,
            requested_mode=render_space_mode,
            color_scheme_id=int(color_scheme_id),
            validation_passed=validation_passed,
            validation_attempts=validation_attempts,
            validation_fail_reason=chosen_reason,
            validation_used_fallback=validation_used_fallback,
            validation_stats=chosen_stats,
        )

        row = _semantic_payload(semantic_row, concept_names)
        row.update(
            {
                "id": int(instance_index),
                "semantic_id": semantic_id,
                "semantic_index": int(semantic_payload["semantic_index"]),
                "render_index_within_semantic": int(repeat_idx),
                "instance_id": _stable_hash(
                    {"semantic_id": semantic_id, "render_id": meta["render_id"]},
                    prefix="inst",
                ),
            }
        )
        row.update(meta)
        row.update(text_helper.pose_metadata_from_row(row))
        rows.append(row)

    catalog_df = pd.DataFrame(rows)
    colors = catalog_df.apply(_resolve_colors_for_row, axis=1, result_type="expand")
    catalog_df["color_left"] = colors[0]
    catalog_df["color_right"] = colors[1]
    catalog_df[OUTCOME_NAME] = OUTCOME_MISSING
    return catalog_df


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
    seed: int = 0,
    render_space_mode: str = "legacy",
    render_nuisance: RobotRenderNuisanceConfig | dict[str, Any] | None = None,
    validation_checks: RobotValidationChecksConfig | dict[str, Any] | None = None,
    validate_renders: bool = True,
    max_render_validation_attempts: int = 8,
    **unused,
):
    """Generate the tabular robot catalog and optionally draw robot images."""

    if verbose:
        print("Starting robot generation...")

    init_catalog_df = build_robot_instance_catalog(
        concepts=concepts,
        num_robots=num_robots,
        resolution=resolution,
        seed=seed,
        render_space_mode=render_space_mode,
        render_nuisance=render_nuisance,
        validation_checks=validation_checks,
        validate_renders=validate_renders,
        max_render_validation_attempts=max_render_validation_attempts,
    )

    catalog_df = init_catalog_df.copy()
    catalog_df, new_features = collapse_robot_subtypes(
        df=catalog_df,
        robot_features=list(concepts.keys()),
        collapse_as_new_feature=additional_features or [],
    )

    concept_like_columns = {
        col
        for col in catalog_df.columns
        if col in concepts or col in new_features
    }
    constant_cols = [
        col
        for col in concept_like_columns
        if catalog_df[col].nunique() == 1 and col not in {"id", "png_filename"}
    ]
    catalog_df = catalog_df.drop(columns=constant_cols)
    new_features = {k: v for k, v in new_features.items() if k not in constant_cols}

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    png_filenames = []
    n_skipped = 0
    n_generated = 0

    iterator = init_catalog_df.iterrows()
    for _, features in tqdm(
        iterator,
        total=len(init_catalog_df),
        desc="Drawing robots" if draw else "Building catalog",
        disable=not draw,
    ):
        if render_space_mode == "legacy":
            png_filename = f"robot_{int(features['id']):03d}.png"
        else:
            png_filename = f"robot_{features['instance_id']}.png"
        png_file = output_path / png_filename

        if draw and not png_file.exists():
            render_state = render_state_from_metadata(features.to_dict())
            png_robot = draw_robot(
                filetype="png",
                width=resolution,
                height=resolution,
                render_state=render_state,
                **features.to_dict(),
            )
            if blur:
                parts = tuple(blur.get("parts", ("hands",)))
                radius = float(blur.get("radius", 2.0))
                expand = blur.get("expand_mask_px", None)
                feather = float(blur.get("feather_mask_px", 0.0))
                mode = blur.get("mask_mode", "uniform_rect")
                blurred = blur_parts(
                    png_robot,
                    parts=parts,
                    radius=radius,
                    expand_mask_px=expand,
                    feather_mask_px=feather,
                    mask_mode=mode,
                    render_state=render_state,
                    **features.to_dict(),
                )
                blurred.save(str(png_file))
            else:
                png_robot.export(str(png_file))

            if color_mode in ["grayscale", "greyscale"]:
                convert_to_grayscale(str(png_file))
            n_generated += 1
        elif draw:
            n_skipped += 1

        png_filenames.append(png_filename)

    if draw and verbose:
        print(f"Images: {n_generated} generated, {n_skipped} skipped (already existed)")

    catalog_df["png_filename"] = png_filenames
    if "color_left" not in catalog_df.columns:
        catalog_df["color_left"] = init_catalog_df["color_left"].values
    if "color_right" not in catalog_df.columns:
        catalog_df["color_right"] = init_catalog_df["color_right"].values
    return catalog_df, new_features
