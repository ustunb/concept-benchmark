from __future__ import annotations

import itertools

from concept_benchmark.config import (
    ROBOT_CONCEPTS,
    RobotRenderNuisanceConfig,
    RobotValidationChecksConfig,
)
from concept_benchmark.synthetic.helper.robot_catalog import build_robot_instance_catalog
from concept_benchmark.synthetic.helper.robot_draw import (
    RobotRenderState,
    compute_robot_geometry,
    render_state_from_metadata,
    validate_robot_render,
)


SMALL_SINGLE_SEMANTIC = {k: v[:1] for k, v in ROBOT_CONCEPTS.items()}


def _single_semantic_features() -> dict[str, str]:
    return {name: values[0] for name, values in SMALL_SINGLE_SEMANTIC.items()}


def _find_hands_body_only_collision_state() -> RobotRenderState:
    features = _single_semantic_features()
    angles = [60.0, 70.0, 80.0, 90.0]
    arm_offsets = [-0.25, -0.20, -0.15, -0.10]
    head_offsets = [-0.06, -0.02, 0.00, 0.04]
    arm_lengths = [0.90, 1.00, 1.10]
    head_offsets_x = [-0.08, -0.04, 0.00, 0.04]
    for angle, arm_y, head_y, arm_length, head_x in itertools.product(
        angles,
        arm_offsets,
        head_offsets,
        arm_lengths,
        head_offsets_x,
    ):
        state = RobotRenderState(
            mode="continuous_light",
            arm_angle_offset_deg=angle,
            arm_length_scale=arm_length,
            arm_y_offset_frac=arm_y,
            head_offset_y_frac=head_y,
            head_offset_x_frac=head_x,
        )
        geometry = compute_robot_geometry(
            state,
            width=32,
            height=32,
            color_scheme=0,
            **features,
        )
        checks_on = RobotValidationChecksConfig()
        passed, reason, _ = validate_robot_render(
            geometry=geometry,
            validation_checks=checks_on,
        )
        if passed or reason != "hands_body_clearance":
            continue
        checks_off = RobotValidationChecksConfig(hands_body_clearance="off")
        passed, _, _ = validate_robot_render(
            geometry=geometry,
            validation_checks=checks_off,
        )
        if passed:
            return state
    raise AssertionError("failed to find a deterministic hands/body-only collision pose")


def test_pathological_render_settings_fall_back_to_legacy():
    nuisance = RobotRenderNuisanceConfig.continuous_light()
    nuisance.translate_x_frac = [0.75, 0.75]
    nuisance.global_scale = [1.15, 1.15]
    df = build_robot_instance_catalog(
        concepts=SMALL_SINGLE_SEMANTIC,
        num_robots=1,
        resolution=32,
        seed=3,
        render_space_mode="continuous_light",
        render_nuisance=nuisance,
        validate_renders=True,
        max_render_validation_attempts=2,
    )
    row = df.iloc[0]
    assert bool(row["validation_used_fallback"])
    assert row["accepted_render_space_mode"] == "legacy"
    assert bool(row["validation_passed"])
    assert row["validation_fail_reason"] == "bbox_out_of_frame"


def test_accepted_continuous_renders_pass_validation():
    df = build_robot_instance_catalog(
        concepts=SMALL_SINGLE_SEMANTIC,
        num_robots=5,
        resolution=32,
        seed=21,
        render_space_mode="continuous_light",
        validate_renders=True,
    )
    for _, row in df.iterrows():
        render_state = render_state_from_metadata(row.to_dict())
        semantic_features = {
            name: row[name]
            for name in SMALL_SINGLE_SEMANTIC
        }
        geometry = compute_robot_geometry(
            render_state,
            width=32,
            height=32,
            color_scheme=int(row["color_scheme"]),
            **semantic_features,
        )
        passed, reason, _ = validate_robot_render(geometry=geometry)
        assert passed, reason


def test_validation_metadata_columns_exist():
    df = build_robot_instance_catalog(
        concepts=SMALL_SINGLE_SEMANTIC,
        num_robots=2,
        resolution=32,
        seed=8,
        render_space_mode="continuous_light",
        validate_renders=True,
    )
    expected = {
        "semantic_id",
        "render_id",
        "instance_id",
        "validation_passed",
        "validation_attempts",
        "validation_fail_reason",
        "validation_failed_check",
        "validation_used_fallback",
        "validation_enabled_checks",
        "validation_min_clearance_pair",
        "validation_min_clearance_px",
        "validation_foreground_fraction",
        "validation_body_area",
        "validation_head_area",
        "validation_mouth_area",
        "validation_feet_area",
    }
    assert expected.issubset(set(df.columns))


def test_individual_collision_check_can_be_disabled():
    state = _find_hands_body_only_collision_state()
    geometry = compute_robot_geometry(
        state,
        width=32,
        height=32,
        color_scheme=0,
        **_single_semantic_features(),
    )
    passed, reason, stats = validate_robot_render(
        geometry=geometry,
        validation_checks=RobotValidationChecksConfig(),
    )
    assert not passed
    assert reason == "hands_body_clearance"
    assert stats["failed_check"] == "hands_body_clearance"

    passed, reason_off, stats_off = validate_robot_render(
        geometry=geometry,
        validation_checks=RobotValidationChecksConfig(hands_body_clearance="off"),
    )
    assert passed, reason_off
    assert stats_off["failed_check"] == ""


def test_asymmetric_arm_pose_can_trigger_hands_head_clearance_only():
    state = RobotRenderState(
        mode="continuous_light",
        arm_angle_offset_deg=-10.0,
        right_arm_angle_delta_deg=220.0,
        left_arm_angle_delta_deg=-120.0,
        right_arm_length_scale=1.0,
        right_arm_y_offset_frac=-0.10,
        left_arm_y_offset_frac=0.10,
        head_offset_x_frac=-0.02,
        head_offset_y_frac=-0.02,
    )
    geometry = compute_robot_geometry(
        state,
        width=32,
        height=32,
        color_scheme=0,
        **_single_semantic_features(),
    )
    passed, reason, stats = validate_robot_render(
        geometry=geometry,
        validation_checks=RobotValidationChecksConfig(),
    )
    assert not passed
    assert reason == "hands_head_clearance"
    assert stats["failed_check"] == "hands_head_clearance"

    passed_off, reason_off, _ = validate_robot_render(
        geometry=geometry,
        validation_checks=RobotValidationChecksConfig(hands_head_clearance="off"),
    )
    assert passed_off, reason_off


def test_validate_renders_false_disables_all_checks():
    nuisance = RobotRenderNuisanceConfig.continuous_light()
    nuisance.translate_x_frac = [0.75, 0.75]
    df = build_robot_instance_catalog(
        concepts=SMALL_SINGLE_SEMANTIC,
        num_robots=1,
        resolution=32,
        seed=19,
        render_space_mode="continuous_light",
        render_nuisance=nuisance,
        validate_renders=False,
        validation_checks=RobotValidationChecksConfig(
            bbox_in_frame=True,
            hands_head_clearance="on",
        ),
    )
    row = df.iloc[0]
    assert bool(row["validation_passed"])
    assert int(row["validation_attempts"]) == 1
    assert row["validation_fail_reason"] == ""
