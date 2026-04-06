"""Robot feature taxonomy, shared geometry, rendering, and mask utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from PIL import ImageFilter as PILImageFilter

from concept_benchmark.config import (
    RobotRenderNuisanceConfig,
    RobotValidationChecksConfig,
)

from .utils import generate_color_schemes

ROBOT_TYPES = ("glorp", "drent")

ALL_ROBOT_FEATURES = {
    "foot_shape": (
        "flat_trapezoid",
        "flat_rounded",
        "flat_square",
        "flat_5sided",
        "flat_lshaped",
        "pointy_trapezoid",
        "pointy_rounded",
        "pointy_square",
        "pointy_3sided",
        "pointy_4sided",
    ),
    "body_shape": ("square", "round"),
    "head_shape": ("square", "round"),
    "has_elbows": ("false", "true"),
    "has_knees": ("false", "true"),
    "has_antennae": ["false", "true"],
    "ears_shape": ("square", "triangle"),
    "mouth_type": ("closed", "open"),
    "hand_shape": (
        "round_circle",
        "round_oval",
        "round_oval2",
        "edgy_triangle",
        "edgy_square",
        "edgy_trapezoid",
    ),
}

COLOR_SCHEMES = generate_color_schemes(
    shuffle=True, random_seed=123456, include_flipped=False
)

DEFAULT_ROBOT_FEATURES = {
    "head_shape": "square",
    "body_shape": "square",
    "foot_shape": "flat",
    "foot_subtype_choice": "default",
    "has_knees": "true",
    "has_elbows": "true",
    "has_antennae": "false",
    "ears_shape": "square",
    "mouth_type": "closed",
    "hand_shape": "round_circle",
    "color_scheme": COLOR_SCHEMES[0],
}

FOOT_SUBTYPES = {
    "flat": ["flat_4sided", "flat_5sided", "flat_lshaped"],
    "pointy": ["pointy_3sided", "pointy_4sided", "pointy_6sided"],
}

_DEFAULT_SUPERSAMPLE = 4
_BLACK = (0, 0, 0, 255)
_WHITE = (255, 255, 255, 255)


@dataclass(frozen=True)
class RobotRenderState:
    """Continuous nuisance-only rendering controls."""

    mode: str = "legacy"
    translate_x_frac: float = 0.0
    translate_y_frac: float = 0.0
    global_scale: float = 1.0
    global_rotation_deg: float = 0.0
    body_aspect_x: float = 1.0
    body_aspect_y: float = 1.0
    head_aspect_x: float = 1.0
    head_aspect_y: float = 1.0
    arm_angle_offset_deg: float = 0.0
    arm_length_scale: float = 1.0
    arm_y_offset_frac: float = 0.0
    left_arm_angle_delta_deg: float = 0.0
    right_arm_angle_delta_deg: float = 0.0
    left_arm_length_scale: float = 1.0
    right_arm_length_scale: float = 1.0
    left_arm_y_offset_frac: float = 0.0
    right_arm_y_offset_frac: float = 0.0
    leg_spread_deg: float = 0.0
    leg_length_scale: float = 1.0
    head_offset_x_frac: float = 0.0
    head_offset_y_frac: float = 0.0
    head_tilt_deg: float = 0.0
    foot_rotation_deg: float = 0.0
    stroke_width_scale: float = 1.0
    point_jitter_frac: float = 0.0
    color_jitter: float = 0.0
    foot_orientation: str = "side"
    jitter_seed: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "render_space_mode": self.mode,
            "accepted_render_space_mode": self.mode,
            "translation_x_frac": self.translate_x_frac,
            "translation_y_frac": self.translate_y_frac,
            "global_scale": self.global_scale,
            "global_rotation_deg": self.global_rotation_deg,
            "body_aspect_x": self.body_aspect_x,
            "body_aspect_y": self.body_aspect_y,
            "head_aspect_x": self.head_aspect_x,
            "head_aspect_y": self.head_aspect_y,
            "arm_angle_offset_deg": self.arm_angle_offset_deg,
            "arm_length_scale": self.arm_length_scale,
            "arm_y_offset_frac": self.arm_y_offset_frac,
            "left_arm_angle_delta_deg": self.left_arm_angle_delta_deg,
            "right_arm_angle_delta_deg": self.right_arm_angle_delta_deg,
            "left_arm_length_scale": self.left_arm_length_scale,
            "right_arm_length_scale": self.right_arm_length_scale,
            "left_arm_y_offset_frac": self.left_arm_y_offset_frac,
            "right_arm_y_offset_frac": self.right_arm_y_offset_frac,
            "leg_spread_deg": self.leg_spread_deg,
            "leg_length_scale": self.leg_length_scale,
            "head_offset_x_frac": self.head_offset_x_frac,
            "head_offset_y_frac": self.head_offset_y_frac,
            "head_tilt_deg": self.head_tilt_deg,
            "foot_rotation_deg": self.foot_rotation_deg,
            "stroke_width_scale": self.stroke_width_scale,
            "point_jitter_frac": self.point_jitter_frac,
            "color_jitter": self.color_jitter,
            "foot_orientation": self.foot_orientation,
            "jitter_seed": self.jitter_seed,
        }


def render_state_from_metadata(metadata: dict[str, Any]) -> RobotRenderState:
    """Reconstruct a render state from catalog metadata."""

    return RobotRenderState(
        mode=str(
            metadata.get(
                "accepted_render_space_mode",
                metadata.get("render_space_mode", "legacy"),
            )
        ),
        translate_x_frac=float(
            metadata.get("translate_x_frac", metadata.get("translation_x_frac", 0.0))
        ),
        translate_y_frac=float(
            metadata.get("translate_y_frac", metadata.get("translation_y_frac", 0.0))
        ),
        global_scale=float(metadata.get("global_scale", 1.0)),
        global_rotation_deg=float(metadata.get("global_rotation_deg", 0.0)),
        body_aspect_x=float(metadata.get("body_aspect_x", 1.0)),
        body_aspect_y=float(metadata.get("body_aspect_y", 1.0)),
        head_aspect_x=float(metadata.get("head_aspect_x", 1.0)),
        head_aspect_y=float(metadata.get("head_aspect_y", 1.0)),
        arm_angle_offset_deg=float(metadata.get("arm_angle_offset_deg", 0.0)),
        arm_length_scale=float(metadata.get("arm_length_scale", 1.0)),
        arm_y_offset_frac=float(metadata.get("arm_y_offset_frac", 0.0)),
        left_arm_angle_delta_deg=float(metadata.get("left_arm_angle_delta_deg", 0.0)),
        right_arm_angle_delta_deg=float(metadata.get("right_arm_angle_delta_deg", 0.0)),
        left_arm_length_scale=float(metadata.get("left_arm_length_scale", 1.0)),
        right_arm_length_scale=float(metadata.get("right_arm_length_scale", 1.0)),
        left_arm_y_offset_frac=float(metadata.get("left_arm_y_offset_frac", 0.0)),
        right_arm_y_offset_frac=float(metadata.get("right_arm_y_offset_frac", 0.0)),
        leg_spread_deg=float(metadata.get("leg_spread_deg", 0.0)),
        leg_length_scale=float(metadata.get("leg_length_scale", 1.0)),
        head_offset_x_frac=float(metadata.get("head_offset_x_frac", 0.0)),
        head_offset_y_frac=float(metadata.get("head_offset_y_frac", 0.0)),
        head_tilt_deg=float(metadata.get("head_tilt_deg", 0.0)),
        foot_rotation_deg=float(metadata.get("foot_rotation_deg", 0.0)),
        stroke_width_scale=float(metadata.get("stroke_width_scale", 1.0)),
        point_jitter_frac=float(metadata.get("point_jitter_frac", 0.0)),
        color_jitter=float(metadata.get("color_jitter", 0.0)),
        foot_orientation=str(metadata.get("foot_orientation", "side")),
        jitter_seed=int(metadata.get("jitter_seed", 0)),
    )


@dataclass(frozen=True)
class PolygonPrimitive:
    points: tuple[tuple[float, float], ...]
    fill_color: tuple[int, int, int, int] | None
    outline_color: tuple[int, int, int, int] | None
    outline_width: float
    part: str
    z: int


@dataclass(frozen=True)
class LinePrimitive:
    points: tuple[tuple[float, float], ...]
    line_color: tuple[int, int, int, int]
    line_width: float
    part: str
    z: int
    closed: bool = False


@dataclass(frozen=True)
class RobotGeometry:
    width: int
    height: int
    features: dict[str, Any]
    state: RobotRenderState
    fill_primitives: tuple[PolygonPrimitive, ...]
    stroke_primitives: tuple[LinePrimitive, ...]
    part_bounds: dict[str, tuple[float, float, float, float]]
    overall_bbox: tuple[float, float, float, float]
    color_left: tuple[int, int, int, int]
    color_right: tuple[int, int, int, int]


class RobotImage:
    """Small wrapper that preserves the historic ``export()`` interface."""

    def __init__(self, image: PILImage.Image):
        self._image = image.convert("RGBA")
        self.width = self._image.width
        self.height = self._image.height

    def export(self, path: str) -> None:
        self._image.save(path)

    def to_pil(self) -> PILImage.Image:
        return self._image.copy()


def _as_bool_string(value: Any) -> str:
    return "true" if str(value).lower() in {"1", "true", "t", "yes", "y"} else "false"


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def _stable_signed_float(text: str, amplitude: float) -> float:
    if amplitude <= 0:
        return 0.0
    v = _stable_int(text) / float(16**16 - 1)
    return (2.0 * v - 1.0) * amplitude


def _sample_range(rng: np.random.Generator, bounds: list[float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


def _normalize_color(value: Any) -> tuple[int, int, int, int]:
    if hasattr(value, "rgb"):
        rgb = value.rgb() if callable(value.rgb) else value.rgb
        if all(0.0 <= c <= 1.0 for c in rgb):
            return tuple(int(round(255 * c)) for c in rgb) + (255,)
        return tuple(int(round(c)) for c in rgb) + (255,)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("#") and len(s) in {4, 7}:
            return tuple(PILImage.new("RGBA", (1, 1), s).getpixel((0, 0)))
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        rgb = tuple(int(round(float(c))) for c in value[:3])
        if all(0 <= c <= 1 for c in rgb):
            rgb = tuple(int(round(255 * c)) for c in rgb)
        return rgb + (255,)
    raise TypeError(f"unsupported color value {value!r}")


def _color_to_hex(color: tuple[int, int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color[:3])


def _apply_color_jitter(
    color: tuple[int, int, int, int], amount: float
) -> tuple[int, int, int, int]:
    if amount == 0:
        return color
    scale = 1.0 + amount
    rgb = []
    for channel in color[:3]:
        if amount >= 0:
            value = channel + (255 - channel) * amount
        else:
            value = channel * scale
        rgb.append(int(min(255, max(0, round(value)))))
    return tuple(rgb) + (color[3],)


def _rotate_point(
    point: tuple[float, float],
    angle_deg: float,
    center: tuple[float, float],
) -> tuple[float, float]:
    if angle_deg == 0.0:
        return point
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return (
        center[0] + cos_t * dx - sin_t * dy,
        center[1] + sin_t * dx + cos_t * dy,
    )


def _ellipse_polygon(
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    *,
    angle_deg: float = 0.0,
    n_points: int = 48,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for idx in range(n_points):
        theta = 2.0 * math.pi * idx / n_points
        pt = (
            center[0] + radius_x * math.cos(theta),
            center[1] + radius_y * math.sin(theta),
        )
        points.append(_rotate_point(pt, angle_deg, center))
    return points


def _ellipse_arc(
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    *,
    start_deg: float,
    end_deg: float,
    n_points: int = 24,
) -> list[tuple[float, float]]:
    if end_deg < start_deg:
        end_deg += 360.0
    points: list[tuple[float, float]] = []
    for idx in range(n_points + 1):
        theta = math.radians(start_deg + (end_deg - start_deg) * idx / n_points)
        points.append(
            (
                center[0] + radius_x * math.cos(theta),
                center[1] + radius_y * math.sin(theta),
            )
        )
    return points


def _rect_polygon(
    center: tuple[float, float],
    width: float,
    height: float,
    *,
    angle_deg: float = 0.0,
) -> list[tuple[float, float]]:
    half_w = width / 2.0
    half_h = height / 2.0
    points = [
        (center[0] - half_w, center[1] - half_h),
        (center[0] + half_w, center[1] - half_h),
        (center[0] + half_w, center[1] + half_h),
        (center[0] - half_w, center[1] + half_h),
    ]
    return [_rotate_point(pt, angle_deg, center) for pt in points]


def _split_vertical_polygon(
    center: tuple[float, float],
    left_top: tuple[float, float],
    left_bottom: tuple[float, float],
    outline: list[tuple[float, float]],
    side: str,
) -> list[tuple[float, float]]:
    if side == "left":
        selected = [pt for pt in outline if pt[0] <= center[0] + 1e-6]
        ordered = sorted(selected, key=lambda pt: math.atan2(pt[1] - center[1], pt[0] - center[0]))
        return [left_top] + ordered + [left_bottom]
    selected = [pt for pt in outline if pt[0] >= center[0] - 1e-6]
    ordered = sorted(selected, key=lambda pt: math.atan2(pt[1] - center[1], pt[0] - center[0]))
    return [left_top] + ordered + [left_bottom]


def _line_bounds(
    points: tuple[tuple[float, float], ...], width: float
) -> tuple[float, float, float, float]:
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    pad = width / 2.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _polygon_bounds(
    points: tuple[tuple[float, float], ...], outline_width: float
) -> tuple[float, float, float, float]:
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    pad = outline_width / 2.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _merge_bounds(
    current: tuple[float, float, float, float] | None,
    new: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if current is None:
        return new
    return (
        min(current[0], new[0]),
        min(current[1], new[1]),
        max(current[2], new[2]),
        max(current[3], new[3]),
    )


def _jitter_points(
    points: list[tuple[float, float]],
    *,
    seed_prefix: str,
    amplitude: float,
) -> list[tuple[float, float]]:
    if amplitude <= 0:
        return points
    jittered = []
    for idx, (x, y) in enumerate(points):
        dx = _stable_signed_float(f"{seed_prefix}:{idx}:x", amplitude)
        dy = _stable_signed_float(f"{seed_prefix}:{idx}:y", amplitude)
        jittered.append((x + dx, y + dy))
    return jittered


def _resolve_color_scheme(
    color_scheme: Any, col_scheme_add: int = 0
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if isinstance(color_scheme, int):
        idx = (int(color_scheme) + int(col_scheme_add)) % len(COLOR_SCHEMES)
        left, right = COLOR_SCHEMES[idx]
    elif isinstance(color_scheme, (tuple, list)) and len(color_scheme) == 2:
        left, right = color_scheme
    else:
        raise TypeError(f"unsupported color scheme {color_scheme!r}")
    return _normalize_color(left), _normalize_color(right)


def sample_robot_render_state(
    *,
    seed: int = 0,
    render_space_mode: str = "legacy",
    render_nuisance: RobotRenderNuisanceConfig | dict[str, Any] | None = None,
    foot_orientation: str | None = None,
) -> RobotRenderState:
    """Sample a deterministic nuisance-only render state."""

    if isinstance(render_nuisance, dict):
        render_nuisance = RobotRenderNuisanceConfig(**render_nuisance)
    nuisance = render_nuisance or RobotRenderNuisanceConfig.for_mode(render_space_mode)
    nuisance.validate()
    rng = np.random.default_rng(int(seed))
    orientation = foot_orientation or (
        "vertex" if int(rng.integers(0, 2)) else "side"
    )
    if render_space_mode == "legacy":
        return RobotRenderState(mode="legacy", foot_orientation=orientation)
    return RobotRenderState(
        mode=render_space_mode,
        translate_x_frac=_sample_range(rng, nuisance.translate_x_frac),
        translate_y_frac=_sample_range(rng, nuisance.translate_y_frac),
        global_scale=_sample_range(rng, nuisance.global_scale),
        global_rotation_deg=_sample_range(rng, nuisance.global_rotation_deg),
        body_aspect_x=_sample_range(rng, nuisance.body_aspect_x),
        body_aspect_y=_sample_range(rng, nuisance.body_aspect_y),
        head_aspect_x=_sample_range(rng, nuisance.head_aspect_x),
        head_aspect_y=_sample_range(rng, nuisance.head_aspect_y),
        arm_angle_offset_deg=_sample_range(rng, nuisance.arm_angle_offset_deg),
        arm_length_scale=_sample_range(rng, nuisance.arm_length_scale),
        arm_y_offset_frac=_sample_range(rng, nuisance.arm_y_offset_frac),
        left_arm_angle_delta_deg=_sample_range(rng, nuisance.left_arm_angle_delta_deg),
        right_arm_angle_delta_deg=_sample_range(
            rng,
            nuisance.right_arm_angle_delta_deg,
        ),
        left_arm_length_scale=_sample_range(rng, nuisance.left_arm_length_scale),
        right_arm_length_scale=_sample_range(rng, nuisance.right_arm_length_scale),
        left_arm_y_offset_frac=_sample_range(rng, nuisance.left_arm_y_offset_frac),
        right_arm_y_offset_frac=_sample_range(rng, nuisance.right_arm_y_offset_frac),
        leg_spread_deg=_sample_range(rng, nuisance.leg_spread_deg),
        leg_length_scale=_sample_range(rng, nuisance.leg_length_scale),
        head_offset_x_frac=_sample_range(rng, nuisance.head_offset_x_frac),
        head_offset_y_frac=_sample_range(rng, nuisance.head_offset_y_frac),
        head_tilt_deg=_sample_range(rng, nuisance.head_tilt_deg),
        foot_rotation_deg=_sample_range(rng, nuisance.foot_rotation_deg),
        stroke_width_scale=_sample_range(rng, nuisance.stroke_width_scale),
        point_jitter_frac=_sample_range(rng, nuisance.point_jitter_frac),
        color_jitter=_sample_range(rng, nuisance.color_jitter),
        foot_orientation=orientation,
        jitter_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _foot_polygon(
    subtype: str,
    center: tuple[float, float],
    foot_width: float,
    foot_height: float,
    *,
    side: str,
) -> list[tuple[float, float]]:
    cx, cy = center
    top = cy - foot_height / 2.0
    if subtype == "flat_trapezoid":
        points = [
            (cx - 0.45 * foot_width, top + 0.10 * foot_height),
            (cx - 0.30 * foot_width, top),
            (cx, top),
            (cx + 0.30 * foot_width, top),
            (cx + 0.45 * foot_width, top + 0.10 * foot_height),
            (cx + 0.15 * foot_width, top + foot_height),
            (cx + 0.03 * foot_width, top + foot_height),
            (cx - 0.03 * foot_width, top + foot_height),
            (cx - 0.15 * foot_width, top + foot_height),
        ]
    elif subtype == "pointy_trapezoid":
        points = [
            (cx - 0.45 * foot_width, top + 0.10 * foot_height),
            (cx - 0.30 * foot_width, top),
            (cx, top),
            (cx + 0.30 * foot_width, top),
            (cx + 0.45 * foot_width, top + 0.10 * foot_height),
            (cx + 0.18 * foot_width, top + 0.88 * foot_height),
            (cx, top + foot_height),
            (cx - 0.18 * foot_width, top + 0.88 * foot_height),
        ]
    elif subtype == "flat_square":
        points = [
            (cx - 0.50 * foot_width, top + 0.15 * foot_height),
            (cx - 0.40 * foot_width, top + 0.05 * foot_height),
            (cx - 0.20 * foot_width, top),
            (cx + 0.20 * foot_width, top),
            (cx + 0.40 * foot_width, top + 0.05 * foot_height),
            (cx + 0.50 * foot_width, top + 0.15 * foot_height),
            (cx + 0.50 * foot_width, top + 0.75 * foot_height),
            (cx + 0.20 * foot_width, top + 0.92 * foot_height),
            (cx + 0.15 * foot_width, top + foot_height),
            (cx + 0.03 * foot_width, top + foot_height),
            (cx - 0.03 * foot_width, top + foot_height),
            (cx - 0.15 * foot_width, top + foot_height),
            (cx - 0.20 * foot_width, top + 0.92 * foot_height),
            (cx - 0.50 * foot_width, top + 0.75 * foot_height),
        ]
    elif subtype == "pointy_square":
        points = [
            (cx - 0.50 * foot_width, top + 0.15 * foot_height),
            (cx - 0.40 * foot_width, top + 0.05 * foot_height),
            (cx - 0.20 * foot_width, top),
            (cx + 0.20 * foot_width, top),
            (cx + 0.40 * foot_width, top + 0.05 * foot_height),
            (cx + 0.50 * foot_width, top + 0.15 * foot_height),
            (cx + 0.50 * foot_width, top + 0.75 * foot_height),
            (cx + 0.20 * foot_width, top + 0.92 * foot_height),
            (cx, top + foot_height),
            (cx - 0.20 * foot_width, top + 0.92 * foot_height),
            (cx - 0.50 * foot_width, top + 0.75 * foot_height),
        ]
    elif subtype == "flat_rounded":
        points = [
            (cx - 0.50 * foot_width, top + 0.50 * foot_height),
            (cx - 0.35 * foot_width, top + 0.15 * foot_height),
            (cx, top),
            (cx + 0.35 * foot_width, top + 0.15 * foot_height),
            (cx + 0.50 * foot_width, top + 0.50 * foot_height),
            (cx + 0.18 * foot_width, top + 0.92 * foot_height),
            (cx + 0.15 * foot_width, top + foot_height),
            (cx + 0.03 * foot_width, top + foot_height),
            (cx - 0.03 * foot_width, top + foot_height),
            (cx - 0.15 * foot_width, top + foot_height),
            (cx - 0.18 * foot_width, top + 0.92 * foot_height),
        ]
    elif subtype == "pointy_rounded":
        points = [
            (cx - 0.50 * foot_width, top + 0.30 * foot_height),
            (cx - 0.40 * foot_width, top + 0.10 * foot_height),
            (cx - 0.20 * foot_width, top),
            (cx, top),
            (cx + 0.20 * foot_width, top),
            (cx + 0.40 * foot_width, top + 0.10 * foot_height),
            (cx + 0.50 * foot_width, top + 0.30 * foot_height),
            (cx + 0.18 * foot_width, top + 0.88 * foot_height),
            (cx, top + foot_height),
            (cx - 0.18 * foot_width, top + 0.88 * foot_height),
        ]
    elif subtype == "flat_5sided":
        points = [
            (cx, top),
            (cx - 0.50 * foot_width, top + 0.30 * foot_height),
            (cx - 0.50 * foot_width, top + foot_height),
            (cx + 0.50 * foot_width, top + foot_height),
            (cx + 0.50 * foot_width, top + 0.30 * foot_height),
        ]
    elif subtype == "flat_lshaped":
        if side == "left":
            points = [
                (cx - 0.35 * foot_width, top),
                (cx - 0.35 * foot_width, top + 0.35 * foot_height),
                (cx - 1.00 * foot_width, top + 0.35 * foot_height),
                (cx - 1.00 * foot_width, top + foot_height),
                (cx + 0.35 * foot_width, top + foot_height),
                (cx + 0.35 * foot_width, top),
            ]
        else:
            points = [
                (cx + 0.35 * foot_width, top),
                (cx + 0.35 * foot_width, top + 0.35 * foot_height),
                (cx + 1.00 * foot_width, top + 0.35 * foot_height),
                (cx + 1.00 * foot_width, top + foot_height),
                (cx - 0.35 * foot_width, top + foot_height),
                (cx - 0.35 * foot_width, top),
            ]
    elif subtype == "pointy_3sided":
        points = [
            (cx - 0.50 * foot_width, top),
            (cx + 0.50 * foot_width, top),
            (cx, top + foot_height),
        ]
    elif subtype == "pointy_4sided":
        points = [
            (cx, top),
            (cx - 0.50 * foot_width, top + 0.50 * foot_height),
            (cx, top + foot_height),
            (cx + 0.50 * foot_width, top + 0.50 * foot_height),
        ]
    else:
        raise TypeError(f"invalid foot type {subtype!r}")
    return points


def _hand_polygon(
    hand_shape: str,
    center: tuple[float, float],
    hand_size: float,
    *,
    side: str,
    angle_deg: float,
) -> list[tuple[float, float]]:
    cx, cy = center
    kind, subtype = hand_shape.split("_", 1)
    if kind == "round":
        if subtype == "circle":
            return _ellipse_polygon(center, hand_size / 2.0, hand_size / 2.0)
        if subtype == "oval":
            return _ellipse_polygon(center, 0.75 * hand_size, 0.50 * hand_size, angle_deg=angle_deg)
        if subtype == "oval2":
            return _ellipse_polygon(center, 0.50 * hand_size, 0.75 * hand_size, angle_deg=angle_deg)
    if kind == "edgy":
        if subtype == "triangle":
            outward = angle_deg if side == "right" else angle_deg + 180.0
            points = [
                (cx, cy - hand_size / 2.0),
                (cx, cy + hand_size / 2.0),
                (cx + hand_size, cy),
            ]
            return [_rotate_point(pt, outward, center) for pt in points]
        if subtype == "square":
            return _rect_polygon(center, hand_size, hand_size, angle_deg=angle_deg)
        if subtype == "trapezoid":
            outward = angle_deg if side == "right" else angle_deg + 180.0
            points = [
                (cx, cy - hand_size / 4.0),
                (cx, cy + hand_size / 4.0),
                (cx + hand_size, cy + hand_size / 2.0),
                (cx + hand_size, cy - hand_size / 2.0),
            ]
            return [_rotate_point(pt, outward, center) for pt in points]
    raise TypeError(f"invalid hand shape {hand_shape!r}")


def _resolve_foot_shape(features: dict[str, Any]) -> str:
    subtype = str(features["foot_shape"])
    if subtype in FOOT_SUBTYPES and features.get("foot_subtype_choice", "default") == "default":
        return str(FOOT_SUBTYPES[subtype][0])
    return subtype


def compute_robot_geometry(
    render_state: RobotRenderState,
    width: int = 600,
    height: int = 600,
    **kwargs,
) -> RobotGeometry:
    """Compute all robot geometry used by rendering, masks, and validation."""

    features = dict(DEFAULT_ROBOT_FEATURES)
    features.update(kwargs)
    features["has_knees"] = _as_bool_string(features.get("has_knees", "true"))
    features["has_elbows"] = _as_bool_string(features.get("has_elbows", "true"))
    features["has_antennae"] = _as_bool_string(features.get("has_antennae", "false"))

    color_left, color_right = _resolve_color_scheme(features["color_scheme"])
    color_left = _apply_color_jitter(color_left, render_state.color_jitter)
    color_right = _apply_color_jitter(color_right, -render_state.color_jitter)

    width = int(width)
    height = int(height)
    r = height / 12.0
    jitter_amp = render_state.point_jitter_frac * r
    stroke_width = max(1.0, render_state.stroke_width_scale)

    x_mid = 0.5 * width + render_state.translate_x_frac * width
    y_top_face = 2.0 * r + render_state.translate_y_frac * height
    scale = render_state.global_scale

    head_width = 1.75 * r * render_state.head_aspect_x * scale
    head_height = 1.75 * r * render_state.head_aspect_y * scale
    body_width = 3.5 * r * render_state.body_aspect_x * scale
    body_height = 4.0 * r * render_state.body_aspect_y * scale

    head_center = (
        x_mid + render_state.head_offset_x_frac * width,
        y_top_face + 0.5 * head_height + render_state.head_offset_y_frac * height,
    )
    body_center = (
        x_mid,
        y_top_face + head_height + 0.5 * body_height,
    )
    body_top = body_center[1] - 0.5 * body_height
    body_bottom = body_center[1] + 0.5 * body_height

    foot_gap = 1.0 * r * scale
    foot_width = max(0.6 * r, (body_width - 2.0 * foot_gap) / 2.0)
    foot_height = foot_width
    leg_length = 0.75 * (body_height - foot_height) * render_state.leg_length_scale
    base_arm_length = 0.75 * body_width * render_state.arm_length_scale
    left_arm_length = base_arm_length * render_state.left_arm_length_scale
    right_arm_length = base_arm_length * render_state.right_arm_length_scale

    shoulder_y = body_top + 0.33 * body_height + render_state.arm_y_offset_frac * height
    shoulder_left = (
        body_center[0] - 0.5 * body_width,
        shoulder_y + render_state.left_arm_y_offset_frac * height,
    )
    shoulder_right = (
        body_center[0] + 0.5 * body_width,
        shoulder_y + render_state.right_arm_y_offset_frac * height,
    )

    left_arm_angle = (
        180.0
        - render_state.arm_angle_offset_deg
        + render_state.left_arm_angle_delta_deg
    )
    right_arm_angle = (
        0.0
        + render_state.arm_angle_offset_deg
        + render_state.right_arm_angle_delta_deg
    )
    hand_left_center = (
        shoulder_left[0] + left_arm_length * math.cos(math.radians(left_arm_angle)),
        shoulder_left[1] + left_arm_length * math.sin(math.radians(left_arm_angle)),
    )
    hand_right_center = (
        shoulder_right[0] + right_arm_length * math.cos(math.radians(right_arm_angle)),
        shoulder_right[1] + right_arm_length * math.sin(math.radians(right_arm_angle)),
    )

    hip_left = (
        body_center[0] - (0.5 * foot_gap + 0.5 * foot_width),
        body_top + 0.9 * body_height,
    )
    hip_right = (
        body_center[0] + (0.5 * foot_gap + 0.5 * foot_width),
        body_top + 0.9 * body_height,
    )
    left_leg_angle = 90.0 + render_state.leg_spread_deg
    right_leg_angle = 90.0 - render_state.leg_spread_deg
    ankle_left = (
        hip_left[0] + leg_length * math.cos(math.radians(left_leg_angle)),
        hip_left[1] + leg_length * math.sin(math.radians(left_leg_angle)),
    )
    ankle_right = (
        hip_right[0] + leg_length * math.cos(math.radians(right_leg_angle)),
        hip_right[1] + leg_length * math.sin(math.radians(right_leg_angle)),
    )
    foot_left_center = (ankle_left[0], ankle_left[1] + 0.5 * foot_height)
    foot_right_center = (ankle_right[0], ankle_right[1] + 0.5 * foot_height)

    global_rotation_center = body_center
    foot_mode_rotation = 0.0
    if render_state.mode != "legacy" and render_state.foot_orientation == "vertex":
        foot_mode_rotation = 8.0 if "light" in render_state.mode else 12.0
    foot_left_rotation = render_state.foot_rotation_deg - foot_mode_rotation
    foot_right_rotation = -render_state.foot_rotation_deg + foot_mode_rotation

    fill_primitives: list[PolygonPrimitive] = []
    stroke_primitives: list[LinePrimitive] = []

    def add_polygon(
        points: list[tuple[float, float]],
        *,
        fill_color: tuple[int, int, int, int] | None,
        outline_color: tuple[int, int, int, int] | None = None,
        outline_width: float = stroke_width,
        part: str,
        z: int,
        jitter_key: str | None = None,
    ) -> None:
        if jitter_key:
            points2 = _jitter_points(points, seed_prefix=f"{render_state.jitter_seed}:{jitter_key}", amplitude=jitter_amp)
        else:
            points2 = points
        points3 = [
            _rotate_point(
                _rotate_point(pt, render_state.head_tilt_deg, head_center)
                if part in {"head", "eyes", "ears", "mouth", "antennae"}
                else pt,
                render_state.global_rotation_deg,
                global_rotation_center,
            )
            for pt in points2
        ]
        fill_primitives.append(
            PolygonPrimitive(
                points=tuple(points3),
                fill_color=fill_color,
                outline_color=outline_color,
                outline_width=outline_width,
                part=part,
                z=z,
            )
        )

    def add_line(
        points: list[tuple[float, float]],
        *,
        line_color: tuple[int, int, int, int] = _BLACK,
        line_width: float = stroke_width,
        part: str,
        z: int,
        jitter_key: str | None = None,
        rotate_with_head: bool = False,
        closed: bool = False,
    ) -> None:
        if jitter_key:
            points2 = _jitter_points(points, seed_prefix=f"{render_state.jitter_seed}:{jitter_key}", amplitude=jitter_amp)
        else:
            points2 = points
        if rotate_with_head:
            points2 = [_rotate_point(pt, render_state.head_tilt_deg, head_center) for pt in points2]
        points3 = [_rotate_point(pt, render_state.global_rotation_deg, global_rotation_center) for pt in points2]
        stroke_primitives.append(
            LinePrimitive(
                points=tuple(points3),
                line_color=line_color,
                line_width=line_width,
                part=part,
                z=z,
                closed=closed,
            )
        )

    if features["head_shape"] == "round":
        head_outline = _ellipse_polygon(head_center, head_width / 2.0, head_height / 2.0)
        head_left_fill = (
            [(head_center[0], head_center[1] - head_height / 2.0)]
            + _ellipse_arc(
                head_center,
                head_width / 2.0,
                head_height / 2.0,
                start_deg=90.0,
                end_deg=270.0,
            )
            + [(head_center[0], head_center[1] + head_height / 2.0)]
        )
        head_right_fill = (
            [(head_center[0], head_center[1] - head_height / 2.0)]
            + _ellipse_arc(
                head_center,
                head_width / 2.0,
                head_height / 2.0,
                start_deg=-90.0,
                end_deg=90.0,
            )
            + [(head_center[0], head_center[1] + head_height / 2.0)]
        )
    else:
        head_outline = _rect_polygon(head_center, head_width, head_height)
        head_left_fill = _rect_polygon(
            (head_center[0] - head_width / 4.0, head_center[1]),
            head_width / 2.0,
            head_height,
        )
        head_right_fill = _rect_polygon(
            (head_center[0] + head_width / 4.0, head_center[1]),
            head_width / 2.0,
            head_height,
        )
    add_polygon(
        head_left_fill,
        fill_color=color_left,
        outline_color=None,
        part="head",
        z=8,
        jitter_key="head_left_fill",
    )
    add_polygon(
        head_right_fill,
        fill_color=color_right,
        outline_color=None,
        part="head",
        z=8,
        jitter_key="head_right_fill",
    )
    add_line(head_outline, part="head", z=9, rotate_with_head=True, jitter_key="head_outline", closed=True)
    add_line(
        [
            (head_center[0], head_center[1] - head_height / 2.0),
            (head_center[0], head_center[1] + head_height / 2.0),
        ],
        part="head",
        z=9,
        rotate_with_head=True,
    )

    if features["body_shape"] == "round":
        body_radius = 0.57 * body_width
        body_outline = _ellipse_polygon(body_center, body_radius, body_radius)
        body_left_fill = (
            [(body_center[0], body_center[1] - body_radius)]
            + _ellipse_arc(
                body_center,
                body_radius,
                body_radius,
                start_deg=90.0,
                end_deg=270.0,
            )
            + [(body_center[0], body_center[1] + body_radius)]
        )
        body_right_fill = (
            [(body_center[0], body_center[1] - body_radius)]
            + _ellipse_arc(
                body_center,
                body_radius,
                body_radius,
                start_deg=-90.0,
                end_deg=90.0,
            )
            + [(body_center[0], body_center[1] + body_radius)]
        )
    else:
        body_outline = _rect_polygon(body_center, body_width, body_height)
        body_left_fill = _rect_polygon(
            (body_center[0] - body_width / 4.0, body_center[1]),
            body_width / 2.0,
            body_height,
        )
        body_right_fill = _rect_polygon(
            (body_center[0] + body_width / 4.0, body_center[1]),
            body_width / 2.0,
            body_height,
        )
    add_polygon(
        body_left_fill,
        fill_color=color_left,
        outline_color=None,
        part="body",
        z=5,
        jitter_key="body_left_fill",
    )
    add_polygon(
        body_right_fill,
        fill_color=color_right,
        outline_color=None,
        part="body",
        z=5,
        jitter_key="body_right_fill",
    )
    add_line(body_outline, part="body", z=6, jitter_key="body_outline", closed=True)
    add_line(
        [(body_center[0], body_top), (body_center[0], body_bottom)],
        part="body",
        z=6,
    )

    add_line([shoulder_left, hand_left_center], part="arms", z=1, jitter_key="arm_left")
    add_line([shoulder_right, hand_right_center], part="arms", z=1, jitter_key="arm_right")

    hand_size = 0.6 * r * scale
    add_polygon(
        _hand_polygon(
            str(features.get("hand_shape", "round_circle")),
            hand_left_center,
            hand_size,
            side="left",
            angle_deg=left_arm_angle,
        ),
        fill_color=color_left,
        outline_color=_BLACK,
        part="hands",
        z=2,
        jitter_key="hand_left",
    )
    add_polygon(
        _hand_polygon(
            str(features.get("hand_shape", "round_circle")),
            hand_right_center,
            hand_size,
            side="right",
            angle_deg=right_arm_angle,
        ),
        fill_color=color_right,
        outline_color=_BLACK,
        part="hands",
        z=2,
        jitter_key="hand_right",
    )

    add_line([hip_left, ankle_left], part="legs", z=3, jitter_key="leg_left")
    add_line([hip_right, ankle_right], part="legs", z=3, jitter_key="leg_right")

    if features["has_elbows"] == "true":
        elbow_radius = 0.10 * r
        elbow_left = (
            shoulder_left[0] + 0.5 * (hand_left_center[0] - shoulder_left[0]),
            shoulder_left[1] + 0.5 * (hand_left_center[1] - shoulder_left[1]),
        )
        elbow_right = (
            shoulder_right[0] + 0.5 * (hand_right_center[0] - shoulder_right[0]),
            shoulder_right[1] + 0.5 * (hand_right_center[1] - shoulder_right[1]),
        )
        add_polygon(
            _ellipse_polygon(elbow_left, elbow_radius, elbow_radius),
            fill_color=color_left,
            outline_color=_BLACK,
            part="elbows",
            z=2,
        )
        add_polygon(
            _ellipse_polygon(elbow_right, elbow_radius, elbow_radius),
            fill_color=color_right,
            outline_color=_BLACK,
            part="elbows",
            z=2,
        )

    if features["has_knees"] == "true":
        knee_radius = 0.10 * r
        knee_left = (
            hip_left[0] + 0.5 * (ankle_left[0] - hip_left[0]),
            hip_left[1] + 0.5 * (ankle_left[1] - hip_left[1]),
        )
        knee_right = (
            hip_right[0] + 0.5 * (ankle_right[0] - hip_right[0]),
            hip_right[1] + 0.5 * (ankle_right[1] - hip_right[1]),
        )
        add_polygon(
            _ellipse_polygon(knee_left, knee_radius, knee_radius),
            fill_color=color_left,
            outline_color=_BLACK,
            part="knees",
            z=4,
        )
        add_polygon(
            _ellipse_polygon(knee_right, knee_radius, knee_radius),
            fill_color=color_right,
            outline_color=_BLACK,
            part="knees",
            z=4,
        )

    foot_subtype = _resolve_foot_shape(features)
    left_foot = _foot_polygon(
        foot_subtype,
        foot_left_center,
        foot_width,
        foot_height,
        side="left",
    )
    right_foot = _foot_polygon(
        foot_subtype,
        foot_right_center,
        foot_width,
        foot_height,
        side="right",
    )
    left_foot = [_rotate_point(pt, foot_left_rotation, foot_left_center) for pt in left_foot]
    right_foot = [_rotate_point(pt, foot_right_rotation, foot_right_center) for pt in right_foot]
    add_polygon(left_foot, fill_color=color_left, outline_color=_BLACK, part="feet", z=7, jitter_key="foot_left")
    add_polygon(right_foot, fill_color=color_right, outline_color=_BLACK, part="feet", z=7, jitter_key="foot_right")

    ear_size = 0.4 * r * scale
    ear_y = head_center[1]
    ear_x_left = head_center[0] - 0.5 * head_width
    ear_x_right = head_center[0] + 0.5 * head_width
    if features["ears_shape"] == "square":
        add_polygon(
            _rect_polygon((ear_x_left - ear_size / 2.0, ear_y), ear_size, ear_size),
            fill_color=color_left,
            outline_color=_BLACK,
            part="ears",
            z=10,
        )
        add_polygon(
            _rect_polygon((ear_x_right + ear_size / 2.0, ear_y), ear_size, ear_size),
            fill_color=color_right,
            outline_color=_BLACK,
            part="ears",
            z=10,
        )
    else:
        add_polygon(
            [
                (ear_x_left, ear_y - ear_size / 2.0),
                (ear_x_left, ear_y + ear_size / 2.0),
                (ear_x_left - ear_size, ear_y),
            ],
            fill_color=color_left,
            outline_color=_BLACK,
            part="ears",
            z=10,
            jitter_key="ear_left",
        )
        add_polygon(
            [
                (ear_x_right, ear_y - ear_size / 2.0),
                (ear_x_right, ear_y + ear_size / 2.0),
                (ear_x_right + ear_size, ear_y),
            ],
            fill_color=color_right,
            outline_color=_BLACK,
            part="ears",
            z=10,
            jitter_key="ear_right",
        )

    eye_radius = 0.10 * r * scale
    add_polygon(
        _ellipse_polygon(
            (head_center[0] - 0.30 * r * scale, head_center[1] - 0.18 * head_height),
            eye_radius,
            eye_radius,
        ),
        fill_color=_BLACK,
        outline_color=None,
        part="eyes",
        z=11,
    )
    add_polygon(
        _ellipse_polygon(
            (head_center[0] + 0.30 * r * scale, head_center[1] - 0.18 * head_height),
            eye_radius,
            eye_radius,
        ),
        fill_color=_BLACK,
        outline_color=None,
        part="eyes",
        z=11,
    )

    mouth_width = 0.40 * head_width
    mouth_center = (
        head_center[0],
        head_center[1] + 0.18 * head_height,
    )
    if features["mouth_type"] == "closed":
        add_polygon(
            _rect_polygon(mouth_center, mouth_width, 0.05 * r * scale),
            fill_color=_BLACK,
            outline_color=None,
            part="mouth",
            z=11,
        )
    else:
        mouth_height = 0.20 * r * scale
        mouth_outline = _rect_polygon(mouth_center, mouth_width, mouth_height)
        add_polygon(
            mouth_outline,
            fill_color=_WHITE,
            outline_color=_BLACK,
            part="mouth",
            z=11,
        )
        spacing = mouth_width / 5.0
        for idx in range(1, 5):
            grill_x = mouth_center[0] - mouth_width / 2.0 + idx * spacing
            add_line(
                [
                    (grill_x, mouth_center[1] - mouth_height / 2.0),
                    (grill_x, mouth_center[1] + mouth_height / 2.0),
                ],
                part="mouth",
                z=12,
                rotate_with_head=True,
            )

    if features["has_antennae"] == "true":
        antenna_length = (1.5 * r if features["head_shape"] == "round" else 1.75 * r) * scale
        antenna_origin = head_center
        for idx, angle_deg in enumerate((240.0, -60.0), start=1):
            end = (
                antenna_origin[0] + antenna_length * math.cos(math.radians(angle_deg)),
                antenna_origin[1] + antenna_length * math.sin(math.radians(angle_deg)),
            )
            add_line(
                [antenna_origin, end],
                line_width=max(1.0, 0.08 * r * scale * render_state.stroke_width_scale),
                part="antennae",
                z=8,
                rotate_with_head=True,
                jitter_key=f"antenna_{idx}",
            )

    part_bounds: dict[str, tuple[float, float, float, float]] = {}
    overall_bounds: tuple[float, float, float, float] | None = None
    for primitive in fill_primitives:
        bounds = _polygon_bounds(primitive.points, primitive.outline_width)
        part_bounds[primitive.part] = _merge_bounds(part_bounds.get(primitive.part), bounds)
        overall_bounds = _merge_bounds(overall_bounds, bounds)
    for primitive in stroke_primitives:
        bounds = _line_bounds(primitive.points, primitive.line_width)
        part_bounds[primitive.part] = _merge_bounds(part_bounds.get(primitive.part), bounds)
        overall_bounds = _merge_bounds(overall_bounds, bounds)

    return RobotGeometry(
        width=width,
        height=height,
        features=features,
        state=render_state,
        fill_primitives=tuple(sorted(fill_primitives, key=lambda item: item.z)),
        stroke_primitives=tuple(sorted(stroke_primitives, key=lambda item: item.z)),
        part_bounds=part_bounds,
        overall_bbox=overall_bounds or (0.0, 0.0, 0.0, 0.0),
        color_left=color_left,
        color_right=color_right,
    )


def _draw_polygon(
    draw: PILImageDraw.ImageDraw,
    primitive: PolygonPrimitive,
    *,
    scale: int,
    mask: bool,
) -> None:
    pts = [(scale * x, scale * y) for x, y in primitive.points]
    if mask:
        draw.polygon(pts, fill=255)
        if primitive.outline_color is not None and primitive.outline_width > 0:
            draw.line(pts + [pts[0]], fill=255, width=max(1, round(scale * primitive.outline_width)))
        return
    if primitive.fill_color is not None:
        draw.polygon(pts, fill=primitive.fill_color)
    if primitive.outline_color is not None and primitive.outline_width > 0:
        draw.line(
            pts + [pts[0]],
            fill=primitive.outline_color,
            width=max(1, round(scale * primitive.outline_width)),
        )


def _draw_line(
    draw: PILImageDraw.ImageDraw,
    primitive: LinePrimitive,
    *,
    scale: int,
    mask: bool,
) -> None:
    pts = [(scale * x, scale * y) for x, y in primitive.points]
    width = max(1, round(scale * primitive.line_width))
    fill = 255 if mask else primitive.line_color
    if primitive.closed and pts:
        pts = pts + [pts[0]]
    draw.line(pts, fill=fill, width=width, joint="curve")


def _render_geometry_to_pil(
    geometry: RobotGeometry,
    *,
    parts: tuple[str, ...] | None = None,
    mask: bool = False,
    supersample: int | None = None,
    mask_mode: str = "exact",
) -> PILImage.Image:
    default_scale = 1 if mask else _DEFAULT_SUPERSAMPLE
    scale = max(1, int(supersample or default_scale))
    size = (scale * geometry.width, scale * geometry.height)
    image = PILImage.new("L" if mask else "RGBA", size, 0 if mask else _WHITE)
    draw = PILImageDraw.Draw(image, "RGBA" if not mask else None)

    if parts is not None and mask_mode == "uniform_rect":
        for part in parts:
            bounds = geometry.part_bounds.get(part)
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            draw.rectangle((scale * x0, scale * y0, scale * x1, scale * y1), fill=255 if mask else None)
        return image

    part_filter = set(parts) if parts is not None else None
    for primitive in geometry.fill_primitives:
        if part_filter is not None and primitive.part not in part_filter:
            continue
        _draw_polygon(draw, primitive, scale=scale, mask=mask)
    for primitive in geometry.stroke_primitives:
        if part_filter is not None and primitive.part not in part_filter:
            continue
        _draw_line(draw, primitive, scale=scale, mask=mask)

    if not mask and scale > 1:
        image = image.resize((geometry.width, geometry.height), resample=PILImage.Resampling.LANCZOS)
    return image


def draw_robot(filetype="svg", col_scheme_add=0, width=600, height=600, **kwargs):
    """Draw a robot using shared geometry.

    The historic ``filetype`` argument is accepted for compatibility. The
    renderer always rasterizes to an RGBA image and returns a wrapper with an
    ``export()`` method.
    """

    features = dict(kwargs)
    if isinstance(features.get("color_scheme"), int):
        features["color_scheme"] = (
            int(features["color_scheme"]) + int(col_scheme_add)
        ) % len(COLOR_SCHEMES)
    render_state = features.pop("render_state", None)
    render_space_mode = features.pop("render_space_mode", "legacy")
    render_nuisance = features.pop("render_nuisance", None)
    render_seed = int(features.pop("render_seed", 0))
    foot_orientation = features.get("foot_orientation")
    if render_state is None:
        render_state = sample_robot_render_state(
            seed=render_seed,
            render_space_mode=render_space_mode,
            render_nuisance=render_nuisance,
            foot_orientation=foot_orientation,
        )
    geometry = compute_robot_geometry(render_state, width=width, height=height, **features)
    return RobotImage(_render_geometry_to_pil(geometry))


def image_to_numpy_and_pillow(img):
    """Render a robot image into NumPy and Pillow representations."""

    if isinstance(img, RobotImage):
        pil_image = img.to_pil()
    elif isinstance(img, PILImage.Image):
        pil_image = img.convert("RGBA")
    else:
        raise TypeError(f"unsupported image type {type(img)!r}")
    return np.asarray(pil_image), pil_image


def draw_robot_mask(
    width=600, height=600, parts=("body",), mode: str = "exact", **kwargs
):
    """Draw a binary mask for selected robot parts."""

    if mode not in {"exact", "uniform_rect"}:
        raise ValueError("mask_mode must be 'exact' or 'uniform_rect'")
    features = dict(kwargs)
    render_state = features.pop("render_state", None)
    render_space_mode = features.pop("render_space_mode", "legacy")
    render_nuisance = features.pop("render_nuisance", None)
    render_seed = int(features.pop("render_seed", 0))
    foot_orientation = features.get("foot_orientation")
    if render_state is None:
        render_state = sample_robot_render_state(
            seed=render_seed,
            render_space_mode=render_space_mode,
            render_nuisance=render_nuisance,
            foot_orientation=foot_orientation,
        )
    geometry = compute_robot_geometry(render_state, width=width, height=height, **features)
    return RobotImage(
        _render_geometry_to_pil(
            geometry,
            parts=tuple(parts),
            mask=True,
            supersample=1,
            mask_mode=mode,
        ).convert("RGBA")
    )


def _mask_area(mask: PILImage.Image) -> int:
    arr = np.asarray(mask.convert("L"))
    return int(np.count_nonzero(arr > 0))


def _normalize_validation_checks(
    validation_checks: RobotValidationChecksConfig | dict[str, Any] | None,
) -> RobotValidationChecksConfig:
    if isinstance(validation_checks, dict):
        validation_checks = RobotValidationChecksConfig(**validation_checks)
    validation_checks = validation_checks or RobotValidationChecksConfig()
    validation_checks.validate()
    return validation_checks


def _mask_bool_array(
    geometry: RobotGeometry,
    *,
    parts: tuple[str, ...],
    supersample: int = 1,
) -> np.ndarray:
    mask = _render_geometry_to_pil(
        geometry,
        parts=parts,
        mask=True,
        supersample=supersample,
    )
    return np.asarray(mask.convert("L")) > 0


def _dilate_bool_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    img = PILImage.fromarray(mask.astype(np.uint8) * 255, mode="L")
    size = max(3, 2 * int(radius_px) + 1)
    dilated = img.filter(PILImageFilter.MaxFilter(size=size))
    return np.asarray(dilated) > 0


def _merge_part_bounds(
    geometry: RobotGeometry,
    parts: tuple[str, ...],
) -> tuple[float, float, float, float] | None:
    bounds = None
    for part in parts:
        part_bounds = geometry.part_bounds.get(part)
        if part_bounds is not None:
            bounds = _merge_bounds(bounds, part_bounds)
    return bounds


def _bbox_clearance(
    bounds_a: tuple[float, float, float, float] | None,
    bounds_b: tuple[float, float, float, float] | None,
) -> float:
    if bounds_a is None or bounds_b is None:
        return float("inf")
    dx = max(bounds_a[0] - bounds_b[2], bounds_b[0] - bounds_a[2], 0.0)
    dy = max(bounds_a[1] - bounds_b[3], bounds_b[1] - bounds_a[3], 0.0)
    if dx == 0.0:
        return float(dy)
    if dy == 0.0:
        return float(dx)
    return float(math.hypot(dx, dy))


def _validation_failure(
    stats: dict[str, Any],
    *,
    check_name: str,
    reason: str,
    failed_part: str | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    stats["failed_check"] = check_name
    if failed_part is not None:
        stats["failed_part"] = failed_part
    return False, reason, stats


def validate_robot_render(
    *,
    geometry: RobotGeometry | None = None,
    width: int = 600,
    height: int = 600,
    render_state: RobotRenderState | None = None,
    validation_checks: RobotValidationChecksConfig | dict[str, Any] | None = None,
    **kwargs,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Cheap deterministic render validation."""

    if geometry is None:
        if render_state is None:
            render_state = sample_robot_render_state(
                seed=int(kwargs.pop("render_seed", 0)),
                render_space_mode=str(kwargs.get("render_space_mode", "legacy")),
                render_nuisance=kwargs.get("render_nuisance"),
                foot_orientation=kwargs.get("foot_orientation"),
            )
        geometry = compute_robot_geometry(
            render_state,
            width=width,
            height=height,
            **kwargs,
        )

    validation_cfg = _normalize_validation_checks(validation_checks)
    enabled_checks = validation_cfg.resolved_checks(geometry.state.mode)

    margin_px = max(1.0, 0.03 * min(geometry.width, geometry.height))
    bbox = geometry.overall_bbox
    stats: dict[str, Any] = {
        "bbox_x0": float(bbox[0]),
        "bbox_y0": float(bbox[1]),
        "bbox_x1": float(bbox[2]),
        "bbox_y1": float(bbox[3]),
        "margin_px": float(margin_px),
        "enabled_checks": sorted(
            name for name, is_enabled in enabled_checks.items() if is_enabled
        ),
        "failed_check": "",
        "failed_part": "",
        "min_clearance_pair": "",
        "min_clearance_px": np.nan,
    }
    if enabled_checks["bbox_in_frame"] and (
        bbox[0] < margin_px
        or bbox[1] < margin_px
        or bbox[2] > geometry.width - margin_px
        or bbox[3] > geometry.height - margin_px
    ):
        return _validation_failure(
            stats,
            check_name="bbox_in_frame",
            reason="bbox_out_of_frame",
        )

    image = _render_geometry_to_pil(geometry)
    arr = np.asarray(image)
    foreground = np.any(arr[:, :, :3] < 250, axis=2)
    foreground_area = int(np.count_nonzero(foreground))
    foreground_fraction = foreground_area / float(geometry.width * geometry.height)
    stats["foreground_area"] = foreground_area
    stats["foreground_fraction"] = foreground_fraction
    min_foreground_fraction = max(0.015, 12.0 / float(geometry.width * geometry.height))
    if enabled_checks["foreground_non_degenerate"] and (
        foreground_fraction < min_foreground_fraction
    ):
        return _validation_failure(
            stats,
            check_name="foreground_non_degenerate",
            reason="degenerate_foreground",
        )

    part_areas: dict[str, int] = {}
    for part in ("body", "head", "mouth", "feet", "hands", "eyes", "ears", "elbows", "knees"):
        mask = _render_geometry_to_pil(geometry, parts=(part,), mask=True, supersample=1)
        part_areas[part] = _mask_area(mask)
    stats["part_areas"] = dict(part_areas)

    min_main_part_area = max(2, int(round(0.004 * geometry.width * geometry.height)))
    if enabled_checks["main_parts_visible"]:
        for part in ("body", "head", "feet"):
            if part_areas[part] < min_main_part_area:
                return _validation_failure(
                    stats,
                    check_name="main_parts_visible",
                    reason=f"{part}_too_small",
                    failed_part=part,
                )

    min_mouth_area = max(1, int(round(0.0010 * geometry.width * geometry.height)))
    if enabled_checks["mouth_visible"] and part_areas["mouth"] < min_mouth_area:
        return _validation_failure(
            stats,
            check_name="mouth_visible",
            reason="mouth_too_small",
            failed_part="mouth",
        )

    min_hands_area = max(1, int(round(0.0012 * geometry.width * geometry.height)))
    if enabled_checks["hands_visible"] and part_areas["hands"] < min_hands_area:
        return _validation_failure(
            stats,
            check_name="hands_visible",
            reason="hands_too_small",
            failed_part="hands",
        )
    min_eyes_area = max(1, int(round(0.0008 * geometry.width * geometry.height)))
    if enabled_checks["eyes_visible"] and part_areas["eyes"] < min_eyes_area:
        return _validation_failure(
            stats,
            check_name="eyes_visible",
            reason="eyes_too_small",
            failed_part="eyes",
        )

    if enabled_checks["key_parts_not_clipped"]:
        for part in ("body", "head", "mouth", "feet"):
            part_bbox = geometry.part_bounds.get(part)
            if part_bbox is None:
                return _validation_failure(
                    stats,
                    check_name="key_parts_not_clipped",
                    reason=f"{part}_missing_bbox",
                    failed_part=part,
                )
            if (
                part_bbox[0] < 0.5
                or part_bbox[1] < 0.5
                or part_bbox[2] > geometry.width - 0.5
                or part_bbox[3] > geometry.height - 0.5
            ):
                return _validation_failure(
                    stats,
                    check_name="key_parts_not_clipped",
                    reason=f"{part}_clipped",
                    failed_part=part,
                )

    if (
        enabled_checks["no_knees_when_absent"]
        and geometry.features["has_knees"] == "false"
        and part_areas["knees"] > 0
    ):
        return _validation_failure(
            stats,
            check_name="no_knees_when_absent",
            reason="knees_visible_without_knees",
            failed_part="knees",
        )
    if (
        enabled_checks["no_elbows_when_absent"]
        and geometry.features["has_elbows"] == "false"
        and part_areas["elbows"] > 0
    ):
        return _validation_failure(
            stats,
            check_name="no_elbows_when_absent",
            reason="elbows_visible_without_elbows",
            failed_part="elbows",
        )

    topology_checks = (
        "hands_head_clearance",
        "hands_body_clearance",
        "feet_body_clearance",
        "elbows_head_clearance",
        "knees_body_clearance",
    )
    if any(enabled_checks[name] for name in topology_checks):
        supersample = validation_cfg.topology_mask_supersample
        part_masks = {
            part: _mask_bool_array(geometry, parts=(part,), supersample=supersample)
            for part in ("body", "head", "mouth", "feet", "hands", "eyes", "elbows", "knees")
        }
        region_masks = {
            "head_region": part_masks["head"] | part_masks["eyes"] | part_masks["mouth"],
            "body_region": part_masks["body"],
            "hands": part_masks["hands"],
            "feet": part_masks["feet"],
            "elbows": part_masks["elbows"],
            "knees": part_masks["knees"],
        }
        region_bounds = {
            "head_region": _merge_part_bounds(geometry, ("head", "eyes", "mouth")),
            "body_region": _merge_part_bounds(geometry, ("body",)),
            "hands": _merge_part_bounds(geometry, ("hands",)),
            "feet": _merge_part_bounds(geometry, ("feet",)),
            "elbows": _merge_part_bounds(geometry, ("elbows",)),
            "knees": _merge_part_bounds(geometry, ("knees",)),
        }
        check_specs = (
            ("hands_head_clearance", "hands", "head_region", validation_cfg.hands_head_clearance_frac),
            ("hands_body_clearance", "hands", "body_region", validation_cfg.hands_body_clearance_frac),
            ("feet_body_clearance", "feet", "body_region", validation_cfg.feet_body_clearance_frac),
            ("elbows_head_clearance", "elbows", "head_region", validation_cfg.elbows_head_clearance_frac),
            ("knees_body_clearance", "knees", "body_region", validation_cfg.knees_body_clearance_frac),
        )
        min_gap = float("inf")
        min_gap_pair = ""
        scaled_min_dim = supersample * min(geometry.width, geometry.height)
        for check_name, source_name, target_name, frac in check_specs:
            if not enabled_checks[check_name]:
                continue
            if check_name == "elbows_head_clearance" and geometry.features["has_elbows"] != "true":
                continue
            if check_name == "knees_body_clearance" and geometry.features["has_knees"] != "true":
                continue
            source_mask = region_masks[source_name]
            target_mask = region_masks[target_name]
            if not source_mask.any() or not target_mask.any():
                continue
            gap_px = _bbox_clearance(
                region_bounds[source_name],
                region_bounds[target_name],
            )
            if gap_px < min_gap:
                min_gap = gap_px
                min_gap_pair = check_name
            clearance_radius = max(0, int(math.ceil(float(frac) * scaled_min_dim)))
            dilated_source = _dilate_bool_mask(source_mask, clearance_radius)
            if np.any(dilated_source & target_mask):
                stats["min_clearance_pair"] = check_name
                stats["min_clearance_px"] = float(gap_px)
                return _validation_failure(
                    stats,
                    check_name=check_name,
                    reason=check_name,
                )
        if min_gap_pair:
            stats["min_clearance_pair"] = min_gap_pair
            stats["min_clearance_px"] = float(min_gap)

    return True, None, stats


def blur_parts(
    img,
    parts=("body",),
    radius=2.0,
    expand_mask_px=None,
    feather_mask_px=0,
    mask_mode: str = "uniform_rect",
    **kwargs,
):
    """Apply Gaussian blur to selected robot parts."""

    _, base_pil = image_to_numpy_and_pillow(img)
    mask_img = draw_robot_mask(
        width=int(base_pil.width),
        height=int(base_pil.height),
        parts=parts,
        mode=mask_mode,
        **kwargs,
    )
    _, mask_pil = image_to_numpy_and_pillow(mask_img)
    mask_l = mask_pil.convert("L")

    if expand_mask_px is None:
        expand_mask_px = int(round(radius))
    if expand_mask_px and expand_mask_px > 0:
        size = max(3, 2 * int(expand_mask_px) + 1)
        mask_l = mask_l.filter(PILImageFilter.MaxFilter(size=size))

    if feather_mask_px and feather_mask_px > 0:
        mask_l = mask_l.filter(PILImageFilter.GaussianBlur(radius=float(feather_mask_px)))

    blurred = base_pil.filter(PILImageFilter.GaussianBlur(radius=radius))
    return PILImage.composite(blurred, base_pil, mask_l)
