"""Typed dataclass configurations for concept benchmark experiments.

Each benchmark domain (robot, sudoku) gets a config class that produces
the same settings dicts used by the original scripts.  Factory methods
produce the exact defaults matching the paper results.
"""

from __future__ import annotations

__all__ = [
    "RobotBenchmarkConfig",
    "RobotRenderNuisanceConfig",
    "RobotValidationChecksConfig",
    "SudokuBenchmarkConfig",
    "PRESET_EXCLUDED_CONCEPTS",
    "TEXT_PRESET_EXCLUDED_CONCEPTS",
]

import copy
import hashlib
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import logging

import yaml

from concept_benchmark.formula import F, LabelFormula
from concept_benchmark.paths import data_dir, results_dir

logger = logging.getLogger(__name__)

# ── Robot benchmark constants ─────────────────────────────────────────

IDEAL_EXCLUDED_CONCEPTS = [
    "has_elbows",
    "hand_shape",
    "foot_shape_flat_rounded",
    "foot_shape_pointy_trapezoid",
    "foot_shape_pointy_3sided",
    "foot_shape_flat_lshaped",
    "foot_shape_pointy_4sided",
    "foot_shape_pointy_square",
    "foot_shape_pointy_rounded",
    "foot_shape_flat_5sided",
    "foot_shape_flat_square",
    "foot_shape_flat_trapezoid",
]

SUBCONCEPT_EXCLUDED_CONCEPTS = [
    "has_elbows",
    "hand_shape",
    "foot_shape_flat_rounded",
    "foot_shape_pointy_trapezoid",
    "foot_shape_pointy_3sided",
    "foot_shape_flat_lshaped",
    "foot_shape",
]

TEXT_IDEAL_EXCLUDED_CONCEPTS: list[str] = []
TEXT_SUBCONCEPT_EXCLUDED_CONCEPTS = ["has_elbows", "hands_are_pointy"]

PRESET_EXCLUDED_CONCEPTS = {
    "ground_truth": tuple(IDEAL_EXCLUDED_CONCEPTS),
    "foot_subtypes": tuple(SUBCONCEPT_EXCLUDED_CONCEPTS),
}

TEXT_PRESET_EXCLUDED_CONCEPTS = {
    "ground_truth": tuple(TEXT_IDEAL_EXCLUDED_CONCEPTS),
    "foot_subtypes": tuple(TEXT_SUBCONCEPT_EXCLUDED_CONCEPTS),
}

ROBOT_CONCEPTS = {
    "head_shape": ["square", "round"],
    "body_shape": ["square", "round"],
    "has_knees": ["false", "true"],
    "has_elbows": ["false", "true"],
    "has_antennae": ["false", "true"],
    "ears_shape": ["square", "triangle"],
    "mouth_type": ["closed", "open"],
    "hand_shape": [
        "round_circle",
        "round_oval",
        "round_oval2",
        "edgy_triangle",
        "edgy_square",
        "edgy_trapezoid",
    ],
    "foot_shape": [
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
    ],
}


ROBOT_SAMPLING_CONSTRAINTS = [
    {"concepts": {"foot_shape_pointy_square": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_pointy_rounded": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_pointy_4sided": 1}, "min_fraction": 0.49},
    {"concepts": {"foot_shape_flat_square": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_flat_trapezoid": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_flat_5sided": 1}, "min_fraction": 0.49},
]

ROBOT_VALIDATION_CHECK_MODES = frozenset({"auto", "on", "off"})

IMAGE_SIZE_TO_PIXELS = {
    "large": 600,
    "medium": 32,
    "small": 8,
}

MISSING_PROPORTION = 0.2

VALID_STRATEGIES = frozenset({"up_to_k", "exactly_k"})
VALID_CBM_FAMILIES = frozenset({"cbm", "cem", "probcbm"})
_CEM_FINGERPRINT_FIELDS = frozenset(
    {
        "cem_emb_size",
        "cem_training_intervention_prob",
        "cem_concept_loss_weight",
        "cem_task_loss_weight",
        "cem_max_epochs",
    }
)
_PROBCBM_FINGERPRINT_FIELDS = frozenset(
    {
        "probcbm_hidden_dim",
        "probcbm_class_hidden_dim",
        "probcbm_latent_dim",
        "probcbm_n_samples_inference",
        "probcbm_intervention_prob",
        "probcbm_max_epochs",
    }
)
ROBOT_VALID_REGIMES = frozenset(
    {"baseline", "expert", "subjective", "machine", "llm", "clip"}
)
ROBOT_TEXT_VALID_REGIMES = frozenset({"baseline", "expert", "subjective", "machine"})
ROBOT_RENDER_SPACE_MODES = frozenset(
    {"legacy", "continuous_light", "continuous_heavy"}
)
ROBOT_POSE_TEXT_MODES = frozenset({"neutral"})


# ── Shared utilities ──────────────────────────────────────────────────


def _dict_sha256(d: dict, *, truncate: int | None = None) -> str:
    """SHA-256 hex digest of a JSON-serialized dict."""
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    h = hashlib.sha256(blob).hexdigest()
    return h[:truncate] if truncate is not None else h


class _BenchmarkConfigBase:
    """Shared serialization and hashing for benchmark configs."""

    _yaml_exclude_fields = frozenset()  # fields to omit from YAML (secrets)

    def _prepare_asdict(self) -> dict:
        """Return a JSON/YAML-safe dict. Override for special field handling."""
        return asdict(self)

    @classmethod
    def _restore_from_yaml_dict(cls, d: dict) -> dict:
        """Post-process loaded YAML dict before passing to constructor."""
        return d

    def _scoped_field_names(self, scope: str) -> frozenset[str]:
        """Field names tagged with the given scope in their metadata."""
        return frozenset(
            f.name for f in fields(self) if f.metadata.get("scope") == scope
        )

    def to_yaml(self, path: str | Path) -> Path:
        """Serialize config to YAML and return the resolved path."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self._prepare_asdict()
        for k in self._yaml_exclude_fields:
            d.pop(k, None)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        logger.info("Config saved to: %s", path)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Load config from YAML."""
        with open(path) as f:
            d = yaml.safe_load(f)
        d = cls._restore_from_yaml_dict(d)
        return cls(**d)


# ── Robot Benchmark Config ────────────────────────────────────────────


@dataclass
class RobotRenderNuisanceConfig:
    """Ranges for nuisance-only robot render variation.

    Each field stores a ``[min, max]`` range. The ranges are interpreted by the
    robot renderer and do not affect semantic concepts or labels.
    """

    translate_x_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    translate_y_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    global_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    global_rotation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    body_aspect_x: list[float] = field(default_factory=lambda: [1.0, 1.0])
    body_aspect_y: list[float] = field(default_factory=lambda: [1.0, 1.0])
    head_aspect_x: list[float] = field(default_factory=lambda: [1.0, 1.0])
    head_aspect_y: list[float] = field(default_factory=lambda: [1.0, 1.0])
    arm_angle_offset_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    arm_length_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    arm_y_offset_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    left_arm_angle_delta_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    right_arm_angle_delta_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    left_arm_length_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    right_arm_length_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    left_arm_y_offset_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    right_arm_y_offset_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    leg_spread_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    leg_length_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    head_offset_x_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    head_offset_y_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    head_tilt_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    foot_rotation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    stroke_width_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    point_jitter_frac: list[float] = field(default_factory=lambda: [0.0, 0.0])
    color_jitter: list[float] = field(default_factory=lambda: [0.0, 0.0])

    @classmethod
    def legacy(cls) -> RobotRenderNuisanceConfig:
        return cls()

    @classmethod
    def continuous_light(cls) -> RobotRenderNuisanceConfig:
        return cls(
            translate_x_frac=[-0.04, 0.04],
            translate_y_frac=[-0.03, 0.03],
            global_scale=[0.94, 1.06],
            global_rotation_deg=[-5.0, 5.0],
            body_aspect_x=[0.95, 1.05],
            body_aspect_y=[0.95, 1.05],
            head_aspect_x=[0.94, 1.06],
            head_aspect_y=[0.94, 1.06],
            arm_angle_offset_deg=[-18.0, 22.0],
            arm_length_scale=[0.95, 1.05],
            arm_y_offset_frac=[-0.03, 0.03],
            left_arm_angle_delta_deg=[0.0, 0.0],
            right_arm_angle_delta_deg=[0.0, 0.0],
            left_arm_length_scale=[1.0, 1.0],
            right_arm_length_scale=[1.0, 1.0],
            left_arm_y_offset_frac=[0.0, 0.0],
            right_arm_y_offset_frac=[0.0, 0.0],
            leg_spread_deg=[-8.0, 10.0],
            leg_length_scale=[0.96, 1.04],
            head_offset_x_frac=[-0.03, 0.03],
            head_offset_y_frac=[-0.02, 0.02],
            head_tilt_deg=[-6.0, 6.0],
            foot_rotation_deg=[-10.0, 10.0],
            stroke_width_scale=[0.9, 1.12],
            point_jitter_frac=[0.0, 0.015],
            color_jitter=[-0.04, 0.04],
        )

    @classmethod
    def continuous_heavy(cls) -> RobotRenderNuisanceConfig:
        return cls(
            translate_x_frac=[-0.08, 0.08],
            translate_y_frac=[-0.06, 0.06],
            global_scale=[0.88, 1.12],
            global_rotation_deg=[-10.0, 10.0],
            body_aspect_x=[0.90, 1.10],
            body_aspect_y=[0.90, 1.10],
            head_aspect_x=[0.88, 1.12],
            head_aspect_y=[0.88, 1.12],
            arm_angle_offset_deg=[-35.0, 35.0],
            arm_length_scale=[0.90, 1.10],
            arm_y_offset_frac=[-0.06, 0.06],
            left_arm_angle_delta_deg=[0.0, 0.0],
            right_arm_angle_delta_deg=[0.0, 0.0],
            left_arm_length_scale=[1.0, 1.0],
            right_arm_length_scale=[1.0, 1.0],
            left_arm_y_offset_frac=[0.0, 0.0],
            right_arm_y_offset_frac=[0.0, 0.0],
            leg_spread_deg=[-16.0, 18.0],
            leg_length_scale=[0.90, 1.10],
            head_offset_x_frac=[-0.05, 0.05],
            head_offset_y_frac=[-0.04, 0.04],
            head_tilt_deg=[-12.0, 12.0],
            foot_rotation_deg=[-18.0, 18.0],
            stroke_width_scale=[0.82, 1.20],
            point_jitter_frac=[0.0, 0.03],
            color_jitter=[-0.08, 0.08],
        )

    @classmethod
    def for_mode(cls, mode: str) -> RobotRenderNuisanceConfig:
        if mode == "legacy":
            return cls.legacy()
        if mode == "continuous_light":
            return cls.continuous_light()
        if mode == "continuous_heavy":
            return cls.continuous_heavy()
        raise ValueError(
            f"render_space_mode must be one of {sorted(ROBOT_RENDER_SPACE_MODES)}, got {mode!r}"
        )

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"render_nuisance.{name} must be a 2-item list")
            lo, hi = float(value[0]), float(value[1])
            if lo > hi:
                raise ValueError(
                    f"render_nuisance.{name} must satisfy min <= max, got {value!r}"
                )
            setattr(self, name, [lo, hi])


@dataclass
class RobotValidationChecksConfig:
    """Per-check validation controls for robot render acceptance."""

    bbox_in_frame: bool = True
    foreground_non_degenerate: bool = True
    main_parts_visible: bool = True
    mouth_visible: bool = True
    hands_visible: bool = True
    eyes_visible: bool = True
    key_parts_not_clipped: bool = True
    no_knees_when_absent: bool = True
    no_elbows_when_absent: bool = True

    hands_head_clearance: str = "auto"
    hands_body_clearance: str = "auto"
    feet_body_clearance: str = "auto"
    elbows_head_clearance: str = "auto"
    knees_body_clearance: str = "auto"

    topology_mask_supersample: int = 4
    hands_head_clearance_frac: float = 0.015
    hands_body_clearance_frac: float = 0.010
    feet_body_clearance_frac: float = 0.010
    elbows_head_clearance_frac: float = 0.008
    knees_body_clearance_frac: float = 0.008

    _BASE_CHECK_FIELDS = (
        "bbox_in_frame",
        "foreground_non_degenerate",
        "main_parts_visible",
        "mouth_visible",
        "hands_visible",
        "eyes_visible",
        "key_parts_not_clipped",
        "no_knees_when_absent",
        "no_elbows_when_absent",
    )
    _TOPOLOGY_CHECK_FIELDS = (
        "hands_head_clearance",
        "hands_body_clearance",
        "feet_body_clearance",
        "elbows_head_clearance",
        "knees_body_clearance",
    )
    _CLEARANCE_THRESHOLD_FIELDS = {
        "hands_head_clearance": "hands_head_clearance_frac",
        "hands_body_clearance": "hands_body_clearance_frac",
        "feet_body_clearance": "feet_body_clearance_frac",
        "elbows_head_clearance": "elbows_head_clearance_frac",
        "knees_body_clearance": "knees_body_clearance_frac",
    }

    @classmethod
    def _resolve_check_mode(cls, mode: str, render_space_mode: str) -> bool:
        if mode == "on":
            return True
        if mode == "off":
            return False
        if mode != "auto":
            raise ValueError(
                f"validation check mode must be one of {sorted(ROBOT_VALIDATION_CHECK_MODES)}, got {mode!r}"
            )
        return render_space_mode != "legacy"

    def validate(self) -> None:
        for name in self._BASE_CHECK_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"validation_checks.{name} must be a boolean")
        for name in self._TOPOLOGY_CHECK_FIELDS:
            value = str(getattr(self, name))
            if value not in ROBOT_VALIDATION_CHECK_MODES:
                raise ValueError(
                    f"validation_checks.{name} must be one of {sorted(ROBOT_VALIDATION_CHECK_MODES)}, got {value!r}"
                )
            setattr(self, name, value)
        if int(self.topology_mask_supersample) < 1:
            raise ValueError(
                "validation_checks.topology_mask_supersample must be positive"
            )
        self.topology_mask_supersample = int(self.topology_mask_supersample)
        for name in self._CLEARANCE_THRESHOLD_FIELDS.values():
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"validation_checks.{name} must be non-negative")
            setattr(self, name, value)

    def resolved_checks(self, render_space_mode: str) -> dict[str, bool]:
        resolved = {
            name: bool(getattr(self, name)) for name in self._BASE_CHECK_FIELDS
        }
        for name in self._TOPOLOGY_CHECK_FIELDS:
            resolved[name] = self._resolve_check_mode(
                str(getattr(self, name)),
                render_space_mode=render_space_mode,
            )
        return resolved

    def pruned_for_mode(self, render_space_mode: str) -> dict[str, Any]:
        d = asdict(self)
        resolved = self.resolved_checks(render_space_mode)
        for check_name, threshold_name in self._CLEARANCE_THRESHOLD_FIELDS.items():
            if not resolved[check_name]:
                d.pop(threshold_name, None)
        if not any(resolved[name] for name in self._TOPOLOGY_CHECK_FIELDS):
            d.pop("topology_mask_supersample", None)
        return d


@dataclass
class RobotBenchmarkConfig(_BenchmarkConfigBase):
    """Configuration for the robot classification benchmark.

    Supports both image and text modalities via ``data_type``.
    """

    _yaml_exclude_fields = frozenset({"llm_api_key"})  # not a dataclass field

    # Data generation
    data_type: str = "image"
    image_size: str = field(default="medium", metadata={"scope": "image"})
    renders_per_robot: int = 4
    render_images: bool = field(default=True, metadata={"scope": "image"})
    seed: int = 1014
    rng_seed: int | None = None  # label RNG seed; None → use self.seed
    concepts: dict[str, list] = field(
        default_factory=lambda: copy.deepcopy(ROBOT_CONCEPTS)
    )
    label_formula: LabelFormula = field(
        default_factory=lambda: LabelFormula(
            score=(
                5 * F("mouth_type").closed
                + 8 * F("foot_shape").pointy
                - 5 * F("has_knees").true
                + 2
            ),
            temperature=4.2,
            stochastic=True,
        )
    )
    expand_concepts: list[str] = field(
        default_factory=lambda: ["foot_shape"],
    )
    color_mode: str = field(default="color", metadata={"scope": "image"})
    render_space_mode: str = "legacy"
    validate_renders: bool = True
    max_render_validation_attempts: int = 8
    group_split_by_semantic_id: bool = False
    include_pose_text: bool = False
    pose_text_mode: str = "neutral"
    render_nuisance: RobotRenderNuisanceConfig | dict | None = None
    validation_checks: RobotValidationChecksConfig | dict | None = None

    # Training (image)
    epochs: int = field(default=50, metadata={"scope": "image"})
    learning_rate: float = field(default=1e-3, metadata={"scope": "image"})
    patience: int = field(default=10, metadata={"scope": "image"})
    batch_size: int = field(default=32, metadata={"scope": "image"})
    cbm_family: str = "cbm"
    cem_emb_size: int = 16
    cem_training_intervention_prob: float = 0.25
    cem_concept_loss_weight: float = 1.0
    cem_task_loss_weight: float = 1.0
    cem_max_epochs: int | None = None
    probcbm_hidden_dim: int = 8
    probcbm_class_hidden_dim: int = 64
    probcbm_latent_dim: int = 8
    probcbm_n_samples_inference: int = 1
    probcbm_intervention_prob: float = 0.25
    probcbm_max_epochs: int | None = None

    # Intervention
    intervention_budgets: list[int] = field(default_factory=lambda: [1, 3])
    intervention_thresholds: list[float] = field(
        default_factory=lambda: [0.2, 0.4],
        metadata={"scope": "image"},
    )
    intervention_accuracy: float = 1.0
    intervention_strategy: str = "up_to_k"  # "up_to_k" or "exactly_k"

    # Intervention regimes
    intervention_regimes: list[str] = field(default_factory=lambda: ["baseline"])
    expert_intervention_accuracy: float = 0.80
    subjective_noise_rate: float = 0.20
    subjective_intervention_accuracy: float = 0.80
    label_free_concepts_file: str = field(default="", metadata={"scope": "image"})
    llm_concepts_file: str = field(default="", metadata={"scope": "image"})
    clip_concepts_file: str = field(default="", metadata={"scope": "image"})
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    llm_api_key: str = ""
    llm_api_key_env: str = "GEMINI_API_KEY"
    force_retrain: bool = False  # force retrain LFCBM/subjective models

    # Alignment (sign constraints for constrained retraining)
    alignment_constraints: dict[str, int] | None = None

    # Variant
    concept_preset: str = "ground_truth"

    # ── Text generation (data_type="text" only) ──────────────────────
    template_complexity: str = field(default="high", metadata={"scope": "text"})

    # ── Text CBM training ────────────────────────────────────────────
    detector_epochs: int = field(default=6, metadata={"scope": "text"})
    detector_batch_size: int = field(default=64, metadata={"scope": "text"})
    detector_lr: float = field(default=2e-3, metadata={"scope": "text"})
    concept_output_type: str = field(default="binary", metadata={"scope": "text"})

    # ── Text DNN training ────────────────────────────────────────────
    dnn_model_name: str = field(
        default="distilbert-base-uncased", metadata={"scope": "text"}
    )
    dnn_epochs: int = field(default=3, metadata={"scope": "text"})
    dnn_batch_size: int = field(default=16, metadata={"scope": "text"})
    dnn_lr: float = field(default=5e-5, metadata={"scope": "text"})

    # ── Text intervention ────────────────────────────────────────────
    concept_uncertainty_threshold: float = field(
        default=0.30, metadata={"scope": "text"}
    )

    # ── Text LFCBM ───────────────────────────────────────────────────
    use_label_free_concepts: bool = field(default=False, metadata={"scope": "text"})
    label_free_encoder: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2", metadata={"scope": "text"}
    )
    label_free_concepts_csv: str = field(default="", metadata={"scope": "text"})

    def __post_init__(self):
        if self.data_type not in ("image", "text"):
            raise ValueError(
                f"data_type must be 'image' or 'text', got {self.data_type!r}"
            )
        if isinstance(self.label_formula, dict):
            self.label_formula = LabelFormula.from_dict(self.label_formula)
        if isinstance(self.render_nuisance, dict):
            self.render_nuisance = RobotRenderNuisanceConfig(**self.render_nuisance)
        if self.render_nuisance is None:
            self.render_nuisance = RobotRenderNuisanceConfig.for_mode(
                self.render_space_mode
            )
        if isinstance(self.validation_checks, dict):
            self.validation_checks = RobotValidationChecksConfig(
                **self.validation_checks
            )
        if self.validation_checks is None:
            self.validation_checks = RobotValidationChecksConfig()
        if self.data_type == "text":
            self._auto_configure_text()
        self._validate_common()
        if self.data_type == "text":
            self._validate_text()
        else:
            self._validate_image()

    def _auto_configure_text(self):
        """Auto-switch defaults for text modality.

        Uses field defaults as sentinels: if the user hasn't changed them from
        the image-oriented defaults, we override to text-appropriate values.
        """
        # For ground_truth preset, clear image-default expand_concepts
        # so text uses collapsed binary features (9 concepts like the original)
        if self.concept_preset == "ground_truth" and self.expand_concepts == [
            "foot_shape"
        ]:
            self.expand_concepts = []
        # Text-appropriate renders_per_robot default
        if self.renders_per_robot == 4:
            self.renders_per_robot = 1

    def _validate_common(self):
        """Validate parameters shared across data types."""
        if self.concept_preset not in ("ground_truth", "foot_subtypes"):
            raise ValueError(
                f"concept_preset must be 'ground_truth' or 'foot_subtypes', "
                f"got {self.concept_preset!r}"
            )
        self.label_formula.validate_against(self.concepts)
        if self.cbm_family not in VALID_CBM_FAMILIES:
            raise ValueError(
                f"cbm_family must be one of {sorted(VALID_CBM_FAMILIES)}, "
                f"got {self.cbm_family!r}"
            )
        if self.render_space_mode not in ROBOT_RENDER_SPACE_MODES:
            raise ValueError(
                f"render_space_mode must be one of {sorted(ROBOT_RENDER_SPACE_MODES)}, "
                f"got {self.render_space_mode!r}"
            )
        if self.pose_text_mode not in ROBOT_POSE_TEXT_MODES:
            raise ValueError(
                f"pose_text_mode must be one of {sorted(ROBOT_POSE_TEXT_MODES)}, "
                f"got {self.pose_text_mode!r}"
            )
        if self.max_render_validation_attempts < 1:
            raise ValueError(
                "max_render_validation_attempts must be positive, got "
                f"{self.max_render_validation_attempts}"
            )
        self.render_nuisance.validate()
        self.validation_checks.validate()

        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if any(b < -1 for b in self.intervention_budgets):
            raise ValueError(
                f"intervention_budgets must be non-negative (or -1 for max), got {self.intervention_budgets}"
            )
        if self.intervention_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"intervention_strategy must be one of {sorted(VALID_STRATEGIES)}, "
                f"got {self.intervention_strategy!r}"
            )

    def _validate_image(self):
        """Validate image-specific parameters."""
        if self.image_size not in IMAGE_SIZE_TO_PIXELS:
            raise ValueError(
                f"image_size must be one of {sorted(IMAGE_SIZE_TO_PIXELS)}, "
                f"got {self.image_size!r}"
            )
        unknown = set(self.intervention_regimes) - ROBOT_VALID_REGIMES
        if unknown:
            raise ValueError(
                f"unknown intervention regimes: {sorted(unknown)}. "
                f"Valid: {sorted(ROBOT_VALID_REGIMES)}"
            )

    def _validate_text(self):
        """Validate text-specific parameters."""
        if self.template_complexity not in ("high", "medium", "low"):
            raise ValueError(
                f"template_complexity must be 'high', 'medium', or 'low', "
                f"got {self.template_complexity!r}"
            )
        unknown = set(self.intervention_regimes) - ROBOT_TEXT_VALID_REGIMES
        if unknown:
            raise ValueError(
                f"unknown intervention regimes: {sorted(unknown)}. "
                f"Valid: {sorted(ROBOT_TEXT_VALID_REGIMES)}"
            )

    # ── Derived properties ─────────────────────────────────────────

    @property
    def _labeling_tag(self) -> str:
        """``'stochastic'`` or ``'deterministic'`` for filename construction."""
        return "stochastic" if self.label_formula.stochastic else "deterministic"

    @property
    def _preset_suffix(self) -> str:
        """``'_subconcept'`` or ``'_ideal'`` for filename construction."""
        return "_subconcept" if self.concept_preset == "foot_subtypes" else "_ideal"

    @property
    def pixel_resolution(self) -> int:
        """Pixel resolution for the current image_size."""
        return IMAGE_SIZE_TO_PIXELS[self.image_size]

    @classmethod
    def default_ideal(cls) -> RobotBenchmarkConfig:
        """Config matching the paper's ideal robot benchmark."""
        return cls()

    @classmethod
    def default_subconcept(cls) -> RobotBenchmarkConfig:
        """Config matching the paper's subconcept robot benchmark."""
        return cls(concept_preset="foot_subtypes")

    @classmethod
    def continuous_light(cls, **kwargs) -> RobotBenchmarkConfig:
        """Config preset for light continuous render variation."""
        return cls(render_space_mode="continuous_light", **kwargs)

    @classmethod
    def continuous_heavy(cls, **kwargs) -> RobotBenchmarkConfig:
        """Config preset for heavier continuous render variation."""
        return cls(render_space_mode="continuous_heavy", **kwargs)

    @property
    def input_size(self) -> int:
        if self.data_type == "text":
            raise ValueError("input_size is not defined for text data_type")
        return self.pixel_resolution

    def _prepare_asdict(self) -> dict:
        """Return a JSON/YAML-safe dict, serializing label_formula."""
        d = asdict(self)
        d["label_formula"] = self.label_formula.to_dict()
        return d

    @classmethod
    def _restore_from_yaml_dict(cls, d: dict) -> dict:
        """Convert label_formula dict back to LabelFormula on load."""
        if "label_formula" in d and isinstance(d["label_formula"], dict):
            d["label_formula"] = LabelFormula.from_dict(d["label_formula"])
        if "render_nuisance" in d and isinstance(d["render_nuisance"], dict):
            d["render_nuisance"] = RobotRenderNuisanceConfig(**d["render_nuisance"])
        if "validation_checks" in d and isinstance(d["validation_checks"], dict):
            d["validation_checks"] = RobotValidationChecksConfig(
                **d["validation_checks"]
            )
        return d

    def _prune_irrelevant_robot_fields(self, d: dict[str, Any]) -> dict[str, Any]:
        """Drop fields that cannot affect the configured robot dataset."""
        render_irrelevant = self.render_space_mode == "legacy" or (
            self.data_type == "text" and not self.include_pose_text
        )
        if render_irrelevant:
            for key in (
                "render_nuisance",
                "validate_renders",
                "max_render_validation_attempts",
                "validation_checks",
            ):
                d.pop(key, None)
        elif self.data_type == "text":
            d.pop("validate_renders", None)
            d.pop("max_render_validation_attempts", None)
            d.pop("validation_checks", None)
        elif not self.validate_renders:
            d.pop("max_render_validation_attempts", None)
            d.pop("validation_checks", None)
        elif "validation_checks" in d:
            checks = d["validation_checks"]
            if not isinstance(checks, RobotValidationChecksConfig):
                checks = RobotValidationChecksConfig(**checks)
            d["validation_checks"] = checks.pruned_for_mode(self.render_space_mode)
        if self.data_type != "text" or not self.include_pose_text:
            d.pop("pose_text_mode", None)
        if not self.group_split_by_semantic_id:
            d.pop("group_split_by_semantic_id", None)
        if self.data_type != "text":
            d.pop("include_pose_text", None)
        elif not self.include_pose_text:
            d.pop("include_pose_text", None)
        return d

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict compatible with DEFAULT_ROBOT_SETTINGS.

        Only supported for image data_type.
        """
        if self.data_type == "text":
            raise NotImplementedError("to_dict() is not supported for data_type='text'")
        return {
            "data_type": self.data_type,
            "samples_per_instance": self.renders_per_robot,
            "draw": self.render_images,
            "output_directory": data_dir / "robot_images",
            "size": self.image_size,
            "color_mode": self.color_mode,
            "train_dnn": 0,
            "seed": self.seed,
            "rng_seed": self.rng_seed if self.rng_seed is not None else self.seed,
            "concepts": copy.deepcopy(self.concepts),
            "additional_features": list(self.expand_concepts),
            "subconcept": self.concept_preset == "foot_subtypes",
            "model_type": self._labeling_tag,
            "model_features": dict(self.label_formula.features),
            "model_weights": dict(self.label_formula.weights),
            "model_intercept": self.label_formula.intercept,
            "model_scalar": self.label_formula.temperature,
            "render_space_mode": self.render_space_mode,
            "validate_renders": self.validate_renders,
            "max_render_validation_attempts": self.max_render_validation_attempts,
            "group_split_by_semantic_id": self.group_split_by_semantic_id,
            "render_nuisance": asdict(self.render_nuisance),
            "validation_checks": asdict(self.validation_checks),
        }

    def setup_fingerprint(self) -> str:
        """Hash of all parameters that affect data generation."""
        if self.data_type == "text":
            d = self._prepare_asdict()
            # Remove non-data params and image-only fields
            _exclude = self._scoped_field_names("image") | {
                "llm_api_key",
                "llm_api_key_env",
                "force_retrain",
                "use_label_free_concepts",
                "label_free_encoder",
                "label_free_concepts_csv",
                "alignment_constraints",
                "intervention_budgets",
                "intervention_strategy",
                "intervention_regimes",
                "intervention_accuracy",
                "intervention_thresholds",
                "expert_intervention_accuracy",
                "subjective_noise_rate",
                "subjective_intervention_accuracy",
                "label_free_concepts_file",
                "llm_concepts_file",
                "clip_concepts_file",
                "llm_provider",
                "llm_model",
                "cbm_family",
                "cem_emb_size",
                "cem_training_intervention_prob",
                "cem_concept_loss_weight",
                "cem_task_loss_weight",
                "cem_max_epochs",
                "probcbm_hidden_dim",
                "probcbm_class_hidden_dim",
                "probcbm_latent_dim",
                "probcbm_n_samples_inference",
                "probcbm_intervention_prob",
                "probcbm_max_epochs",
            }
            for k in _exclude:
                d.pop(k, None)
            d = self._prune_irrelevant_robot_fields(d)
            return _dict_sha256(d)

        d = self.to_dict()
        d.pop("draw", None)  # meta-flag, not a data param
        d.pop("output_directory", None)  # derived from image_resolution
        d.pop("train_dnn", None)  # not a data param
        d = self._prune_irrelevant_robot_fields(d)
        return _dict_sha256(d)

    def model_fingerprint(self, model_class: str | None = None) -> str:
        """Hash of all parameters that affect model training."""
        d = self._prepare_asdict()
        # Remove params that don't affect training
        for k in (
            "llm_api_key",
            "llm_api_key_env",
            "force_retrain",
            "render_images",
            "alignment_constraints",
            "intervention_budgets",
            "intervention_thresholds",
            "intervention_accuracy",
            "intervention_strategy",
            "intervention_regimes",
            "expert_intervention_accuracy",
            "subjective_noise_rate",
            "subjective_intervention_accuracy",
            "label_free_concepts_file",
            "llm_concepts_file",
            "clip_concepts_file",
            "llm_provider",
            "llm_model",
        ):
            d.pop(k, None)
        # Exclude data_type-irrelevant fields
        exclude = (
            self._scoped_field_names("text")
            if self.data_type != "text"
            else self._scoped_field_names("image")
        )
        for k in exclude:
            d.pop(k, None)
        d = self._prune_irrelevant_robot_fields(d)
        if model_class is not None:
            d.pop("cbm_family", None)
            if model_class in {"cbm", "cbm_subjective", "lfcbm", "dnn"}:
                for k in _CEM_FINGERPRINT_FIELDS | _PROBCBM_FINGERPRINT_FIELDS:
                    d.pop(k, None)
            elif model_class == "cem":
                for k in _PROBCBM_FINGERPRINT_FIELDS:
                    d.pop(k, None)
            elif model_class == "probcbm":
                for k in _CEM_FINGERPRINT_FIELDS:
                    d.pop(k, None)
        return _dict_sha256(d)

    def get_dataset_path(self) -> Path:
        """Return the path where the dataset file is saved."""
        if self.data_type == "text":
            return results_dir / f"robot_text_seed{self.seed}.data"
        filename = (
            f"robot_{self.data_type}_{self.renders_per_robot}{self._preset_suffix}"
        )
        return results_dir / f"{filename}.data"

    def get_model_path(self, model_class: str) -> Path:
        """Return the path where a trained model is saved."""
        if self.data_type == "text":
            return results_dir / f"robot_text_{model_class}_seed{self.seed}.model"
        filename = (
            f"robot_{self.data_type}_{self._labeling_tag}_{self.renders_per_robot}"
            f"{self._preset_suffix}"
            f"_{model_class}.model"
        )
        return results_dir / filename

    def get_results_path(self, model_class: str = "cbm") -> Path:
        """Return the path where results CSV is saved."""
        if self.data_type == "text":
            return results_dir / f"robot_text_{model_class}_seed{self.seed}_results.csv"
        filename = f"robot_{self.data_type}_{self._labeling_tag}"
        if model_class in {"cbm", "cem", "probcbm"}:
            filename += self._preset_suffix
        filename += f"_{model_class}_results.csv"
        return results_dir / filename

    def get_alignment_constraints(self) -> dict[str, int]:
        """Return monotonicity constraints for alignment.

        Default: ``{"has_knees": 1}`` for image, ``{"mouth_is_open": 1}``
        for text.  Override via ``alignment_constraints``.
        """
        if self.alignment_constraints is not None:
            return self.alignment_constraints
        if self.data_type == "text":
            return {"mouth_is_open": 1}
        return {"has_knees": 1}

    def get_alignment_results_path(self) -> Path:
        """Return the path where alignment results JSON is saved."""
        if self.data_type == "text":
            return results_dir / f"robot_text_alignment_seed{self.seed}.json"
        filename = (
            f"robot_{self.data_type}_{self._labeling_tag}"
            f"{self._preset_suffix}"
            f"_alignment.json"
        )
        return results_dir / filename

    def _config_hash(self) -> str:
        """Short hex hash of all config fields for filename uniqueness."""
        d = self._prepare_asdict()
        d.pop("llm_api_key", None)
        # Exclude data_type-irrelevant fields for stable hashes
        exclude = (
            self._scoped_field_names("text")
            if self.data_type != "text"
            else self._scoped_field_names("image")
        )
        for k in exclude:
            d.pop(k, None)
        d = self._prune_irrelevant_robot_fields(d)
        return _dict_sha256(d, truncate=8)

    def get_collect_path(self) -> Path:
        """Return the path for the collect-stage summary CSV."""
        if self.data_type == "text":
            return results_dir / "robot_text_results.csv"
        variant = "subconcept" if self.concept_preset == "foot_subtypes" else "ideal"
        return (
            results_dir
            / f"robot_{variant}_seed{self.seed}_{self._config_hash()}_results.csv"
        )


# ── Sudoku Benchmark Config ──────────────────────────────────────────


@dataclass
class SudokuBenchmarkConfig(_BenchmarkConfigBase):
    """Configuration for the sudoku validation benchmark."""

    block_size: int = 3
    n_boards: int = 1000
    valid_board_ratio: float = 0.5
    max_cell_swaps: int = 9
    seed: int = 171
    data_type: str = "image"
    render_images: bool = True

    # Training
    epochs: int = 20
    patience: int = 5
    batch_size: int = 32
    cs_epochs: int = 100
    cs_patience: int = 20
    cbm_family: str = "cbm"
    cem_emb_size: int = 16
    cem_training_intervention_prob: float = 0.25
    cem_concept_loss_weight: float = 1.0
    cem_task_loss_weight: float = 1.0
    cem_max_epochs: int | None = None
    probcbm_hidden_dim: int = 8
    probcbm_class_hidden_dim: int = 64
    probcbm_latent_dim: int = 8
    probcbm_n_samples_inference: int = 1
    probcbm_intervention_prob: float = 0.25
    probcbm_max_epochs: int | None = None

    # Intervention
    intervention_budgets: list[int] = field(default_factory=lambda: [1, 3, 27])
    intervention_thresholds: list[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8]
    )
    target_accuracy: float = 0.9
    decision_threshold: float = 0.5

    # Alignment
    alignment_weights: dict[str, float] | None = None

    # OCR settings
    cell_px: int = 50
    cell_margin_px: int = 2
    gridline_px: int = 2
    block_border_px: int = 5
    font_size: int = 25
    font_style: str = "handwritten"  # "handwritten" or "printed"

    def __post_init__(self):
        if self.font_style not in ("handwritten", "printed"):
            raise ValueError(
                f"font_style must be 'handwritten' or 'printed', got {self.font_style!r}"
            )
        if self.cbm_family not in VALID_CBM_FAMILIES:
            raise ValueError(
                f"cbm_family must be one of {sorted(VALID_CBM_FAMILIES)}, "
                f"got {self.cbm_family!r}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.n_boards < 1:
            raise ValueError(f"n_boards must be positive, got {self.n_boards}")

    @classmethod
    def default(cls) -> SudokuBenchmarkConfig:
        """Config matching the paper's sudoku benchmark."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict compatible with DEFAULT_SUDOKU_SETTINGS."""
        return {
            "data_name": "sudoku",
            "n": self.block_size,
            "n_samples": self.n_boards,
            "valid_ratio": self.valid_board_ratio,
            "max_corrupt": self.max_cell_swaps,
            "seed": self.seed,
            "temp_train_data_path": (
                data_dir
                / "sudoku"
                / "multimodal_m_21"
                / "tabular"
                / "sudoku_dataset.pkl"
            ),
            "epochs": self.epochs,
            "patience": self.patience,
        }

    def setup_fingerprint(self) -> str:
        """Hash of all parameters that affect data generation."""
        d = self.to_dict()
        d.pop("temp_train_data_path", None)
        return _dict_sha256(d)

    def model_fingerprint(self, model_class: str | None = None) -> str:
        """Hash of all parameters that affect model training."""
        d = self._prepare_asdict()
        # Remove params that don't affect training
        for k in (
            "render_images",
            "intervention_thresholds",
            "target_accuracy",
            "decision_threshold",
            "alignment_weights",
        ):
            d.pop(k, None)
        if model_class is not None:
            d.pop("cbm_family", None)
            if model_class in {"cs", "dnn", "ocr"}:
                for k in _CEM_FINGERPRINT_FIELDS | _PROBCBM_FINGERPRINT_FIELDS:
                    d.pop(k, None)
            elif model_class == "cem":
                for k in _PROBCBM_FINGERPRINT_FIELDS:
                    d.pop(k, None)
            elif model_class == "probcbm":
                for k in _CEM_FINGERPRINT_FIELDS:
                    d.pop(k, None)
        return _dict_sha256(d)

    def get_dataset_path(self, data_type: str | None = None) -> Path:
        """Return the directory path for the dataset."""
        dt = data_type or self.data_type
        filename = f"sudoku_{dt}_n{self.block_size}_ns{self.n_boards}_mc{self.max_cell_swaps}_seed{self.seed}"
        return data_dir / "sudoku" / filename

    def get_model_path(self, model_class: str, data_type: str | None = None) -> Path:
        """Return the path where a trained model is saved."""
        dt = data_type or self.data_type
        filename = (
            f"sudoku_{model_class}_{dt}_n{self.block_size}_mc{self.max_cell_swaps}"
        )
        return results_dir / f"{filename}.model"

    def get_results_path(
        self, model_class: str = "cbm", data_type: str | None = None
    ) -> Path:
        """Return the path where results are saved."""
        dt = data_type or self.data_type
        filename = (
            f"sudoku_{model_class}_{dt}_n{self.block_size}_mc{self.max_cell_swaps}"
        )
        return results_dir / f"{filename}.results"

    def get_alignment_weights(self) -> dict[str, float]:
        """Return alignment weights, computing defaults if not explicitly set.

        Default: all row/col/block concepts get weight 1.0 (AND semantics).
        """
        if self.alignment_weights is not None:
            return self.alignment_weights
        board_size = self.block_size * self.block_size  # 9 for standard sudoku
        weights: dict[str, float] = {}
        for i in range(board_size):
            weights[f"row_valid_{i + 1}"] = 1.0
            weights[f"col_valid_{i + 1}"] = 1.0
            weights[f"block_valid_{i + 1}"] = 1.0
        weights["bias"] = -(3 * board_size - 0.5)
        return weights

    def get_alignment_results_path(self, data_type: str | None = None) -> Path:
        """Return the path where alignment results JSON is saved."""
        dt = data_type or self.data_type
        filename = f"sudoku_alignment_{dt}_n{self.block_size}_mc{self.max_cell_swaps}"
        return results_dir / f"{filename}.json"

    def _config_hash(self) -> str:
        """Short hex hash of all config fields for filename uniqueness."""
        return _dict_sha256(self._prepare_asdict(), truncate=8)

    def get_collect_path(self) -> Path:
        """Return the path for the collect-stage summary CSV."""
        return results_dir / f"sudoku_seed{self.seed}_{self._config_hash()}_results.csv"
