"""Typed dataclass configurations for concept benchmark experiments.

Each benchmark domain (robot, sudoku) gets a config class that produces
the same settings dicts used by the original scripts.  Factory methods
produce the exact defaults matching the paper results.
"""
from __future__ import annotations

__all__ = [
    "RobotBenchmarkConfig",
    "SudokuBenchmarkConfig",
    "RobotTextBenchmarkConfig",
]

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from concept_benchmark.paths import data_dir, results_dir

# ── Robot benchmark constants ─────────────────────────────────────────

IDEAL_DROP = [
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

SUBCONCEPT_DROP = [
    "foot_shape_flat_rounded",
    "foot_shape_pointy_trapezoid",
    "foot_shape_pointy_3sided",
    "foot_shape_flat_lshaped",
    "foot_shape",
]

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

ROBOT_MODEL_RULE = (
    "'glorp' if (int(row['mouth_type']=='closed') "
    "+ int(row['foot_shape']=='pointy') "
    "+ int(row['has_knees']=='true'))>= 3 else 'drent'"
)

ROBOT_SKEW_SPECS = [
    {"concepts": {"foot_shape_pointy_square": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_pointy_rounded": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_pointy_4sided": 1}, "min_fraction": 0.49},
    {"concepts": {"foot_shape_flat_square": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_flat_trapezoid": 1}, "min_fraction": 0.005},
    {"concepts": {"foot_shape_flat_5sided": 1}, "min_fraction": 0.49},
]

INPUT_MAP = {
    "large": 600,
    "medium": 32,
    "small": 8,
}

MISSING_PROP = 0.2


# ── Robot Benchmark Config ────────────────────────────────────────────

@dataclass
class RobotBenchmarkConfig:
    """Configuration for the robot classification benchmark."""

    # Data generation
    data_type: str = "image"
    size: str = "medium"
    samples_per_instance: int = 4
    draw: int = 0
    seed: int = 1014
    concepts: Dict[str, list] = field(default_factory=lambda: copy.deepcopy(ROBOT_CONCEPTS))
    model_rule: str = ROBOT_MODEL_RULE
    model_type: str = "stochastic"
    scalar: float = 4.2
    intercept: float = -2.0
    weights: Dict[str, float] = field(
        default_factory=lambda: {"mouth_type": 5, "foot_shape": 8, "has_knees": -5}
    )
    test_size: int = 10000
    train_skew_size: int = 3800
    skew_specs: List[Dict] = field(default_factory=lambda: copy.deepcopy(ROBOT_SKEW_SPECS))
    drop_concepts: List[str] = field(default_factory=lambda: list(IDEAL_DROP))
    additional_features: List[str] = field(default_factory=lambda: ["foot_shape_subtype"])
    spurious_features: List[str] = field(default_factory=lambda: ["has_elbows", "hand_shape"])
    color_mode: str = "color"
    knows_concepts: bool = False

    # Training
    epochs: int = 50
    lr: float = 1e-3
    patience: int = 10
    batch_size: int = 32

    # Intervention
    intervention_budgets: List[int] = field(default_factory=lambda: [1, 3])
    intervention_thresholds: List[float] = field(default_factory=lambda: [0.2, 0.4])
    intervention_accuracy: float = 1.0
    intervention_strategy: str = "kflip"  # "kflip" (up-to-k) or "exact_k"

    # Intervention regimes
    intervention_regimes: List[str] = field(default_factory=lambda: ["baseline"])
    expert_intervention_accuracy: float = 0.80
    subjective_noise_rate: float = 0.20
    subjective_intervention_accuracy: float = 0.80
    lfcbm_concepts_file: str = ""
    llm_concepts_file: str = ""
    clip_concepts_file: str = ""
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    llm_api_key: str = ""
    llm_api_key_env: str = "GEMINI_API_KEY"
    force_retrain: bool = False  # force retrain LFCBM/subjective models

    # Missingness
    concept_missing: float = 0.0
    concept_missing_mech: str = "none"

    # Alignment (sign constraints for constrained retraining)
    alignment_constraints: Optional[Dict[str, int]] = None

    # Variant
    subconcept: bool = False

    _VALID_REGIMES = frozenset({"baseline", "expert", "subjective", "machine", "llm", "clip"})
    _VALID_STRATEGIES = frozenset({"kflip", "exact_k"})

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.size not in INPUT_MAP:
            raise ValueError(
                f"size must be one of {sorted(INPUT_MAP)}, got {self.size!r}"
            )
        if any(b < 0 for b in self.intervention_budgets):
            raise ValueError(f"intervention_budgets must be non-negative, got {self.intervention_budgets}")
        if self.intervention_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"intervention_strategy must be one of {sorted(self._VALID_STRATEGIES)}, "
                f"got {self.intervention_strategy!r}"
            )
        unknown = set(self.intervention_regimes) - self._VALID_REGIMES
        if unknown:
            raise ValueError(
                f"unknown intervention regimes: {sorted(unknown)}. "
                f"Valid: {sorted(self._VALID_REGIMES)}"
            )

    @classmethod
    def default_ideal(cls) -> RobotBenchmarkConfig:
        """Config matching the paper's ideal robot benchmark."""
        return cls()

    @classmethod
    def default_subconcept(cls) -> RobotBenchmarkConfig:
        """Config matching the paper's subconcept robot benchmark."""
        return cls(
            subconcept=True,
            drop_concepts=list(SUBCONCEPT_DROP),
        )

    @property
    def input_size(self) -> int:
        return INPUT_MAP[self.size]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict compatible with DEFAULT_ROBOT_SETTINGS."""
        return {
            "data_type": self.data_type,
            "samples_per_instance": self.samples_per_instance,
            "draw": self.draw,
            "output_directory": data_dir / "robot_images",
            "size": self.size,
            "color_mode": self.color_mode,
            "train_dnn": 0,
            "seed": self.seed,
            "model": self.model_rule,
            "test_size": self.test_size,
            "train_skew_size": self.train_skew_size,
            "knows_concepts": self.knows_concepts,
            "concepts": copy.deepcopy(self.concepts),
            "additional_features": list(self.additional_features),
            "spurious_features": list(self.spurious_features),
            "subconcept": self.subconcept,
            "drop_concepts": list(self.drop_concepts),
            "model_type": self.model_type,
            "scalar": self.scalar,
            "intercept": self.intercept,
            "weights": dict(self.weights),
            "skew_specs": copy.deepcopy(self.skew_specs),
            "concept_missing": self.concept_missing,
            "concept_missing_mech": self.concept_missing_mech,
        }

    def get_dataset_path(self) -> Path:
        """Return the path where the dataset file is saved."""
        filename = f"robot_{self.data_type}_{self.samples_per_instance}"
        if self.subconcept:
            filename += "_subconcept"
        else:
            filename += "_ideal"
        return results_dir / f"{filename}.data"

    def get_model_path(self, model_class: str) -> Path:
        """Return the path where a trained model is saved."""
        filename = f"robot_{self.data_type}_{self.model_type}_{self.samples_per_instance}"
        if self.subconcept:
            filename += "_subconcept"
        else:
            filename += "_ideal"
        if self.concept_missing_mech != "none":
            filename += f"_{self.concept_missing_mech}_{int(self.concept_missing * 100)}"
        filename += f"_{model_class}.model"
        return results_dir / filename

    def get_results_path(self, model_class: str = "cbm") -> Path:
        """Return the path where results CSV is saved."""
        filename = f"robot_{self.data_type}_{self.model_type}"
        if model_class == "cbm":
            if self.subconcept:
                filename += "_subconcept"
            else:
                filename += "_ideal"
        if self.concept_missing_mech != "none":
            filename += f"_{self.concept_missing_mech}_{int(self.concept_missing * 100)}"
        filename += f"_{model_class}_results.csv"
        return results_dir / filename

    def get_alignment_constraints(self) -> Dict[str, int]:
        """Return monotonicity constraints for alignment.

        Default: ``{"has_knees": 1}`` — the single constraint used in the
        paper.  Override via ``alignment_constraints`` to test other settings.
        """
        if self.alignment_constraints is not None:
            return self.alignment_constraints
        return {"has_knees": 1}

    def get_alignment_results_path(self) -> Path:
        """Return the path where alignment results JSON is saved."""
        filename = f"robot_{self.data_type}_{self.model_type}"
        if self.subconcept:
            filename += "_subconcept"
        else:
            filename += "_ideal"
        if self.concept_missing_mech != "none":
            filename += f"_{self.concept_missing_mech}_{int(self.concept_missing * 100)}"
        filename += "_alignment.json"
        return results_dir / filename

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to YAML."""
        d = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotBenchmarkConfig:
        """Load config from YAML."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)


# ── Sudoku Benchmark Config ──────────────────────────────────────────

@dataclass
class SudokuBenchmarkConfig:
    """Configuration for the sudoku validation benchmark."""

    n: int = 3
    n_samples: int = 1000
    valid_ratio: float = 0.5
    max_corrupt: int = 9
    seed: int = 171
    data_type: str = "tabular"

    # Training
    epochs: int = 20
    patience: int = 5
    batch_size: int = 32
    cs_epochs: int = 100
    cs_patience: int = 20

    # Missingness
    concept_missing: float = 0.0
    concept_missing_mech: str = "none"

    # Intervention
    intervention_thresholds: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8]
    )
    target_accuracy: float = 0.9
    decision_threshold: float = 0.5

    # Alignment
    alignment_weights: Optional[Dict[str, float]] = None

    # OCR settings
    cell_px: int = 50
    margin_px: int = 2
    line_px: int = 2
    bold_px: int = 5
    font_size: int = 25
    handwriting: bool = True

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")

    @classmethod
    def default(cls) -> SudokuBenchmarkConfig:
        """Config matching the paper's sudoku benchmark."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict compatible with DEFAULT_SUDOKU_SETTINGS."""
        return {
            "data_name": "sudoku",
            "n": self.n,
            "n_samples": self.n_samples,
            "valid_ratio": self.valid_ratio,
            "max_corrupt": self.max_corrupt,
            "seed": self.seed,
            "temp_train_data_path": (
                data_dir / "sudoku" / "multimodal_m_21" / "tabular" / "sudoku_dataset.pkl"
            ),
            "epochs": self.epochs,
            "patience": self.patience,
            "concept_missing_mech": self.concept_missing_mech,
        }

    def get_dataset_path(self, data_type: Optional[str] = None) -> Path:
        """Return the directory path for the dataset."""
        dt = data_type or self.data_type
        filename = f"sudoku_{dt}_n{self.n}_ns{self.n_samples}_mc{self.max_corrupt}_seed{self.seed}"
        return data_dir / "sudoku" / filename

    def get_model_path(self, model_class: str, data_type: Optional[str] = None) -> Path:
        """Return the path where a trained model is saved."""
        dt = data_type or self.data_type
        filename = f"sudoku_{model_class}_{dt}_n{self.n}_mc{self.max_corrupt}"
        if self.concept_missing_mech != "none" and self.concept_missing > 0.0:
            filename += f"_cm{self.concept_missing_mech}{self.concept_missing}"
        return results_dir / f"{filename}.model"

    def get_results_path(self, model_class: str = "cbm", data_type: Optional[str] = None) -> Path:
        """Return the path where results are saved."""
        dt = data_type or self.data_type
        filename = f"sudoku_{model_class}_{dt}_n{self.n}_mc{self.max_corrupt}"
        if self.concept_missing_mech != "none" and self.concept_missing > 0.0:
            filename += f"_cm{self.concept_missing_mech}{self.concept_missing}"
        return results_dir / f"{filename}.results"

    def get_alignment_weights(self) -> Dict[str, float]:
        """Return alignment weights, computing defaults if not explicitly set.

        Default: all row/col/block concepts get weight 1.0 (AND semantics).
        """
        if self.alignment_weights is not None:
            return self.alignment_weights
        N = self.n * self.n  # 9 for standard sudoku
        weights: Dict[str, float] = {}
        for i in range(N):
            weights[f"row_valid_{i + 1}"] = 1.0
            weights[f"col_valid_{i + 1}"] = 1.0
            weights[f"block_valid_{i + 1}"] = 1.0
        weights["bias"] = -(3 * N - 0.5)
        return weights

    def get_alignment_results_path(self, data_type: Optional[str] = None) -> Path:
        """Return the path where alignment results JSON is saved."""
        dt = data_type or self.data_type
        filename = f"sudoku_alignment_{dt}_n{self.n}_mc{self.max_corrupt}"
        if self.concept_missing_mech != "none" and self.concept_missing > 0.0:
            filename += f"_cm{self.concept_missing_mech}{self.concept_missing}"
        return results_dir / f"{filename}.json"

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to YAML."""
        d = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SudokuBenchmarkConfig:
        """Load config from YAML."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)


# ── Robot Text Benchmark Config ──────────────────────────────────────

TEXT_LABEL_EXPR = (
    "'glorp' if (min(int(row[\"mouth_type\"]==\"open\"), "
    "int(str(row[\"foot_shape\"]).startswith(\"pointy_\"))) >= 1) "
    "else 'drent'"
)


@dataclass
class RobotTextBenchmarkConfig:
    """Configuration for the robot text classification benchmark."""

    seed: int = 1337
    label_expr: str = TEXT_LABEL_EXPR
    difficulty: str = "hard"

    # Data generation
    variants_per_row_minority: int = 3
    variants_per_row_majority: int = 1
    generic_enable: bool = True
    generic_rate: float = 0.7
    generic_target: str = "foot"

    # K-fold splitting
    cv_k: int = 5
    cv_fold: int = 0
    dev_per_fold: int = 1000
    deployment_size: int = 10000

    # TextConceptDetector
    detector_epochs: int = 6
    detector_batch_size: int = 64
    detector_lr: float = 2e-3
    concept_mode: str = "hard"

    # DNN baseline (DistilBERT)
    dnn_model_name: str = "distilbert-base-uncased"
    dnn_epochs: int = 3
    dnn_batch_size: int = 16
    dnn_lr: float = 5e-5

    # Intervention
    intervention_budgets: List[int] = field(default_factory=lambda: [0, 1, 2, 5, 10])
    intervention_accuracy: float = 1.0
    flip_threshold: float = 0.30
    intervention_strategy: str = "kflip"  # "kflip" (up-to-k) or "exact_k"

    # Intervention regimes
    intervention_regimes: List[str] = field(default_factory=lambda: ["baseline"])
    expert_intervention_accuracy: float = 0.80
    subjective_noise_rate: float = 0.20
    subjective_intervention_accuracy: float = 0.80
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    llm_api_key: str = ""
    llm_api_key_env: str = "GEMINI_API_KEY"
    force_retrain: bool = False  # force retrain LFCBM/subjective models

    # LFCBM (optional)
    lfcbm_enable: bool = False
    lfcbm_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    lfcbm_concepts_csv: str = ""

    # Alignment
    alignment_constraints: Optional[Dict[str, int]] = None

    _VALID_REGIMES = frozenset({"baseline", "expert", "subjective", "machine"})
    _VALID_STRATEGIES = frozenset({"kflip", "exact_k"})

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.intervention_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"intervention_strategy must be one of {sorted(self._VALID_STRATEGIES)}, "
                f"got {self.intervention_strategy!r}"
            )
        unknown = set(self.intervention_regimes) - self._VALID_REGIMES
        if unknown:
            raise ValueError(
                f"unknown intervention regimes: {sorted(unknown)}. "
                f"Valid: {sorted(self._VALID_REGIMES)}"
            )

    def get_dataset_path(self) -> Path:
        """Return the path where the dataset file is saved."""
        return results_dir / f"robot_text_seed{self.seed}.data"

    def get_model_path(self, model_class: str) -> Path:
        """Return the path where a trained model is saved."""
        return results_dir / f"robot_text_{model_class}_seed{self.seed}.model"

    def get_results_path(self, model_class: str = "cbm") -> Path:
        """Return the path where results CSV is saved."""
        return results_dir / f"robot_text_{model_class}_seed{self.seed}_results.csv"

    def get_alignment_results_path(self) -> Path:
        """Return the path where alignment results JSON is saved."""
        return results_dir / f"robot_text_alignment_seed{self.seed}.json"

    def get_alignment_constraints(self) -> Dict[str, int]:
        """Return monotonicity constraints for alignment.

        Default: mouth_is_open +1 — open mouth correlates with glorp in
        the default label rule.
        """
        if self.alignment_constraints is not None:
            return self.alignment_constraints
        return {"mouth_is_open": 1}

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to YAML."""
        d = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotTextBenchmarkConfig:
        """Load config from YAML."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
