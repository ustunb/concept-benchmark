"""High-level dataset generators for concept benchmarks.

Provides a unified ``DatasetGenerator`` API that wraps data creation,
splitting, and seed management into a single call:

    >>> from concept_benchmark import DatasetGenerator
    >>> dataset = DatasetGenerator("robot", seed=1014, render_images=False).generate()
    >>> dataset.training.C.shape
    (3800, 7)

    >>> dataset = DatasetGenerator("sudoku", seed=171, data_type="tabular").generate()
    >>> dataset.training.C.shape
    (600, 27)
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from concept_benchmark.config import (
    RobotBenchmarkConfig,
    SudokuBenchmarkConfig,
)
from concept_benchmark.synthetic.robot import create_synthetic_dataset
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset
from concept_benchmark.utils import create_skewed_splits_full, set_deterministic_seed

if TYPE_CHECKING:
    from concept_benchmark.data import ConceptDataset

__all__ = ["DatasetGenerator"]


# ── Standalone generation functions ──────────────────────────────────


def generate_robot_dataset(config: RobotBenchmarkConfig) -> ConceptDataset:
    """Generate a robot dataset from a config, with train/val/test splits."""
    set_deterministic_seed(config.seed)
    settings = config.to_dict()
    data = create_synthetic_dataset(**settings)
    data.generate_cvindices(seed=config.seed)
    rng = np.random.default_rng(config.seed)
    return create_skewed_splits_full(dataset=data, rng=rng, **settings)


def generate_sudoku_dataset(config: SudokuBenchmarkConfig) -> ConceptDataset:
    """Generate a sudoku dataset from a config, with train/val/test splits."""
    from functools import partial

    set_deterministic_seed(config.seed)

    data_type = config.data_type
    kwargs = {}
    if data_type == "image" and config.render_images:
        from concept_benchmark.synthetic.sudoku import image_transform

        kwargs["transform"] = partial(
            image_transform,
            cell_px=config.cell_px,
            margin_px=config.cell_margin_px,
            line_px=config.gridline_px,
            bold_px=config.block_border_px,
            font_size=config.font_size,
            handwriting=config.font_style == "handwritten",
        )
    elif data_type == "image" and not config.render_images:
        # Image mode but skip rendering — use tabular transform internally,
        # metadata still says "image".
        pass

    data = create_sudoku_dataset(
        n=config.block_size,
        n_samples=config.n_boards,
        valid_ratio=config.valid_board_ratio,
        max_corrupt=config.max_cell_swaps,
        seed=config.seed,
        data_type=data_type if config.render_images else "tabular",
        **kwargs,
    )
    data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=config.seed)
    data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)
    return data


def generate_robot_text_dataset(config: RobotBenchmarkConfig) -> ConceptDataset:
    """Generate a robot text dataset from a config, with train/val/test splits.

    Uses the robot text catalog to enumerate concept combinations, generates
    text descriptions from corpus templates, and splits by robot identity.
    Applies ``excluded_concepts``, ``missing_fraction``, and
    ``sampling_constraints`` using the same mechanisms as the image pipeline.
    """
    from concept_benchmark.synthetic.robot_text.catalog import (
        compute_label,
        enumerate_robot_concepts,
    )
    from concept_benchmark.synthetic.robot_text.corpus import (
        compute_text_concept_names,
        get_corpus_path,
    )
    from concept_benchmark.synthetic.robot_text.dataset import (
        build_text_dataset,
        kfold_by_robot_identity,
    )
    from concept_benchmark.synthetic.helper.robot_catalog import collapse_robot_subtypes
    from concept_benchmark.synthetic.helper.utils import build_model_expression

    from concept_benchmark.utils import set_deterministic_seed

    set_deterministic_seed(config.seed)

    # 1. Compute concept names from config
    all_concept_names = compute_text_concept_names(
        config.concepts, config.fine_grained_concepts
    )

    # Enumerate all robot concept combinations
    catalog_df = enumerate_robot_concepts(concepts=config.concepts, seed=config.seed)

    # Collapse subtypes for label computation only
    catalog_for_labels = catalog_df.copy()
    collapse_robot_subtypes(
        catalog_for_labels, robot_features=list(config.concepts.keys())
    )

    # Build expression from structured params (same as image pipeline)
    expr = build_model_expression(
        config.label_features,
        model_type="deterministic",
        weights=config.label_weights,
        intercept=config.label_intercept,
    )

    model_type = "stochastic" if config.use_stochastic_labels else "deterministic"
    catalog_df["label"] = compute_label(
        catalog_for_labels,
        expr,
        label_model_type=model_type,
        alpha=config.label_temperature if model_type == "stochastic" else 1.0,
        seed=config.seed,
    )

    # 2. Build dataset with uniform renders_per_robot variants
    corpus_path = get_corpus_path(config)
    ds = build_text_dataset(
        catalog_df=catalog_df,
        corpus_path=corpus_path,
        variants_per_row=config.renders_per_robot,
        seed=config.seed,
        concept_names=all_concept_names,
    )

    # 3. Split by robot identity (hardcoded sensible defaults)
    ds = kfold_by_robot_identity(
        ds,
        cv_k=5,
        cv_fold=0,
        dev_per_fold=1000,
        deployment_size=config.test_size,
        seed=config.seed,
        corpus_path=corpus_path,
        catalog_df=catalog_df,
        concept_names=all_concept_names,
    )

    # 4. Apply excluded_concepts — filter C matrix columns on all splits
    if config.excluded_concepts:
        _drop = set(config.excluded_concepts)
        keep_idx = [i for i, n in enumerate(all_concept_names) if n not in _drop]
        kept_names = tuple(all_concept_names[i] for i in keep_idx)
        for split in (ds, ds.training, ds.validation, ds.test):
            if split is not None and hasattr(split, "C"):
                split.C = split.C[:, keep_idx]
                new_meta = dict(split.meta)
                new_meta["concepts"] = kept_names
                split.meta = new_meta

    # 5. Apply missing_fraction / missing_mechanism on training split
    if config.missing_fraction > 0 and ds.training is not None:
        from concept_benchmark.helper.data_utils import (
            sample_mcar_mask,
            sample_mnar_mask,
        )

        rng = np.random.default_rng(config.seed + 999)
        if config.missing_mechanism == "mcar":
            mask = sample_mcar_mask(rng, ds.training.C.shape, config.missing_fraction)
        else:
            mask = sample_mnar_mask(
                rng,
                ds.training.C,
                base_p=config.missing_fraction,
                config=None,
            )
        ds.training.set_concept_missing_mask(mask)
        ds.training.has_concept_missing = True

    return ds


# ── Benchmark registry ───────────────────────────────────────────────

_BENCHMARKS = MappingProxyType(
    {
        "robot": (RobotBenchmarkConfig, generate_robot_dataset),
        "sudoku": (SudokuBenchmarkConfig, generate_sudoku_dataset),
    }
)


# ── Unified DatasetGenerator ─────────────────────────────────────────


class DatasetGenerator:
    """Unified generator for all concept benchmarks.

    Follows the HuggingFace ``load_dataset`` pattern — the first argument
    selects the benchmark, remaining kwargs configure it:

        >>> DatasetGenerator("robot", seed=1014, render_images=False).generate()
        >>> DatasetGenerator("sudoku", seed=171, data_type="tabular").generate()

    Parameters
    ----------
    benchmark : str
        Benchmark name: ``"robot"`` or ``"sudoku"``.
    **kwargs
        Benchmark-specific configuration parameters. Passed directly to the
        underlying config dataclass (``RobotBenchmarkConfig`` or
        ``SudokuBenchmarkConfig``). Unknown parameters raise ``ValueError``
        with a list of valid parameters.
    """

    @overload
    def __init__(
        self,
        benchmark: Literal["robot"],
        *,
        # ── Common (image + text) ──
        seed: int = ...,
        data_type: str = ...,
        concepts: dict[str, list] | None = ...,
        label_formula: dict | None = ...,
        use_stochastic_labels: bool = ...,
        train_size: int = ...,
        test_size: int = ...,
        missing_fraction: float = ...,
        missing_mechanism: str = ...,
        concept_preset: str = ...,
        fine_grained_concepts: list[str] | None = ...,
        sampling_constraints: list[dict] | None = ...,
        excluded_concepts: list[str] | None = ...,
        renders_per_robot: int = ...,
        # ── Image-only ──
        render_images: bool = ...,
        image_size: str = ...,
        color_mode: str = ...,
        # ── Text-only ──
        template_complexity: str = ...,
        # ── Additional config fields (training, intervention, etc.) ──
        **kwargs,
    ) -> None: ...

    @overload
    def __init__(
        self,
        benchmark: Literal["sudoku"],
        *,
        seed: int = ...,
        data_type: str = ...,
        render_images: bool = ...,
        block_size: int = ...,
        n_boards: int = ...,
        valid_board_ratio: float = ...,
        max_cell_swaps: int = ...,
        missing_fraction: float = ...,
        missing_mechanism: str = ...,
        # OCR rendering (image only)
        cell_px: int = ...,
        cell_margin_px: int = ...,
        gridline_px: int = ...,
        block_border_px: int = ...,
        font_size: int = ...,
        font_style: str = ...,
        # ── Additional config fields (training, intervention, etc.) ──
        **kwargs,
    ) -> None: ...

    def __init__(self, benchmark: str, **kwargs):
        if benchmark in _BENCHMARKS:
            config_class, self._generate_fn = _BENCHMARKS[benchmark]
        else:
            raise ValueError(
                f"Unknown benchmark {benchmark!r}. Available: {sorted(_BENCHMARKS)}"
            )

        # Text modality routes to text generator
        if benchmark == "robot" and kwargs.get("data_type") == "text":
            self._generate_fn = generate_robot_text_dataset

        self.benchmark = benchmark
        try:
            self.config = config_class(**kwargs)
        except TypeError as e:
            valid = [f.name for f in dataclasses.fields(config_class)]
            raise ValueError(
                f"Invalid parameter for benchmark {benchmark!r}: {e}. "
                f"Available parameters: {', '.join(valid)}"
            ) from None

    @classmethod
    def from_config(
        cls, config: RobotBenchmarkConfig | SudokuBenchmarkConfig
    ) -> "DatasetGenerator":
        """Create a generator from an existing config object.

        This is the preferred entry point for pipelines and scripts that
        already have a config (e.g. from CLI parsing or YAML):

            >>> cfg = RobotBenchmarkConfig(seed=1014, concept_preset="foot_subtypes")
            >>> dataset = DatasetGenerator.from_config(cfg).generate()
        """
        for name, (config_class, _) in _BENCHMARKS.items():
            if isinstance(config, config_class):
                obj = cls.__new__(cls)
                obj.benchmark = name
                obj.config = config
                _, obj._generate_fn = _BENCHMARKS[name]
                # Text modality routes to text generator
                if name == "robot" and getattr(config, "data_type", None) == "text":
                    obj._generate_fn = generate_robot_text_dataset
                return obj
        raise TypeError(
            f"Unsupported config type {type(config).__name__!r}. "
            f"Expected one of: {', '.join(c.__name__ for c, _ in _BENCHMARKS.values())}"
        )

    def generate(self) -> ConceptDataset:
        """Generate the dataset with train/val/test splits."""
        return self._generate_fn(self.config)

    @classmethod
    def available_benchmarks(cls) -> list[str]:
        """Return sorted list of registered benchmark names."""
        return sorted(_BENCHMARKS)
