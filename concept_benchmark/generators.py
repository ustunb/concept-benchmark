"""High-level dataset generators for concept benchmarks.

Provides a unified ``DatasetGenerator`` API that wraps data creation
and seed management into a single call.  After ``generate()``, call
``sample()`` to split into train/val/test:

    >>> from concept_benchmark.robot import DatasetGenerator
    >>> from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS
    >>> gen = DatasetGenerator(seed=1014, render_images=False)
    >>> dataset = gen.generate()
    >>> dataset.drop_concepts(PRESET_EXCLUDED_CONCEPTS["ground_truth"])
    >>> dataset.sample(test_size=10000, val_size=0.2, train_size=3800, seed=1014)
    >>> dataset.train.C.shape
    (3800, 7)

    >>> from concept_benchmark.sudoku import DatasetGenerator
    >>> dataset = DatasetGenerator(seed=171, data_type="tabular").generate()
    >>> dataset.sample(test_size=0.2, val_size=0.2, stratify=dataset.y, seed=171)
    >>> dataset.train.C.shape
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
from concept_benchmark.utils import set_deterministic_seed

if TYPE_CHECKING:
    from concept_benchmark.data import ConceptDataset

__all__ = ["DatasetGenerator"]


# ── Standalone generation functions ──────────────────────────────────


def generate_robot_dataset(config: RobotBenchmarkConfig) -> ConceptDataset:
    """Generate a robot dataset from a config.

    Returns an unsplit dataset. Call ``data.sample(...)`` to create
    train/val/test splits.
    """
    set_deterministic_seed(config.seed)
    settings = config.to_dict()
    data = create_synthetic_dataset(**settings)
    return data


def generate_sudoku_dataset(config: SudokuBenchmarkConfig) -> ConceptDataset:
    """Generate a sudoku dataset from a config.

    Returns an unsplit dataset. Call ``data.sample(...)`` to create
    train/val/test splits.
    """
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
    return data


def generate_robot_text_dataset(config: RobotBenchmarkConfig) -> ConceptDataset:
    """Generate a robot text dataset from a config.

    Returns an unsplit dataset. Call ``data.sample(...)`` to create
    train/val/test splits. The ``row_index`` array stored in
    ``meta["row_index"]`` can be passed as ``groups=`` to ``sample()``
    to prevent template leakage across splits.
    """
    from concept_benchmark.data import ConceptDataset
    from concept_benchmark.synthetic.robot_text.catalog import (
        compute_label,
        enumerate_robot_concepts,
    )
    from concept_benchmark.synthetic.robot_text.corpus import (
        compute_text_concept_names,
        get_corpus_path,
    )
    from concept_benchmark.synthetic.robot_text.dataset import build_text_dataset
    from concept_benchmark.synthetic.helper.robot_catalog import collapse_robot_subtypes

    set_deterministic_seed(config.seed)

    # 1. Compute concept names from config
    all_concept_names = compute_text_concept_names(
        config.concepts, config.expand_concepts
    )

    # Enumerate all robot concept combinations
    catalog_df = enumerate_robot_concepts(concepts=config.concepts, seed=config.seed)

    # Collapse subtypes for label computation only
    catalog_for_labels = catalog_df.copy()
    collapse_robot_subtypes(
        catalog_for_labels, robot_features=list(config.concepts.keys())
    )

    catalog_df["label"] = compute_label(
        catalog_for_labels,
        config.label_formula,
        stochastic=config.use_stochastic_labels,
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

    # 3. Wrap in ConceptDataset (unsplit — caller uses sample())
    row_index = getattr(ds, "_row_index", np.arange(len(ds.y)))
    meta = dict(ds.meta)
    meta["row_index"] = row_index
    data = ConceptDataset(
        X=np.array(ds.X, dtype=object),
        C=ds.C,
        y=ds.y,
        meta=meta,
    )

    return data


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

        >>> from concept_benchmark.robot import DatasetGenerator
        >>> DatasetGenerator(seed=1014, render_images=False).generate()
        >>> from concept_benchmark.sudoku import DatasetGenerator
        >>> DatasetGenerator(seed=171, data_type="tabular").generate()

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
        concept_preset: str = ...,
        expand_concepts: list[str] | None = ...,
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
                f"Invalid parameter for benchmark {benchmark!r}: {e!r}. "
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
        """Generate the dataset.

        Returns raw data with all concepts. Apply layers explicitly::

            from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS
            data = gen.generate()
            data.drop_concepts(PRESET_EXCLUDED_CONCEPTS["ground_truth"])
            data.sample(test_size=0.2, val_size=0.2, seed=42)
            data.sample_concept_missingness(p=0.2, rng=99)
        """
        return self._generate_fn(self.config)

    @classmethod
    def available_benchmarks(cls) -> list[str]:
        """Return sorted list of registered benchmark names."""
        return sorted(_BENCHMARKS)
