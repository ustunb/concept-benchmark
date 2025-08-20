# tests/conftest.py
from __future__ import annotations

from typing import Callable, Dict, Tuple, List

import numpy as np
import pytest

from pathlib import Path
from PIL import Image

from concept_benchmark.data import ConceptDataset


@pytest.fixture(scope="session", autouse=True)
def rng_seed() -> int:
    """Global RNG seed for deterministic tests."""
    np.random.seed(1337)
    return 1337


def make_tabular_arrays(
    n: int,
    d: int,
    k: int,
    n_classes: int,
    *,
    class_balance: str = "balanced",
    concept_density: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Create X, C, y, meta for tabular datasets.

    Args:
      n: Number of samples.
      d: Number of feature columns.
      k: Number of concept columns.
      n_classes: Number of classes.
      class_balance: "balanced" or "random".
      concept_density: Bernoulli p for concepts in (0, 1).

    Returns:
      Tuple of (X, C, y, meta).
    """
    X = np.random.normal(size=(n, d)).astype(np.float32)
    C = (np.random.rand(n, k) < concept_density).astype(np.int8)

    if class_balance == "balanced":
      reps = int(np.ceil(n / n_classes))
      y = np.arange(n_classes, dtype=np.int32).repeat(reps)[:n]
      rng = np.random.default_rng(202)
      rng.shuffle(y)
      y = y.astype(np.int32)
    else:
      y = np.random.randint(0, n_classes, size=n).astype(np.int32)

    meta = {
        "classes": [f"c{i}" for i in range(n_classes)],
        "concepts": [f"z{i}" for i in range(k)],
        "data_type": "tabular",
    }
    return X, C, y, meta


def make_cvindices(
    n: int,
    *,
    K: int = 5,
    replicate: int = 1,
) -> Dict[str, np.ndarray]:
    """Create a simple K-fold index vector inside a cvindices dict.

    Args:
      n: Number of samples.
      K: Number of folds.
      replicate: Replicate number (1-based).
      fold_id_prefix: Prefix to form fold_id.

    Returns:
      Dict mapping fold_id -> folds array in {1..K}.
    """
    folds = (np.arange(n) % K) + 1
    fold_id_prefix = f"K{K:02d}N"
    fid = f"{fold_id_prefix}{replicate:02d}"
    return {fid: folds.astype(np.int32)}


def make_tabular_dataset(
    n: int,
    d: int,
    k: int,
    n_classes: int,
    *,
    transform_x: Callable | None = None,
    transform_c: Callable | None = None,
    transform_y: Callable | None = None,
    with_cv: bool = False,
    K: int = 5,
) -> Tuple[ConceptDataset, Dict[str, np.ndarray]]:
    """Build a ConceptDataset for tabular data.

    Args:
      n: Samples.
      d: Feature dim.
      k: Concept dim.
      n_classes: Number of classes.
      transform_x: Optional feature transform.
      transform_c: Optional concept transform.
      transform_y: Optional label transform.
      with_cv: Whether to attach cvindices.
      K: Number of folds if with_cv.

    Returns:
      (dataset, cvindices_dict). cvindices_dict may be {} if with_cv=False.
    """
    X, C, y, meta = make_tabular_arrays(
        n=n,
        d=d,
        k=k,
        n_classes=n_classes,
    )
    cv = make_cvindices(n=n, K=K) if with_cv else {}
    ds = ConceptDataset(
        X=X,
        C=C,
        y=y,
        meta=meta,
        transform_x=transform_x,
        transform_c=transform_c,
        transform_y=transform_y,
        cvindices=cv or None,
    )
    return ds, cv


# -----------------------------
# Image dataset factories
# -----------------------------

def _write_synthetic_images(
    root: Path,
    n: int,
    *,
    size: Tuple[int, int] = (16, 16),
    mode: str = "L",
) -> List[str]:
    """Create `n` deterministic synthetic images on disk and return their paths.

    Args:
        root: Directory to write images into.
        n: Number of images to create.
        size: Image size (width, height).
        mode: PIL mode, e.g., "L" or "RGB".

    Returns:
        List of file paths as strings.
    """
    root.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    w, h = size
    for i in range(n):
        # Deterministic pixel pattern per index
        arr = (np.arange(w * h).reshape(h, w) + (i * 7)) % 256
        if mode == "RGB":
            # Stack into 3 channels
            img_arr = np.stack([arr, np.roll(arr, 1, axis=0), np.roll(arr, 1, axis=1)], axis=-1).astype(np.uint8)
        else:
            img_arr = arr.astype(np.uint8)
        img = Image.fromarray(img_arr)
        p = root / f"img_{i:03d}.png"
        img.save(p)
        paths.append(str(p))
    return paths

def make_image_arrays(
    n: int,
    k: int,
    n_classes: int,
    *,
    concept_density: float = 0.3,
    class_balance: str = "balanced",
    size: Tuple[int, int] = (16, 16),
    mode: str = "L",
    root: Path,
    add_missing: bool = False,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
    """Create image paths X, concepts C, labels y, and meta for image datasets.

    Args:
        n: Number of samples.
        k: Number of concept columns.
        n_classes: Number of classes.
        concept_density: Bernoulli p for concepts.
        class_balance: "balanced" or "random" class distribution.
        size: Image width and height.
        mode: PIL mode ("L" or "RGB").
        root: Directory where images are written.
        add_missing: If True, append one nonexistent path to test error branches.

    Returns:
        (X_paths, C, y, meta).
    """
    X_paths = _write_synthetic_images(root=root, n=n, size=size, mode=mode)
    if add_missing:
        X_paths.append(str(root / "does_not_exist.png"))
    
    X_paths = np.array(X_paths, dtype=object)  # Use object dtype for string paths

    C = (np.random.rand(n, k) < concept_density).astype(np.int8)

    if class_balance == "balanced":
        reps = int(np.ceil(n / n_classes))
        y = np.arange(n_classes, dtype=np.int32).repeat(reps)[:n]
        rng = np.random.default_rng(404)
        rng.shuffle(y)
        y = y.astype(np.int32)
    else:
        y = np.random.randint(0, n_classes, size=n).astype(np.int32)

    # If we appended a missing path, make C and y match X length by duplicating the first row/label
    if add_missing:
        C = np.vstack([C, C[:1].copy()])
        y = np.concatenate([y, y[:1].copy()])

    meta = {
        "classes": [f"c{i}" for i in range(n_classes)],
        "concepts": [f"z{i}" for i in range(k)],
        "data_type": "image",
        "image": {"mode": mode, "size": list(size)},
    }
    return X_paths, C, y, meta

def make_image_dataset(
    n: int,
    k: int,
    n_classes: int,
    *,
    root: Path,
    size: Tuple[int, int] = (16, 16),
    mode: str = "L",
    with_cv: bool = False,
    K: int = 5,
    add_missing: bool = False,
) -> Tuple[ConceptDataset, Dict[str, np.ndarray]]:
    """Build a `ConceptDataset` configured for image samples.

    Args:
        n: Samples.
        k: Concept dim.
        n_classes: Number of classes.
        root: Directory for image files.
        size: Image width and height.
        mode: PIL mode.
        with_cv: Whether to attach cvindices.
        K: Number of folds if with_cv.
        add_missing: Append one nonexistent path for error-path tests.

    Returns:
        (dataset, cvindices_dict).
    """
    X_paths, C, y, meta = make_image_arrays(
        n=n,
        k=k,
        n_classes=n_classes,
        size=size,
        mode=mode,
        root=root,
        add_missing=add_missing,
    )
    cv = make_cvindices(n=n, K=K) if with_cv else {}
    ds = ConceptDataset(
        X=X_paths,
        C=C,
        y=y,
        meta=meta,
        cvindices=cv or None,
    )
    return ds, cv


# Reusable pytest fixtures

@pytest.fixture
def tab_small() -> ConceptDataset:
    """Tiny balanced dataset: n=12, d=3, k=4, classes=2."""
    ds, _ = make_tabular_dataset(n=12, d=3, k=4, n_classes=2, with_cv=False)
    return ds


@pytest.fixture
def tab_small_cv() -> Tuple[ConceptDataset, str]:
    """Tiny dataset with CV indices and a valid fold_id."""
    ds, cv = make_tabular_dataset(n=12, d=3, k=4, n_classes=2, with_cv=True, K=3)
    fold_id = next(iter(cv.keys()))
    return ds, fold_id


@pytest.fixture
def tab_medium_cv() -> Tuple[ConceptDataset, str]:
    """Medium dataset with CV for split tests."""
    ds, cv = make_tabular_dataset(n=60, d=8, k=6, n_classes=3, with_cv=True, K=5)
    fold_id = next(iter(cv.keys()))
    return ds, fold_id


@pytest.fixture
def img_small(tmp_path: Path) -> ConceptDataset:
    """Tiny image dataset: n=6, k=3, classes=2, grayscale."""
    ds, _ = make_image_dataset(
        n=6,
        k=3,
        n_classes=2,
        root=tmp_path / "imgs_small",
        size=(16, 16),
        mode="L",
        with_cv=False,
    )
    return ds

@pytest.fixture
def img_small_cv(tmp_path: Path) -> Tuple[ConceptDataset, str]:
    """Tiny image dataset with CV indices and a valid fold_id."""
    ds, cv = make_image_dataset(
        n=12,
        k=4,
        n_classes=3,
        root=tmp_path / "imgs_small_cv",
        size=(16, 16),
        mode="RGB",
        with_cv=True,
        K=3,
    )
    fold_id = next(iter(cv.keys()))
    return ds, fold_id

@pytest.fixture
def img_with_missing(tmp_path: Path) -> ConceptDataset:
    """Image dataset that includes one nonexistent path to hit error branches."""
    ds, _ = make_image_dataset(
        n=5,
        k=2,
        n_classes=2,
        root=tmp_path / "imgs_missing",
        add_missing=True,
    )
    return ds