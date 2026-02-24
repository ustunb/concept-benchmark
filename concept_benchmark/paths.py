"""Directory paths used throughout the package."""
from __future__ import annotations

__all__ = [
    "pkg_dir",
    "data_dir",
    "results_dir",
]

from pathlib import Path

# path to the GitHub repository root
repo_dir = Path(__file__).resolve().parent.parent

# path to the Python package
pkg_dir = repo_dir / "concept_benchmark/"

# directory where we store datasets
data_dir = repo_dir / "data/"

# directory where we store results
results_dir = repo_dir / "results/"
