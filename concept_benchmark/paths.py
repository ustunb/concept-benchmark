"""Directory paths used throughout the package."""
from __future__ import annotations

__all__ = [
    "pkg_dir",
    "data_dir",
    "results_dir",
]

from pathlib import Path

# path to the Python package
pkg_dir = Path(__file__).resolve().parent

# Detect repo root: works for editable installs and running from source.
# For non-editable installs (site-packages), fall back to CWD so that
# users who run from the cloned repo still find data/ and results/.
_repo_dir = pkg_dir.parent
if (_repo_dir / "pyproject.toml").is_file():
    data_dir = _repo_dir / "data"
    results_dir = _repo_dir / "results"
else:
    data_dir = Path.cwd() / "data"
    results_dir = Path.cwd() / "results"
