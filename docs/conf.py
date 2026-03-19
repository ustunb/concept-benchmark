# Configuration file for the Sphinx documentation builder.

project = "concept-benchmark"
copyright = "2025, Skirzynski et al."
author = "Julian Skirzynski, Harry Cheon, Shreyas Kadekodi, Meredith Stewart, Berk Ustun"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_class = True

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "exclude-members": "__check_rep__, __copy__, __eq__, __len__, __repr__",
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Mock imports (heavy C extensions) ---------------------------------------
autodoc_mock_imports = ["cairo", "pero"]

# -- Theme --------------------------------------------------------------------
html_theme = "furo"
html_title = "concept-benchmark"
html_logo = "assets/logo.svg"
html_static_path = []

# The README (included via quickstart.rst) references images as
# ``docs/assets/foo.png`` (relative to the repo root, for GitHub rendering).
# When Sphinx builds from ``docs/``, those paths resolve to
# ``<build>/docs/assets/`` which doesn't exist.  Copy the assets directory
# into the output under ``docs/assets/`` so the links work in both contexts.
import shutil
from pathlib import Path


def _copy_readme_assets(app, exception):
    """Post-build hook: copy docs/assets/ → _build/html/docs/assets/."""
    if exception:
        return
    src = Path(__file__).parent / "assets"
    dst = Path(app.outdir) / "docs" / "assets"
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def setup(app):
    app.connect("build-finished", _copy_readme_assets)

# -- Myst settings -----------------------------------------------------------
suppress_warnings = ["myst.xref_missing"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
