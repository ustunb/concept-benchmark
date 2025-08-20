"""
This file contains paths to key directories we'll use in the project
"""
from pathlib import Path

# path to the GitHub repository
repo_dir = Path(__file__).resolve().parent

# path to the Python package
pkg_dir = repo_dir

# directory for static elements used in flask
static_dir = pkg_dir / 'static'

# directory where we keep image files
image_dir = static_dir / 'images'

# directory where we store results
results_dir = repo_dir / 'results'

# directory where we store reporting code
reporting_dir = repo_dir / 'reporting'

# directory where we store RMarkdown templates
report_templates = reporting_dir / 'templates'