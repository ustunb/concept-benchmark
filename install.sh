#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/venv"
PYTHON="${PYTHON:-python3}"

echo "=== Concept Benchmark Installer ==="
echo "Repository: ${REPO_DIR}"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR} ..."
    "${PYTHON}" -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

# 2. Activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Using python: $(which python) ($(python --version))"

# 3. Upgrade pip
python -m pip install --upgrade pip --quiet

# 4. Install the package in editable mode with all dependencies
echo "Installing concept-benchmark and dependencies ..."
pip install -e "${REPO_DIR}" --quiet

# 5. Install dev dependencies
echo "Installing dev dependencies ..."
pip install pytest ruff jupyter ipykernel rich --quiet

echo ""
echo "=== Installation complete ==="
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To verify:"
echo "  python -c \"import concept_benchmark; print('concept-benchmark installed successfully')\""
