#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/venv"
PYTHON="${PYTHON:-python3}"

echo "=== Concept Benchmark Installer ==="
echo "Repository: ${REPO_DIR}"

# ── 1. Install system dependencies needed to compile pycairo ──────────
install_system_deps() {
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            echo "Installing system dependencies via Homebrew ..."
            brew install cairo pkg-config 2>/dev/null || true
        else
            echo "ERROR: Homebrew not found. Install it from https://brew.sh then re-run."
            echo "  pycairo requires the cairo C library: brew install cairo pkg-config"
            exit 1
        fi
    elif [[ -f /etc/debian_version ]]; then
        echo "Installing system dependencies via apt ..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq libcairo2-dev pkg-config python3-dev
    elif [[ -f /etc/redhat-release ]]; then
        echo "Installing system dependencies via dnf ..."
        sudo dnf install -y cairo-devel pkg-config python3-devel
    else
        echo "WARNING: Unrecognized OS. pycairo requires the cairo C library."
        echo "  Install it manually (e.g. libcairo2-dev on Debian, cairo-devel on Fedora)"
        echo "  then re-run this script."
    fi
}

# Check if cairo headers are already available
if ! pkg-config --exists cairo 2>/dev/null; then
    install_system_deps
fi

# ── 2. Create virtual environment if it doesn't exist ─────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR} ..."
    "${PYTHON}" -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

# ── 3. Activate ──────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Using python: $(which python) ($(python --version))"

# ── 4. Upgrade pip ───────────────────────────────────────────────────
python -m pip install --upgrade pip --quiet

# ── 5. Install the package in editable mode with all dependencies ────
echo "Installing concept-benchmark and dependencies ..."
pip install -e "${REPO_DIR}" --quiet

# ── 6. Install dev dependencies ──────────────────────────────────────
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
