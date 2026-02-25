#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/venv"
PYTHON="${PYTHON:-python3}"

# ── Progress helpers ─────────────────────────────────────────────────
TOTAL_STEPS=5
CURRENT_STEP=0

step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo ""
    echo "[${CURRENT_STEP}/${TOTAL_STEPS}] $1"
}

# ── 1. System dependencies (cairo for pycairo) ──────────────────────
step "Checking system dependencies ..."

install_system_deps() {
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            echo "  Installing cairo via Homebrew ..."
            brew install cairo pkg-config 2>/dev/null || true
        else
            echo "ERROR: Homebrew not found. Install it from https://brew.sh then re-run."
            echo "  pycairo requires the cairo C library: brew install cairo pkg-config"
            exit 1
        fi
    elif [[ -f /etc/debian_version ]]; then
        echo "  Installing cairo via apt ..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq libcairo2-dev pkg-config python3-dev
    elif [[ -f /etc/redhat-release ]]; then
        echo "  Installing cairo via dnf ..."
        sudo dnf install -y cairo-devel pkg-config python3-devel
    else
        echo "WARNING: Unrecognized OS. pycairo requires the cairo C library."
        echo "  Install it manually (e.g. libcairo2-dev on Debian, cairo-devel on Fedora)"
        echo "  then re-run this script."
    fi
}

if pkg-config --exists cairo 2>/dev/null; then
    echo "  cairo found"
else
    install_system_deps
fi

# ── 2. Virtual environment ───────────────────────────────────────────
step "Setting up virtual environment ..."

if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
    echo "  Created ${VENV_DIR}"
else
    echo "  Already exists at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "  Python: $(python --version)"

# ── 3. Upgrade pip ───────────────────────────────────────────────────
step "Upgrading pip ..."
python -m pip install --upgrade pip --quiet

# ── 4. Install package + dependencies ────────────────────────────────
step "Installing concept-benchmark (this may take a few minutes) ..."
pip install -e "${REPO_DIR}" --quiet
echo "  Done"

# ── 5. Dev dependencies ─────────────────────────────────────────────
step "Installing dev tools (pytest, ruff, jupyter) ..."
pip install pytest ruff jupyter ipykernel rich --quiet
echo "  Done"

echo ""
echo "=== Installation complete ==="
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To verify:"
echo "  python3 -c \"import concept_benchmark; print('concept-benchmark installed successfully')\""
