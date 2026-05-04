#!/usr/bin/env bash
# Install the official mateoespinosa/cem package which provides three
# additional CBM families: CEM, ProbCBM, and ECBM.
# After running this script you can use:
#   --cbm-family cem
#   --cbm-family probcbm
#   --cbm-family ecbm
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
CEM_DIR="${THIRD_PARTY_DIR}/cem"
CEM_REPO_URL="https://github.com/mateoespinosa/cem.git"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${CEM_DIR}/.git" ]]; then
  echo "[install] cloning official cem repo into ${CEM_DIR}"
  git clone "${CEM_REPO_URL}" "${CEM_DIR}"
else
  echo "[install] updating existing cem checkout in ${CEM_DIR}"
  git -C "${CEM_DIR}" pull --ff-only
fi

echo "[install] installing pytorch-lightning compatibility deps"
python -m pip install "pytorch-lightning>=1.6,<2.0" "torchmetrics<1.0"

echo "[install] installing official cem requirements"
python -m pip install -r "${CEM_DIR}/requirements.txt"

echo "[install] installing official cem package in editable mode"
python -m pip install -e "${CEM_DIR}"

echo "[install] done — CEM, ProbCBM, and ECBM are now available"
