#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
CEM_DIR="${THIRD_PARTY_DIR}/cem"
CEM_REPO_URL="https://github.com/mateoespinosa/cem.git"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${CEM_DIR}/.git" ]]; then
  echo "[install_cem_repo] cloning official cem repo into ${CEM_DIR}"
  git clone "${CEM_REPO_URL}" "${CEM_DIR}"
else
  echo "[install_cem_repo] updating existing cem checkout in ${CEM_DIR}"
  git -C "${CEM_DIR}" pull --ff-only
fi

echo "[install_cem_repo] installing pytorch-lightning compatibility deps"
python -m pip install "pytorch-lightning>=1.6,<2.0" "torchmetrics<1.0"

echo "[install_cem_repo] installing official cem requirements"
python -m pip install -r "${CEM_DIR}/requirements.txt"

echo "[install_cem_repo] installing official cem package in editable mode"
python -m pip install -e "${CEM_DIR}"

echo "[install_cem_repo] done"
