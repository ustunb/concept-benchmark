#!/bin/bash
# Exp 4: Concept Source — all 4 CBM families × 6 regimes
# Subconcept setup, exactly_k strategy, budgets 1 2 5
#
# Priority order:
#   1. LFCBM regimes (machine/llm/clip) for CBM — these were stuck on CPU before
#   2. CEM/ProbCBM/ECBM × all 6 regimes
#
# Pre-requisites on DSMLP:
#   - Data generated (robot images exist)
#   - All 4 family models trained (cbm/cem/probcbm/ecbm .model files)
#   - LFCBM models trained (machine/llm/clip .model files)
#   - LLM intervention cache (10000 entries)
#   - CLIP embedding caches (lfcbm_cache, lfcbm_llm_cache, lfcbm_clip_cache)
#
# What this script does:
#   - Skips setup/cbm/dnn stages (already done)
#   - Runs only intervene + collect for each family × regime combo

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

COMMON="--seed 1014 --concept-preset foot_subtypes --strategy exactly_k --budgets 1 2 5 --llm-cache-only"
STAGES="--stages intervene collect"

echo "============================================"
echo "Exp 4: Concept Source (all families × all regimes)"
echo "============================================"
echo ""

# --- Priority 1: CBM family, LFCBM regimes (were stuck on CPU) ---
echo ">>> [1/4] CBM × llm,clip regimes (previously stuck)"
python scripts/robot_pipeline.py $COMMON \
    --cbm-family cbm \
    --regimes llm clip \
    $STAGES
echo ">>> CBM × llm,clip DONE"

# --- Priority 2: CEM × all 6 regimes ---
echo ">>> [2/4] CEM × all regimes"
python scripts/robot_pipeline.py $COMMON \
    --cbm-family cem \
    --regimes baseline expert subjective machine llm clip \
    $STAGES
echo ">>> CEM DONE"

# --- Priority 3: ProbCBM × all 6 regimes ---
echo ">>> [3/4] ProbCBM × all regimes"
python scripts/robot_pipeline.py $COMMON \
    --cbm-family probcbm \
    --regimes baseline expert subjective machine llm clip \
    $STAGES
echo ">>> ProbCBM DONE"

# --- Priority 4: ECBM × all 6 regimes ---
echo ">>> [4/4] ECBM × all regimes"
python scripts/robot_pipeline.py $COMMON \
    --cbm-family ecbm \
    --regimes baseline expert subjective machine llm clip \
    $STAGES
echo ">>> ECBM DONE"

echo ""
echo "============================================"
echo "ALL DONE. Results in results/"
echo "============================================"
ls -lt results/*regime* results/*subconcept*results* 2>/dev/null | head -20
