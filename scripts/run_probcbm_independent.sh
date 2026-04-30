#!/bin/bash
# Run ProbCBM with train_class_mode=independent on all robot datasets
# for the big table (5 concept sources × 3 intervention sources)
#
# Requires: code on codex-cem-probcbm-baselines branch with independent default

set -euo pipefail

# Activate conda env (needed on DSMLP)
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate /home/jskirzynski/myenv
fi

cd "$(dirname "$0")/.."
export PYTHONPATH=.

SEED=1014

echo "============================================"
echo "ProbCBM Independent: all concept sources"
echo "============================================"

# 1. true_concepts (m=7) — needs ground_truth preset
echo ">>> [1/2] true_concepts (ground_truth preset)"
python -u scripts/robot_pipeline.py \
    --seed $SEED \
    --concept-preset ground_truth \
    --cbm-family probcbm \
    --intervention-sources perfect expert llm \
    --llm-cache-only \
    --force-retrain \
    --stages cbm intervene collect

# 2. human_concepts + machine_annotation + llm_concepts + clip_concepts (m=12)
echo ">>> [2/2] subconcepts (foot_subtypes preset, all concept sources)"
python -u scripts/robot_pipeline.py \
    --seed $SEED \
    --concept-preset foot_subtypes \
    --cbm-family probcbm \
    --concept-sources human_concepts machine_annotation llm_concepts clip_concepts \
    --intervention-sources perfect expert llm \
    --llm-cache-only \
    --force-retrain \
    --stages cbm intervene collect

echo ""
echo "============================================"
echo "DONE. Results:"
echo "============================================"
ls -lt results/*probcbm* 2>/dev/null | head -20
