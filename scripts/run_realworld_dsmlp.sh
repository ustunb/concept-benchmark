#!/usr/bin/env bash
#
# Run real-world automation experiments (pistachio + rice) on DSMLP with ViT backbone.
#
# Usage:
#   ./scripts/run_realworld_dsmlp.sh                    # run both datasets
#   ./scripts/run_realworld_dsmlp.sh pistachio          # run pistachio only
#   ./scripts/run_realworld_dsmlp.sh rice               # run rice only
#
# Monitor progress from another terminal:
#   ./scripts/monitor_dsmlp.sh
#
# Fetch results after completion:
#   ./scripts/fetch_dsmlp_results.sh

set -euo pipefail

DATASET="${1:-both}"
DSMLP_HOST="dsmlp"
LAUNCH="/opt/launch-sh/bin/launch-scipy-ml.sh"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGFILE="realworld_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Real-World DSMLP Runner ==="
echo "Dataset:   $DATASET"
echo "Log file:  /home/jskirzynski/$LOGFILE"
echo ""

# Write the runner script to DSMLP
ssh "$DSMLP_HOST" "source ~/.bashrc; cat > ~/dsmlp_realworld.sh" << 'HEREDOC'
#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/jskirzynski/myenv
cd /home/jskirzynski/concept-benchmark

# Pull latest code
git pull --ff-only

# Install transformers for ViT
pip install -q transformers

# Run pipeline with unbuffered output, tee to log
PYTHONPATH=. python -u scripts/realworld_pipeline.py \
    --dataset DATASET_PLACEHOLDER \
    --seed 42 \
    --epochs 50 \
    --patience 10 \
    --batch-size 16 \
    --lr 5e-5 \
    2>&1 | tee /home/jskirzynski/LOGFILE_PLACEHOLDER

echo ""
echo "=== DONE ==="
echo "Results saved to results/"
ls -la results/*realworld* 2>/dev/null || true
HEREDOC

# Replace placeholders
ssh "$DSMLP_HOST" "sed -i 's/DATASET_PLACEHOLDER/${DATASET}/g; s/LOGFILE_PLACEHOLDER/${LOGFILE}/g' ~/dsmlp_realworld.sh"

echo ">>> Pushing latest commits..."
cd "$LOCAL_REPO"
git push origin HEAD 2>/dev/null || echo "(push skipped or failed — make sure DSMLP has latest code)"

echo ">>> Launching container via nohup (survives SSH disconnect)..."
echo ">>> Log file on DSMLP: ~/$LOGFILE"
echo "---"

# Launch via nohup so it survives SSH disconnection
# Note: /home/jskirzynski/ is the path INSIDE the container; ~ on login node is /dsmlp/home-fs03/...
ssh "$DSMLP_HOST" "source ~/.bashrc; nohup $LAUNCH -s -f -g 1 -c 4 -m 32 bash /home/jskirzynski/dsmlp_realworld.sh > ~/realworld_launch.log 2>&1 &"

echo ">>> Job submitted in background on DSMLP."
echo ">>> Monitor with: ./scripts/monitor_dsmlp.sh status"
echo ">>> Tail log with: ssh dsmlp 'tail -f ~/$LOGFILE'"
echo ">>> When done, fetch results with:"
echo ">>>   rsync -avz dsmlp:~/concept-benchmark/results/*realworld* results/"
