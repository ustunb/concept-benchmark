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

echo ">>> Launching container (1 GPU, 4 CPUs, 32GB RAM)..."
echo ">>> Output will stream below AND save to ~/$LOGFILE on DSMLP"
echo "---"

ssh -t "$DSMLP_HOST" "source ~/.bashrc; $LAUNCH -s -f -g 1 -c 4 -m 32 bash /home/jskirzynski/dsmlp_realworld.sh"

EXIT_CODE=$?
echo "---"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ">>> Success! Fetching results..."
    rsync -avz --progress \
        "${DSMLP_HOST}:~/concept-benchmark/results/*realworld*" \
        "${LOCAL_REPO}/results/"
    echo ">>> Results synced to ${LOCAL_REPO}/results/"
else
    echo ">>> FAILED (exit code $EXIT_CODE)"
    echo ">>> Check log: ssh $DSMLP_HOST cat ~/$LOGFILE"
    echo ">>> Check pod: ssh $DSMLP_HOST then set KUBECONFIG and kubectl get pods"
fi

exit $EXIT_CODE
