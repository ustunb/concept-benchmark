#!/bin/bash
#
# Sync caches to DSMLP and start the overnight run in tmux.
# Run this locally once the LLM caches are generated.
#
# Usage:
#   bash scripts/sync_and_start_overnight.sh

set -euo pipefail

echo "=== Pushing code ==="
git add -A && git commit -m "Add overnight run scripts" 2>/dev/null || true
git push origin codex-cem-probcbm-baselines

echo "=== Syncing LLM caches to DSMLP ==="
rsync -avz --progress results/cache/ dsmlp:~/concept-benchmark/results/cache/

echo "=== Cache status on DSMLP ==="
ssh dsmlp 'wc -l ~/concept-benchmark/results/cache/llm_interventions_*.jsonl 2>/dev/null'

echo ""
echo "=== Starting overnight run in tmux ==="
echo "This will SSH to DSMLP and start the run."
echo "After it starts, press Ctrl+B then D to detach tmux."
echo ""

ssh -t dsmlp 'tmux new-session -d -s overnight "bash ~/concept-benchmark/scripts/dsmlp_overnight.sh 2>&1 | tee ~/overnight.log" && tmux attach -t overnight'
