#!/bin/bash
#
# DSMLP Overnight Runner
# Run inside tmux on the DSMLP login node.
# Chains GPU pods sequentially, each within the 6h limit.
#
# Usage:
#   ssh dsmlp
#   tmux new -s overnight
#   bash ~/concept-benchmark/scripts/dsmlp_overnight.sh
#   # Ctrl+B, D to detach. Close laptop.
#   # Later: ssh dsmlp && tmux attach -t overnight

set -euo pipefail

LAUNCH="/opt/launch-sh/bin/launch-scipy-ml.sh"
REPO="$HOME/concept-benchmark"
COMMON="--seed 1014 --concept-preset foot_subtypes --strategy exactly_k --budgets 1 2 5"

# Helper: run a command in a GPU pod, wait for completion
run_pod() {
    local label="$1"
    shift
    local cmd="$*"

    echo ""
    echo "========================================"
    echo "=== POD: $label ==="
    echo "=== $(date) ==="
    echo "========================================"

    # Write runner script
    cat > "$HOME/dsmlp_run.sh" << INNEREOF
#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate $HOME/myenv
cd $REPO
git checkout -- .
git fetch origin && git checkout codex-cem-probcbm-baselines && git pull --ff-only 2>&1
export PYTHONUNBUFFERED=1
pip install openai -q 2>/dev/null || true
$cmd
INNEREOF

    # Launch pod and wait
    source /opt/launch-sh/lib/kubevars.sh 2>/dev/null
    $LAUNCH -s -g 1 -c 4 -m 32 -f bash "$HOME/dsmlp_run.sh"
    local exit_code=$?

    echo "=== POD $label finished at $(date), exit=$exit_code ==="
    return $exit_code
}

# Helper: run a CPU-only pod (no GPU needed)
run_cpu_pod() {
    local label="$1"
    shift
    local cmd="$*"

    echo ""
    echo "=== CPU POD: $label ==="
    echo "=== $(date) ==="

    cat > "$HOME/dsmlp_run.sh" << INNEREOF
#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate $HOME/myenv
cd $REPO
git checkout -- .
git fetch origin && git checkout codex-cem-probcbm-baselines && git pull --ff-only 2>&1
export PYTHONUNBUFFERED=1
pip install openai -q 2>/dev/null || true
$cmd
INNEREOF

    source /opt/launch-sh/lib/kubevars.sh 2>/dev/null
    $LAUNCH -s -g 0 -c 4 -m 16 -f bash "$HOME/dsmlp_run.sh"
    echo "=== CPU POD $label done at $(date) ==="
}

echo "============================================"
echo "=== OVERNIGHT RUN STARTING ==="
echo "=== $(date) ==="
echo "============================================"

# ─── POD 1: Fast cells (CBM/CEM/ProbCBM × missing non-LLM) ──────────
# ground_truth × expert: CEM, ProbCBM
# machine_annotation × perfect: CBM, CEM, ProbCBM
# llm_concepts × {perfect, expert}: CBM, CEM, ProbCBM
# clip_concepts × {perfect, expert}: CBM, CEM, ProbCBM
# Estimated: ~3.5h

for family in cbm cem probcbm; do
    run_pod "fast-$family" \
        "PYTHONPATH=. python -u scripts/robot_pipeline.py $COMMON \
            --cbm-family $family \
            --concept-sources ground_truth machine_annotation llm_concepts clip_concepts \
            --intervention-sources perfect expert \
            --llm-cache-only \
            --stages intervene collect"
done

# ─── POD 2-6: ECBM cells (2 cells per pod, ~5h each) ────────────────
# Each concept_source × {perfect, expert} = 2 cells × ~2.5h = ~5h

for cs in human_concepts ground_truth machine_annotation llm_concepts clip_concepts; do
    run_pod "ecbm-$cs" \
        "PYTHONPATH=. python -u scripts/robot_pipeline.py $COMMON \
            --cbm-family ecbm \
            --concept-sources $cs \
            --intervention-sources perfect expert \
            --stages intervene collect"
done

# ─── POD 7: LLM intervention cells (after caches synced) ─────────────
# All families × all concept sources × llm intervention
# Caches must exist by now (synced from local or generated on DSMLP)

for family in cbm cem probcbm ecbm; do
    run_pod "llm-$family" \
        "PYTHONPATH=. python -u scripts/robot_pipeline.py $COMMON \
            --cbm-family $family \
            --concept-sources ground_truth human_concepts machine_annotation llm_concepts clip_concepts \
            --intervention-sources llm \
            --llm-cache-only \
            --stages intervene collect"
done

echo ""
echo "============================================"
echo "=== ALL DONE ==="
echo "=== $(date) ==="
echo "============================================"
ls -lt "$REPO/results/"*results*.csv | head -20
