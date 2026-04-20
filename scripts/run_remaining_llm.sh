#!/bin/bash
#
# Run remaining LLM intervention cells on DSMLP.
# Pod 1: ECBM × llm (4 sources, no clip — cache not ready yet)
# Pod 2: ALL families × clip_concepts × llm (after clip cache synced)
#
# Usage: nohup bash scripts/run_remaining_llm.sh > ~/remaining_llm.log 2>&1 &

set -uo pipefail
source /opt/launch-sh/lib/kubevars.sh 2>/dev/null

COMMON="--seed 1014 --concept-preset foot_subtypes --strategy exactly_k --budgets 0 1 3 max --llm-cache-only --stages intervene collect"
NODE=18

run_pod() {
    local label="$1"; shift; local cmd="$*"
    echo ""; echo "=== POD: $label at $(date) ==="

    echo "#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/jskirzynski/myenv
cd /home/jskirzynski/concept-benchmark
git checkout -- . && git pull --ff-only 2>&1
export PYTHONUNBUFFERED=1
$cmd" > "$HOME/dsmlp_run.sh"

    /opt/launch-sh/bin/launch-scipy-ml.sh -s -g 1 -c 4 -m 32 -n $NODE -b bash /home/jskirzynski/dsmlp_run.sh 2>&1

    while true; do
        sleep 30
        POD=$(kubectl get pods -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        [ -z "$POD" ] && echo "No pod found" && break
        STATUS=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        case "$STATUS" in
            Succeeded) echo "=== $label DONE at $(date) ==="; kubectl logs "$POD" --tail=10 2>/dev/null; kubectl delete pod "$POD" 2>/dev/null; break ;;
            Failed|Error) echo "=== $label FAILED at $(date) ==="; kubectl logs "$POD" --tail=20 2>/dev/null; kubectl delete pod "$POD" 2>/dev/null; break ;;
            *) printf "." ;;
        esac
    done
}

echo "=== REMAINING LLM CELLS at $(date) ==="

# Pod 1: ECBM × llm (no clip)
run_pod "ecbm-llm" \
    "PYTHONPATH=. python -u scripts/robot_pipeline.py $COMMON --cbm-family ecbm --concept-sources ground_truth human_concepts machine_annotation llm_concepts --intervention-sources llm"

# Pod 2: ALL families × clip_concepts × llm
for family in cbm cem probcbm ecbm; do
    run_pod "${family}-clip-llm" \
        "PYTHONPATH=. python -u scripts/robot_pipeline.py $COMMON --cbm-family $family --concept-sources clip_concepts --intervention-sources llm"
done

echo "=== ALL DONE at $(date) ==="
