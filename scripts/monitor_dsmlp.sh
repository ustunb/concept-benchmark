#!/usr/bin/env bash
#
# Monitor DSMLP real-world experiment progress.
#
# Usage:
#   ./scripts/monitor_dsmlp.sh           # tail latest log
#   ./scripts/monitor_dsmlp.sh status    # show current model being trained + epoch
#   ./scripts/monitor_dsmlp.sh pods      # show running pods
#   ./scripts/monitor_dsmlp.sh kill      # kill running pod (requires confirmation)

set -euo pipefail

DSMLP_HOST="dsmlp"
ACTION="${1:-tail}"

case "$ACTION" in
    tail)
        echo ">>> Tailing latest log on DSMLP (Ctrl+C to stop)..."
        ssh "$DSMLP_HOST" 'ls -t ~/realworld_*.log 2>/dev/null | head -1 | xargs -I{} tail -f {}'
        ;;

    status)
        echo ">>> Current training status:"
        ssh "$DSMLP_HOST" bash -c '
            LOG=$(ls -t ~/realworld_*.log 2>/dev/null | head -1)
            if [ -z "$LOG" ]; then
                echo "No log files found"
                exit 0
            fi
            echo "Log: $LOG"
            echo ""

            # Last model being trained
            echo "=== Current Model ==="
            grep -E "Training (DNN|CBM|CEM|ProbCBM|ECBM)" "$LOG" | tail -1

            # Last epoch info
            echo ""
            echo "=== Last Epoch ==="
            grep -iE "epoch|Epoch" "$LOG" | tail -3

            # Any errors?
            echo ""
            echo "=== Recent Errors ==="
            grep -iE "error|exception|traceback|failed" "$LOG" | tail -5 || echo "(none)"

            # Completed models
            echo ""
            echo "=== Completed ==="
            grep -E "(DNN|CBM|CEM|ProbCBM|ECBM) trained|raw accuracy" "$LOG" || echo "(none yet)"
        '
        ;;

    pods)
        echo ">>> Checking DSMLP pods..."
        ssh "$DSMLP_HOST" bash -c '
            K8S_UID=$(id -u)
            export KUBECONFIG=/home/linux/dsmlp/${K8S_UID: -2}/${K8S_UID: -3}/jskirzynski/.kube/config
            kubectl get pods 2>/dev/null || echo "kubectl failed — KUBECONFIG may need updating"
        '
        ;;

    kill)
        echo ">>> Finding pod to kill..."
        ssh "$DSMLP_HOST" bash -c '
            K8S_UID=$(id -u)
            export KUBECONFIG=/home/linux/dsmlp/${K8S_UID: -2}/${K8S_UID: -3}/jskirzynski/.kube/config
            PODS=$(kubectl get pods --no-headers 2>/dev/null | awk "{print \$1}")
            if [ -z "$PODS" ]; then
                echo "No pods found"
                exit 0
            fi
            echo "Found pods:"
            kubectl get pods 2>/dev/null
            echo ""
            echo "Deleting all pods..."
            echo "$PODS" | xargs -I{} kubectl delete pod {} 2>/dev/null
            echo "Done."
        '
        ;;

    *)
        echo "Usage: $0 [tail|status|pods|kill]"
        exit 1
        ;;
esac
