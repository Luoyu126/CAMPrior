#!/bin/bash
# Wait for a training run to exit, then test its best.pt (chosen on validation) with both samplers x
# test seeds 0 and 1 via test.sh. Each of the 4 tests waits for an idle GPU from the given list (no
# compute process on it), so nothing is launched onto a GPU someone else is using.
# Usage: nohup bash experiments/20261117_difusco_rbsmall_heatflip/test_after_train.sh \
#            <train_pid> <randflip|lam0|lam0p1> <gpus, e.g. 1,2,3,4> > <log> 2>&1 &
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID=${1:?usage}; CKPT=${2:?usage}; IFS=, read -ra GPUS <<< "${3:?usage}"

while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "$(date -u) training pid $PID exited"
sleep 60   # let the DDP workers release their GPUs

idle() { [ "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ]; }
used=()
for job in "posterior 0" "posterior 1" "renoise 0" "renoise 1"; do
    set -- $job
    while :; do
        for g in "${GPUS[@]}"; do
            [[ " ${used[*]-} " == *" $g "* ]] && continue
            if idle "$g"; then used+=("$g"); bash "$EXP_DIR/test.sh" "$g" "$CKPT" "$1" "$2"; break 2; fi
        done
        sleep 300
    done
    sleep 20   # let the test claim its GPU before the next idle check
done
echo "$(date -u) all 4 tests launched"
