#!/bin/bash
# Stage E evaluation of one run (protocol of 1C / 1E, experiments/20261125_sbdifusco_rbsmall/evaluate.sh): wait until its
# training process (pid, optional) has exited, then evaluate best.pt on the validation anchor (graphs 64-499, best of 20,
# evaluation seed 0): renoise 50 / 10 / 5 steps and posterior 50 steps; then the start-dependence pair trace (anchor,
# renoise 50, seed 100). Existing result files are skipped. Selection decisions use the anchor only.
# Training must have finished (epoch_<EPOCHS-1>.pt exists, EPOCHS default 400); otherwise nothing is evaluated.
# Usage: bash experiments/20261126_sb_endpoint/eval_run.sh <gpu> <run name> <omega> [training pid]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; RUN="$REPO/runs/experiments/$(basename "$EXP_DIR")/$2"; W=$3; PID=${4:-}
EV="$REPO/experiments/20261125_sbdifusco_rbsmall/evaluate.sh"
[ -n "$PID" ] && while kill -0 "$PID" 2>/dev/null; do sleep 60; done
sleep 30
LAST="$RUN/epoch_$(printf %04d $((${EPOCHS:-400} - 1))).pt"
test -f "$LAST" || { echo "$(date '+%F %T') $(basename "$RUN"): no $LAST, training did not finish; not evaluating"; exit 1; }
for spec in "renoise 50 0" "renoise 10 0" "posterior 50 0" "renoise 5 0" "renoise 50 100 pair"; do
    set -- $spec
    [ -f "$RUN/anchor_$1_t$2_seed$3_result.txt" ] && continue
    if [ "${4:-}" = pair ]; then PAIR_TRACE=1 bash "$EV" "$GPU" "$RUN" anchor "$1" "$2" "$3" --sb_coupling --sb_omega "$W"
    else bash "$EV" "$GPU" "$RUN" anchor "$1" "$2" "$3" --sb_coupling --sb_omega "$W"; fi
done
echo "$(date '+%F %T') $(basename "$RUN") evaluations done"
