#!/bin/bash
# Run a list of evaluations one after another on one GPU. Each line of LIST: <run dir> <anchor|test> <sampler> <steps>
# <seed> [extra flags]; lines starting with # are skipped, and an evaluation whose result file exists is skipped.
# Usage: bash experiments/20261125_sbdifusco_rbsmall/eval_queue.sh <gpu> <LIST>
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=$1; LIST=$2
while read -r RUN SET SAMPLER STEPS SEED EXTRA; do
  [ -z "${RUN:-}" ] && continue; [[ "$RUN" == \#* ]] && continue
  if [ -f "$RUN/${SET}_${SAMPLER}_t${STEPS}_seed${SEED}_result.txt" ]; then echo "skip $RUN $SET $SAMPLER $STEPS $SEED"; continue; fi
  bash "$EXP_DIR/evaluate.sh" "$GPU" "$RUN" "$SET" "$SAMPLER" "$STEPS" "$SEED" $EXTRA
done < "$LIST"
echo "queue $LIST done $(date '+%F %T')"
