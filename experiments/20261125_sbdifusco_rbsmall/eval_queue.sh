#!/bin/bash
# Run a list of evaluations one after another on one GPU. Each line of LIST: <run> <anchor|test|val64> <sampler> <steps>
# <seed> [extra flags], <run> a directory name under runs/experiments/20261125_sbdifusco_rbsmall/ (the repo path has a
# space, so no full paths); lines starting with # are skipped, and an evaluation whose result file exists is skipped.
# PAIR_TRACE=1 in the environment is passed on to evaluate.sh.
# Usage: bash experiments/20261125_sbdifusco_rbsmall/eval_queue.sh <gpu> <LIST>
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$(cd "$EXP_DIR/../.." && pwd)/runs/experiments/$(basename "$EXP_DIR")"
GPU=$1; LIST=$2
while read -r NAME SET SAMPLER STEPS SEED EXTRA; do
  [ -z "${NAME:-}" ] && continue; [[ "$NAME" == \#* ]] && continue
  RUN="$RUNS/$NAME"
  if [ -f "$RUN/${SET}_${SAMPLER}_t${STEPS}_seed${SEED}_result.txt" ]; then echo "skip $RUN $SET $SAMPLER $STEPS $SEED"; continue; fi
  bash "$EXP_DIR/evaluate.sh" "$GPU" "$RUN" "$SET" "$SAMPLER" "$STEPS" "$SEED" $EXTRA < /dev/null
done < "$LIST"
echo "queue $LIST done $(date '+%F %T')"
