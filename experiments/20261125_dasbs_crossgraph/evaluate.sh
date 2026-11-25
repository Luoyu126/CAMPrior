#!/bin/bash
# Stage 2B evaluation: exact terminal law of the trained cross-graph DASBS (cam/scripts/sb/dasbs_exact_eval.py) on the
# held-out small graphs. The checkpoint is fixed in advance (last epoch, epoch_0047.pt): validation MIS size saturates
# on graphs this small, so no checkpoint selection is done. Sets: valid, test, wl (pairs of graphs that 1-WL cannot
# tell apart). Writes exact_<set>_<ckpt>.jsonl next to the checkpoint; the last line is the summary (means;
# tv_le_half_source = fraction of graphs whose TV is at most half of the source TV; energy_rel_gap).
# About 1 h per set on 4 threads (64 graphs, n <= 12).
# Usage: bash experiments/20261125_dasbs_crossgraph/evaluate.sh <mode: sb|ml|nocorr> [seed] [ckpt name]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
MODE=$1; SEED=${2:-0}; CK=${3:-epoch_0047}
case $MODE in
  sb|nocorr) ALPHA=0.5 ;;
  ml) ALPHA=0 ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"
export WANDB_MODE=disabled OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""   # the three sets run in parallel
D=/usr0/home/shengyuf/yunyi/data/sb_small
RUN="$REPO/runs/experiments/20261125_dasbs_crossgraph/20261125_dasbs_crossgraph_${MODE}_s$SEED"
cd "$REPO/cam"
for SET in test valid wl; do
  OUT="$RUN/exact_${SET}_${CK}.jsonl"
  [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
  EXTRA=""; [ "$SET" = wl ] && EXTRA="--pairs $D/wl/pairs.txt"
  "$CAM_PYTHON" scripts/sb/dasbs_exact_eval.py --ckpt "$RUN/$CK.pt" --data $D/$SET --dasbs_alpha $ALPHA --tau 1 \
    --beta 2 $EXTRA > "$OUT.tmp" 2> "$OUT.err" && mv "$OUT.tmp" "$OUT" &
done
wait
for SET in test valid wl; do echo "$MODE $SET: $(tail -1 "$RUN/exact_${SET}_${CK}.jsonl" 2>/dev/null | cut -c1-400)"; done
