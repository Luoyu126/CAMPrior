#!/bin/bash
# Stage 2A of docs/20261125/plan.md: single-graph DASBS on path6 and cycle8, 15 stages, the 11-15 configuration
# (controller 80 / corrector 40 steps per stage, batch 32, 4 replicas, rollout refresh every 10 steps, 30 sampling
# steps, temperature 1, penalty 2, log-linear reference gamma 1 alpha 0.5), evaluation on 4096 samples after every
# stage. CPU only, one thread per run, all runs in parallel.
# Variants: faithful = worktree behaviour (corrector-DM at t ~ U[0, 0.99]); tzero = corrector-DM at t = 0.
# Usage: bash experiments/20261125_dasbs_smallgraph/launch.sh [variant ...]   (default: faithful tzero)
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
PY=/usr0/home/shengyuf/yunyi/envs/cam/bin/python
OUT=/usr0/home/shengyuf/yunyi/Schrödinger\ Bridge/CAMPrior/runs/experiments/20261125_dasbs_smallgraph
mkdir -p "$OUT"
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(faithful tzero)
cd "$REPO/cam"
for v in "${VARIANTS[@]}"; do
  case $v in faithful) CT=uniform ;; tzero) CT=zero ;; *) echo "unknown variant $v"; exit 1 ;; esac
  for g in path6 cycle8; do
    for s in 0 1 2; do
      d="$OUT/${v}_${g}_s$s"
      [ -f "$d/checkpoint.pt" ] && { echo "skip $d"; continue; }
      OMP_NUM_THREADS=1 nohup "$PY" scripts/sb/dasbs_single.py --graph $g --output "$d" --seed $s --stages 15 \
        --controller-steps 80 --corrector-steps 40 --batch-size 32 --replicas 4 --refresh 10 --sampling-steps 30 \
        --eval-samples 4096 --corrector-time $CT > "$d.log" 2>&1 &
      echo "$v $g seed $s pid $!"
    done
  done
done
