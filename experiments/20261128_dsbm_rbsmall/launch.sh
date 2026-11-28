#!/bin/bash
# Route 1' (DSBM, docs/20261128/plan_route1prime.md): one run of cam/scripts/sb/dsbm_loop.py on RB-small, output in
# runs/experiments/20261128_dsbm_rbsmall/<name>, W&B under this directory. The driver restarts from its state.json,
# so the same command resumes an interrupted run.
# Usage: bash experiments/20261128_dsbm_rbsmall/launch.sh <name> <driver flags...>
#   e.g.  launch.sh dsbm_w0.35 --mode dsbm --omega 0.35 --gpus 0,1
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
NAME=$1; shift
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=${CAM_WANDB:-1}
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR"
OUT="$REPO/runs/experiments/$EXP_NAME/${EXP_NAME}_$NAME"; mkdir -p "$OUT"
cd "$REPO"
nohup "$CAM_PYTHON" cam/scripts/sb/dsbm_loop.py --out "$OUT" "$@" >> "$OUT/driver.out" 2>&1 &
echo "$NAME pid $! out $OUT"
