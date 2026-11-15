#!/bin/bash
# Template: copy this folder to experiments/YYYYMMDD_<name>/ and edit.
# Usage: bash experiments/<name>/launch.sh <gpu> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=${1:?usage: launch.sh <gpu> [seed]}
SEED=${2:-0}

source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu

RUN=${EXP_NAME}_s${SEED}
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"
mkdir -p "$OUT"
DATA=/usr1/shengyuf/MIS   # RB_small_{train,valid,test}, read-only

cd "$REPO/cam"
nohup "$CAM_PYTHON" main.py \
    --train_path "$DATA/RB_small_train" \
    --valid_path "$DATA/RB_small_valid" \
    --test_path "$DATA/RB_small_test" \
    --problem mis \
    --seed "$SEED" \
    --do_train \
    --save_dir "$OUT" \
    --wandb_name SLVUL \
    --wandb_run_name "$RUN" \
    > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU log $OUT/train.log"
