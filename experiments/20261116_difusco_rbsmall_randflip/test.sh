#!/bin/bash
# Test best.pt of one training run on all 1000 RB_small_test graphs, the way the CAM paper did
# (Appendix B.2; Shengyu's DiffUCO test runs kur6gnkv / 40tvlj2t): the training command with
# --do_train removed, so it loads save_dir/best.pt; no --test_samples limit; 50 cosine steps x 20
# samples, greedy decoding, best of 20 per graph; seed 0.
# wandb stays off (no separate test run); the result goes into the training run's summary afterwards.
# Usage: bash experiments/20261116_difusco_rbsmall_randflip/test.sh <gpu> [seed] [test_seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=${1:?usage: test.sh <gpu> [seed] [test_seed]}
SEED=${2:-0}
TEST_SEED=${3:-0}

source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"   # CAM_WANDB unset: wandb disabled
export CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu

RUN=${EXP_NAME}_s${SEED}
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"
test -f "$OUT/best.pt"
DATA=/usr1/shengyuf/MIS
LOG="$OUT/test_seed${TEST_SEED}.log"

cd "$REPO/cam"
nohup "$CAM_PYTHON" main.py \
    --train_path "$DATA/RB_small_train" \
    --valid_path "$DATA/RB_small_valid" \
    --test_path "$DATA/RB_small_test" \
    --problem mis \
    --valid_samples 64 \
    --seed "$TEST_SEED" \
    --method DIFUSCO \
    --model EncodeProcessDecode \
    --num_h 64 \
    --num_l 8 \
    --num_tp 1000 \
    --diffusion_schedule linear \
    --batch_size 40 \
    --optimizer AdamW \
    --lr 0.0002 \
    --weight_decay 0.0001 \
    --lr_scheduler cosine-decay \
    --epochs 200 \
    --patience 20 \
    --num_t 50 \
    --num_k 20 \
    --inference_schedule cosine \
    --num_workers 4 \
    --save_dir "$OUT" \
    --log_file "$OUT/test_seed${TEST_SEED}_result.txt" \
    --wandb_name SLVUL \
    --wandb_run_name "$RUN" \
    > "$LOG" 2>&1 &
echo "$RUN test seed $TEST_SEED pid $! gpu $GPU log $LOG"
