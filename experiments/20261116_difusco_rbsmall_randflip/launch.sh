#!/bin/bash
# DIFUSCO on RB-[200-300], forward process = uniform random flip (D3PM, Q_t = (1-b)I + b/2 11^T).
# Center config: 40 graphs per optimizer step (same as the CAM-paper DiffUCO run: 4 per GPU x 10 GPUs),
# one epoch = all 4000 training graphs = 100 steps (also the same as DiffUCO).
# Hyperparameters otherwise follow Shengyu's mis/mis_rb_small_difusco.sh (lr 2e-4, wd 1e-4, 200 epochs,
# patience 20, num_tp 1000, cosine inference 50 steps x 20 samples).
# Usage: bash experiments/20261116_difusco_rbsmall_randflip/launch.sh <gpu> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=${1:?usage: launch.sh <gpu> [seed]}
SEED=${2:-0}

source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR"   # wandb records go to experiments/<name>/wandb/
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
    --valid_samples 64 \
    --seed "$SEED" \
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
    --do_train \
    --save_dir "$OUT" \
    --wandb_name SLVUL \
    --wandb_run_name "$RUN" \
    > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU log $OUT/train.log"
