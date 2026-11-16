#!/bin/bash
# DIFUSCO on RB-[200-300], forward process = energy-heated flip (cam/UPSTREAM.md item 42):
# node i flips at step s with prob beta_s * sigmoid(lambda_s * dE_i), lambda_s decays linearly from
# LAMBDA to 0 at t = 500, uniform flips after that. Validation and test use the re-noising sampler
# (predict x0, draw it, run the same forward to the next step), so inference sees the training marginals.
# LAMBDA = 0 is the control: uniform forward (same training as 20261116_difusco_rbsmall_randflip),
# but the re-noising sampler.
# Everything else is the same as 20261116_difusco_rbsmall_randflip.
# Usage: bash experiments/20261116_difusco_rbsmall_heatflip/launch.sh <gpu> <lambda> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=${1:?usage: launch.sh <gpu> <lambda> [seed]}
LAM=${2:?usage: launch.sh <gpu> <lambda> [seed]}
SEED=${3:-0}

source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR"   # wandb records go to experiments/<name>/wandb/
export CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu

RUN=${EXP_NAME}_lam${LAM/./p}_s${SEED}
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
    --heat_lambda "$LAM" \
    --heat_until 0.5 \
    --difusco_sampler renoise \
    --do_train \
    --save_dir "$OUT" \
    --wandb_name SLVUL \
    --wandb_run_name "$RUN" \
    > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU log $OUT/train.log"
