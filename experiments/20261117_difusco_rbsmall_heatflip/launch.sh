#!/bin/bash
# DIFUSCO on RB-[200-300], energy-heated forward (cam/UPSTREAM.md item 42), re-noising sampler.
# Same as 20261116_difusco_rbsmall_heatflip except: no early stop (patience 1000 > 200 epochs, so all
# 200 epochs run), a checkpoint per epoch (--save_every_epoch), and data parallel over the given GPUs
# with 40 graphs per optimizer step in total (40 / #GPUs per GPU; 4 GPUs -> 10 each, validation
# 64 / 4 = 16 graphs per GPU, still exactly 64).
# NCCL_IB_DISABLE=1: NCCL's InfiniBand probe segfaults on sa (checked 11-17 with a 2-process all-reduce).
# Usage: bash experiments/20261117_difusco_rbsmall_heatflip/launch.sh <gpus, e.g. 1,2,3,4> <lambda> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPUS=${1:?usage: launch.sh <gpus> <lambda> [seed]}
LAM=${2:?usage: launch.sh <gpus> <lambda> [seed]}
SEED=${3:-0}

source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR"   # wandb records go to experiments/<name>/wandb/
export CUDA_VISIBLE_DEVICES=$GPUS
export NCCL_IB_DISABLE=1
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
test $((40 % NPROC)) -eq 0
PORT=$((29500 + ${GPUS%%,*}))
cam_require_gpu

RUN=${EXP_NAME}_lam${LAM/./p}_s${SEED}
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"
mkdir -p "$OUT"
DATA=/usr1/shengyuf/MIS   # RB_small_{train,valid,test}, read-only

cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=$NPROC --main_process_port $PORT main.py \
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
    --batch_size $((40 / NPROC)) \
    --optimizer AdamW \
    --lr 0.0002 \
    --weight_decay 0.0001 \
    --lr_scheduler cosine-decay \
    --epochs 200 \
    --patience 1000 \
    --save_every_epoch \
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
echo "$RUN pid $! gpus $GPUS log $OUT/train.log"
