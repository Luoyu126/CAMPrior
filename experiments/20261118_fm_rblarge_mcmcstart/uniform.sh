#!/bin/bash
# Control for this experiment (README, baselines item 5): the original FM, uniform Ber(1/2) start, no start
# bank, mixture path. Everything else as pipeline.sh: 8 x 64, 40 graphs per step (2 GPUs x 20), AdamW 2e-4,
# 200 epochs, no early stop, same local KaMIS labels; then test best.pt on the 1000 test graphs.
# Usage: bash experiments/20261118_fm_rblarge_mcmcstart/uniform.sh <gpu> <gpu>
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
DATA=/usr0/home/shengyuf/yunyi/data/rb_large_kamis
RUN=20261118_fm_rblarge_uniformstart_s0
O="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$O"
COMMON=(--train_path "$DATA/RB_large_train" --valid_path "$DATA/RB_large_valid" --test_path "$DATA/RB_large_test"
    --problem mis --valid_samples 64 --seed 0 --method FM --model EncodeProcessDecode --num_h 64 --num_l 8
    --batch_size 20 --optimizer AdamW --lr 0.0002 --weight_decay 0.0001 --epochs 200 --patience 1000
    --num_t 50 --num_k 20 --inference_schedule cosine --num_workers 4 --wandb_name SLVUL --wandb_run_name "$RUN")
L() { CUDA_VISIBLE_DEVICES=$1,$2 "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=2 \
    --main_process_port $((29900 + $1)) main.py "${@:3}"; }
cd "$REPO/cam"
L $1 $2 "${COMMON[@]}" --save_every_epoch --use_activation_checkpoint --do_train --save_dir "$O" > "$O/train.log" 2>&1
echo "$(date -u '+%F %T') train exit $?"
L $1 $2 "${COMMON[@]}" --save_dir "$O" --log_file "$O/test_result.txt" > "$O/test.log" 2>&1
echo "$(date -u '+%F %T') test exit $? $(grep 'Avg. size' "$O/test_result.txt" 2>/dev/null)"
