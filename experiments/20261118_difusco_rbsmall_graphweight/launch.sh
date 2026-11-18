#!/bin/bash
# DIFUSCO on RB-[200-300] with per-graph weights of the training loss (cam/UPSTREAM.md item 45).
# Everything but the loss weights is 20261117_difusco_rbsmall_heatflip lambda 0 (the KaMIS-label baseline):
# uniform forward, 1000 steps, AdamW 2e-4 cosine, 200 epochs, no early stop, a checkpoint per epoch,
# 40 graphs per step split over the GPUs, validation (64 graphs) with the re-noising sampler.
# Weights (KaMIS label throughout; optimal-set counts from data/rb_small_solsets/train.pt):
#   nsets -> w = (number of optimal sets)^-0.5, static
#   ce    -> w = exp(standardized log hardness), hardness = cross-entropy relative to its noise level
#            (after the hardness-weighted curriculum of Zhang et al., AAAI 2022)
# Usage: bash experiments/20261118_difusco_rbsmall_graphweight/launch.sh <gpus, e.g. 0,1> <nsets|ce> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPUS=${1:?usage: launch.sh <gpus> <nsets|ce> [seed]}
MODE=${2:?usage: launch.sh <gpus> <nsets|ce> [seed]}
SEED=${3:-0}
SETS=/usr0/home/shengyuf/yunyi/data/rb_small_solsets
case "$MODE" in
    nsets|ce) LABEL=(--graph_weight "$MODE" --graph_weight_sets "$SETS/train.pt") ;;
    *) echo "unknown mode $MODE"; exit 1 ;;
esac

source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
export CUDA_VISIBLE_DEVICES=$GPUS
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
test $((40 % NPROC)) -eq 0
PORT=$((29500 + ${GPUS%%,*}))
cam_require_gpu

RUN=${EXP_NAME}_${MODE}_s${SEED}
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"
mkdir -p "$OUT"
DATA=/usr1/shengyuf/MIS

cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=$NPROC --main_process_port $PORT main.py \
    --train_path "$DATA/RB_small_train" --valid_path "$DATA/RB_small_valid" --test_path "$DATA/RB_small_test" \
    --problem mis --valid_samples 64 --seed "$SEED" --method DIFUSCO --model EncodeProcessDecode \
    --num_h 64 --num_l 8 --num_tp 1000 --diffusion_schedule linear --batch_size $((40 / NPROC)) \
    --optimizer AdamW --lr 0.0002 --weight_decay 0.0001 --lr_scheduler cosine-decay --epochs 200 \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule cosine --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise "${LABEL[@]}" \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpus $GPUS log $OUT/train.log"
