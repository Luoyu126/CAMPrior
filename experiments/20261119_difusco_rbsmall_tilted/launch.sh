#!/bin/bash
# DIFUSCO on RB-[200-300] trained on the size-tilted sample distribution of DiffUCO (cam/UPSTREAM.md item 47).
# Everything but the label is 20261117_difusco_rbsmall_heatflip lambda 0 (the KaMIS-label baseline):
# uniform forward, 1000 steps, AdamW 2e-4 cosine, 200 epochs, no early stop, a checkpoint per epoch,
# 40 graphs per step split over the GPUs, validation (64 graphs) with the re-noising sampler.
# Labels: data/rb_small_ullabels/train_diffuco_all200_sets.pt, all 200 samples (with repeats) of the DiffUCO
#   RB-small checkpoint (20260513_diffuco_rbsmall_annealing, test 19.88) on each training graph, from
#   scripts/mis/ul_samples.py (seed 2). Each step draws one per graph with probability ~ exp(size / T).
#   Expected label size: T=inf 19.11 (DiffUCO per-sample mean), 0.5 19.71, 0.3 19.92, 0.2 20.04.
# Usage: bash experiments/20261119_difusco_rbsmall_tilted/launch.sh <gpus, e.g. 0 or 0,1> <T, e.g. inf|0.5> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPUS=${1:?usage: launch.sh <gpus> <T> [seed]}
TEMP=${2:?usage: launch.sh <gpus> <T> [seed]}
SEED=${3:-0}
SETS=/usr0/home/shengyuf/yunyi/data/rb_small_ullabels
MODE=T$TEMP
if [ "$TEMP" = inf ]; then LABEL=(--label_set "$SETS/train_diffuco_all200_sets.pt" --label_choice random)
else LABEL=(--label_set "$SETS/train_diffuco_all200_sets.pt" --label_choice tilted --label_temp "$TEMP"); fi

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
