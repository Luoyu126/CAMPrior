#!/bin/bash
# TSP-500 label sources x label use: DIFUSCO trained on several tours per instance from
#   lkh  LKH-3, 32 seeds per instance (RUNS=1, MAX_TRIALS=1), distinct tours (cam/scripts/tsp/lkh_labels.py)
#   cam  the released CAM TSP-500 checkpoint, 16 samples, 10 steps, greedy merge + 1000 2-opt (dump_tours.py)
# both cut per instance to the same number M = min(8, counts) of their shortest tours (match_tour_sets.py),
# used as
#   random  one tour of the set drawn uniformly per step ("最大解": a random one of the best tours)
#   soft    posterior mean over the set given x_t as the target (--label_choice soft)
# Tours are undirected edge labels on the sparse kNN graph (cam/utils/tour_sets.py) in every arm.
# Network and training as DIFUSCO TSP-500 (Sun & Yang 2023, App. E): AnGNN 12 layers x 256, sparse k=50,
# AdamW 2e-4 cosine decay, wd 1e-4; 50 cosine diffusion steps (as our MIS SOP, instead of DIFUSCO's 1000).
# Differences, for compute: 12,800 instances (DIFUSCO: 128k), 50 epochs from scratch (DIFUSCO: from the TSP-100
# checkpoint), batch 2 GPUs x 8 = 16 (DIFUSCO: 64). Validation (checkpoint selection): 16 valid instances,
# renoise, 50 steps, 4 samples, no 2-opt.
# Usage: bash experiments/20261121_difusco_tsp500_labels/launch.sh <gpus, e.g. 0,1> <lkh|cam>_<random|soft> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPUS=$1; MODE=$2; SEED=${3:-0}
SRC=${MODE%_*}; CHOICE=${MODE#*_}
[[ "$SRC" =~ ^(lkh|cam)$ && "$CHOICE" =~ ^(random|soft)$ ]] || { echo "unknown mode $MODE"; exit 1; }
L=/usr0/home/shengyuf/yunyi/data/tsp500_sl; T=/usr0/home/shengyuf/yunyi/data/CAM_data/TSP
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS
cam_require_gpu
RUN=${EXP_NAME}_${MODE}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=$NP --num_machines=1 --dynamo_backend=no \
    --main_process_port $((29600 + ${GPUS%%,*})) main.py \
    --train_path $L/tsp500_train_lkh.txt --train_samples 12800 --valid_path $T/tsp500_valid_concorde.txt \
    --valid_samples 16 --test_path $T/tsp500_test_concorde.txt --test_samples 128 --problem tsp --seed "$SEED" \
    --method DIFUSCO --model AnGNN --num_h 256 --num_l 12 --sparse_factor 50 --use_activation_checkpoint \
    --num_tp 50 --diffusion_schedule cosine --batch_size 8 --optimizer AdamW --lr 0.0002 --weight_decay 0.0001 \
    --lr_scheduler cosine-decay --epochs 50 --patience 1000 --save_every_epoch \
    --num_t 50 --num_k 4 --inference_schedule linear --two_opt_iterations 0 --difusco_sampler renoise \
    --label_set $L/matched/matched_$SRC.pt --label_choice $CHOICE --num_workers 4 \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpus $GPUS"
