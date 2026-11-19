#!/bin/bash
# Goal: DIFUSCO above DiffUCO (19.88) in the fixed test setting (50 cosine steps, 20 samples, renoise).
# Same model / optimiser as 20261118_difusco_rbsmall_ullabel (AdamW 2e-4 cosine, 200 epochs, 40 graphs/step) unless noted.
#   sd      labels = the largest of 20 samples of the 400-step sampler of the DiffUCO-k200-label DIFUSCO on each training
#           graph (data/rb_small_ullabels/train_difusco_k200_t400_best.pt): distil many-step sampling into 50 steps
#           (sequence-level KD, Kim & Rush 2016), from scratch.
#   sd_ft   the same labels, fine-tuned from that k200-label checkpoint (seed s), AdamW 5e-5 cosine, 30 epochs.
#   tp50_k200 / tp50_kamis   train with 50 diffusion steps (cosine noise schedule) so that each of the 50 test steps is a
#           trained transition (linear inference schedule: steps 50, 49, ..., 1); k200 DiffUCO labels / KaMIS labels.
#   tp50_k200_long  tp50_k200 for 400 epochs.   tp50_fill  tp50 with the k200 labels filled up to the KaMIS size.
#   tp50_near / tp50_tilt  tp50 on all DiffUCO samples (data/rb_small_ullabels/train_diffuco_all200*.pt): among the
#           largest ones, the one most likely under the model (nearest) / one drawn with weight exp(size / 0.2) (tilted).
#   tp100_k200  100 diffusion steps (cosine noise), tested with 50 linear steps (every second trained transition).
# Usage: bash experiments/20261119_difusco_rbsmall_fewstep/launch.sh <gpu> <sd|sd_ft|tp50_k200|tp50_kamis|tp50_k200_long|tp50_fill|tp50_near|tp50_tilt|tp100_k200> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; MODE=$2; SEED=${3:-0}
SETS=/usr0/home/shengyuf/yunyi/data/rb_small_ullabels
INF=cosine; TP=(--num_tp 1000 --diffusion_schedule linear); OPT=(--lr 0.0002 --epochs 200); INIT=()
case "$MODE" in
    sd) LABEL=(--label_set "$SETS/train_difusco_k200_t400_best.pt" --label_choice first) ;;
    sd_ft) LABEL=(--label_set "$SETS/train_difusco_k200_t400_best.pt" --label_choice first); OPT=(--lr 0.00005 --epochs 30)
           INIT=(--checkpoint "$REPO/runs/experiments/20261118_difusco_rbsmall_ullabel/20261118_difusco_rbsmall_ullabel_diffuco_k200_s$SEED/best.pt") ;;
    tp50_k200) LABEL=(--label_set "$SETS/train_diffuco_k200.pt" --label_choice first); TP=(--num_tp 50 --diffusion_schedule cosine); INF=linear ;;
    tp50_kamis) LABEL=(); TP=(--num_tp 50 --diffusion_schedule cosine); INF=linear ;;
    tp50_k200_long) LABEL=(--label_set "$SETS/train_diffuco_k200.pt" --label_choice first); TP=(--num_tp 50 --diffusion_schedule cosine); INF=linear
           OPT=(--lr 0.0002 --epochs 400) ;;
    tp50_fill) LABEL=(--label_set "$SETS/train_diffuco_k200_fill.pt" --label_choice first); TP=(--num_tp 50 --diffusion_schedule cosine); INF=linear ;;
    tp50_near) LABEL=(--label_set "$SETS/train_diffuco_all200_top.pt" --label_choice nearest); TP=(--num_tp 50 --diffusion_schedule cosine); INF=linear ;;
    tp50_tilt) LABEL=(--label_set "$SETS/train_diffuco_all200_sets.pt" --label_choice tilted --label_temp 0.2); TP=(--num_tp 50 --diffusion_schedule cosine); INF=linear ;;
    tp100_k200) LABEL=(--label_set "$SETS/train_diffuco_k200.pt" --label_choice first); TP=(--num_tp 100 --diffusion_schedule cosine); INF=linear ;;
    *) echo "unknown mode $MODE"; exit 1 ;;
esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu
RUN=${EXP_NAME}_${MODE}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
D=/usr1/shengyuf/MIS
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 --main_process_port $((29500 + GPU)) main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_test \
    --problem mis --valid_samples 64 --seed "$SEED" --method DIFUSCO --model EncodeProcessDecode \
    --num_h 64 --num_l 8 "${TP[@]}" --batch_size 40 \
    --optimizer AdamW "${OPT[@]}" --weight_decay 0.0001 --lr_scheduler cosine-decay \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule $INF --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise "${LABEL[@]}" "${INIT[@]}" \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU"
