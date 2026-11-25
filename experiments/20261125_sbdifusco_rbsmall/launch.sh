#!/bin/bash
# Route 1 of docs/20261125/plan.md (stages 1B-1F): DIFUSCO trained on the KaMIS multi-solution label sets
# (matched_kamis_loaderorder.pt, ~42 per graph), with the independent coupling (baseline) or the Schrödinger-bridge
# coupling between Bernoulli(0.5)^n and the label set (math_route1.md). Training as the baseline
# 20261119_difusco_rbsmall_reach q5_mkamisfix_long{,_soft}: 50 diffusion steps, cosine, 400 epochs, 40 graphs per step.
# Modes:
#   indep        independent coupling, one label drawn uniformly per step   (= q5_mkamisfix_long)
#   indep_soft   independent coupling, soft target (posterior mean)         (= q5_mkamisfix_long_soft)
# <gpu> may be a list (e.g. 4,5): one process per GPU, 40 / #GPUs graphs each.
# Usage: bash experiments/20261125_sbdifusco_rbsmall/launch.sh <gpu> <mode> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; MODE=$2; SEED=${3:-0}
LS=/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt
case "$MODE" in
    indep) LABEL=(--label_set "$LS" --label_choice random) ;;
    indep_soft) LABEL=(--label_set "$LS" --label_choice soft) ;;
    *) echo "unknown mode $MODE"; exit 1 ;;
esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu
RUN=${EXP_NAME}_${MODE}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
D=/usr1/shengyuf/MIS
NPROC=$(( $(tr -cd , <<< "$GPU" | wc -c) + 1 )); (( 40 % NPROC == 0 )) || { echo "40 graphs per step need a GPU count dividing 40"; exit 1; }
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=$NPROC --main_process_port $((29600 + ${GPU%%,*})) main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_test \
    --problem mis --valid_samples 64 --seed "$SEED" --method DIFUSCO --model EncodeProcessDecode \
    --num_h 64 --num_l 8 --num_tp 50 --diffusion_schedule cosine --batch_size $((40 / NPROC)) \
    --optimizer AdamW --lr 0.0002 --epochs 400 --weight_decay 0.0001 --lr_scheduler cosine-decay \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule linear --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise "${LABEL[@]}" \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU"
