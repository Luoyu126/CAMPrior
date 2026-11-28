#!/bin/bash
# Stage E of docs/20261126/plan_endpoint.md: route-1 SB-coupled DIFUSCO with new end distributions pi1 built around the
# KaMIS optima. Training identical to experiments/20261125_sbdifusco_rbsmall_labelvar/launch.sh (1E): 50 steps, cosine,
# 400 epochs, batch 40 on one GPU, AdamW 2e-4, soft targets. Only the label set and its weighting change:
#   <set> is data/rb_small_sb/<set>_loaderorder.pt (built by cam/scripts/sb/neighborhood_labels.py or a sibling
#   script; see the plan), or k128 for the KaMIS multi-solutions. pi1(m) ∝ exp(|s_m| / T), or uniform (T = none).
# Coupling: omega W > 0 the SB coupling, W = 0 the independent coupling (the control of the same pi1).
# Potentials: data/rb_small_sb/logg_<set>_w<W>[_T<T>].pt (scripts/sb/precompute_logg.py; for W = 0 only if weighted).
# Env: EPOCHS (default 400), TAG (run-name suffix), RESUME=<epoch>: continue an interrupted run from epoch_<epoch-1>.pt
#   (--start_epoch: the learning-rate schedule resumes, optimizer moments start fresh; best.pt tracking starts at the best
#   eval/object of the earlier logs, --best_eval_init); log train_resume<epoch>.log.
# Usage: bash experiments/20261126_sb_endpoint/launch.sh <gpu> <set> <W> [T|none] [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; SET=$2; W=$3; T=${4:-none}; SEED=${5:-0}; EPOCHS=${EPOCHS:-400}; TAG=${TAG:-}
DS=/usr0/home/shengyuf/yunyi/data/rb_small_sb
case "$SET" in
    k128) LS=/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/k128_loaderorder.pt ;;
    *) LS=$DS/${SET}_loaderorder.pt ;;
esac
test -f "$LS" || { echo "missing $LS"; exit 1; }
LABEL=(--label_set "$LS" --label_choice soft --sb_coupling --sb_omega "$W")
TAGT=""; [ "$T" != none ] && TAGT="_T$T"
G=$DS/logg_${SET}_w${W}${TAGT}.pt
if [ "$W" != 0 ] || [ "$T" != none ]; then
    test -f "$G" || { echo "missing $G (scripts/sb/precompute_logg.py)"; exit 1; }
    LABEL+=(--sb_logg "$G")
fi
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu
RUN=${EXP_NAME}_${SET}_w${W}${TAGT}_s${SEED}${TAG}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"
LOG="$OUT/train.log"; RES=()
if [ -n "${RESUME:-}" ]; then
    CK="$OUT/epoch_$(printf %04d $((RESUME - 1))).pt"; test -f "$CK" || { echo "no $CK"; exit 1; }
    LOG="$OUT/train_resume$RESUME.log"; test -e "$LOG" && { echo "$LOG exists"; exit 1; }
    BEST=$(cat "$OUT"/train*.log | grep -aoE "^Epoch: [0-9]+ eval/object: [-0-9.e]+" \
        | awk -v r="$RESUME" '$2 < r {print $4}' | sort -g | head -1)
    test -n "$BEST" || { echo "no eval/object before epoch $RESUME in $OUT/train*.log"; exit 1; }
    RES=(--checkpoint "$CK" --start_epoch "$RESUME" --best_eval_init "$BEST")
    export CAM_ALLOW_OVERWRITE=1  # main.py otherwise refuses a save_dir that already holds checkpoints
else
    test -e "$OUT/train.log" && { echo "$OUT exists"; exit 1; }
    mkdir -p "$OUT"
fi
D=/usr1/shengyuf/MIS
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 --main_process_port $((29800 + RANDOM % 1000)) main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_test \
    --problem mis --valid_samples 64 --seed "$SEED" --method DIFUSCO --model EncodeProcessDecode \
    --num_h 64 --num_l 8 --num_tp 50 --diffusion_schedule cosine --batch_size 40 \
    --optimizer AdamW --lr 0.0002 --epochs "$EPOCHS" --weight_decay 0.0001 --lr_scheduler cosine-decay \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule linear --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise "${LABEL[@]}" "${RES[@]}" \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$LOG" 2>&1 &
echo "$RUN pid $! gpu $GPU"
