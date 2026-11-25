#!/bin/bash
# Stage 1E of docs/20261125/plan.md: label variants for the route-1 SB-coupled DIFUSCO. Same training as
# experiments/20261125_sbdifusco_rbsmall/launch.sh (50 steps, cosine, 400 epochs, batch 40 on one GPU, AdamW 2e-4,
# soft targets), different end distribution pi1:
#   single   the single KaMIS label + 32 perturbations (data/rb_small_sb/single_repair10_loaderorder.pt)
#   mkamis   the KaMIS multi-solutions (matched, ~42) + 2 perturbations each (mkamis_repair10_loaderorder.pt)
#   k128     more KaMIS multi-solutions (rb_small_kamis_multi/k128_loaderorder.pt, ~66.9 per graph)
# Perturbation: flip 10% of the bits, then greedy repair to a maximal independent set (cam/scripts/sb/perturb_labels.py;
# analysis in this directory's README). Weighting: pi1(m) ∝ exp(|s_m| / T), or uniform (T = none).
# Coupling: omega W > 0 the SB coupling, W = 0 the independent coupling (the control of the same label set).
# Potentials: data/rb_small_sb/logg_<set>_w<W>[_T<T>].pt (scripts/sb/precompute_logg.py; for W = 0 only if weighted).
# Usage: bash experiments/20261125_sbdifusco_rbsmall_labelvar/launch.sh <gpu> <set> <W> [T|none] [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; SET=$2; W=$3; T=${4:-none}; SEED=${5:-0}
DS=/usr0/home/shengyuf/yunyi/data/rb_small_sb
case "$SET" in
    single) LS=$DS/single_repair10_loaderorder.pt ;;
    mkamis) LS=$DS/mkamis_repair10_loaderorder.pt ;;
    k128) LS=/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/k128_loaderorder.pt ;;
    *) echo "unknown set $SET"; exit 1 ;;
esac
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
RUN=${EXP_NAME}_${SET}_w${W}${TAGT}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
D=/usr1/shengyuf/MIS
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 --main_process_port $((29700 + GPU)) main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_test \
    --problem mis --valid_samples 64 --seed "$SEED" --method DIFUSCO --model EncodeProcessDecode \
    --num_h 64 --num_l 8 --num_tp 50 --diffusion_schedule cosine --batch_size 40 \
    --optimizer AdamW --lr 0.0002 --epochs 400 --weight_decay 0.0001 --lr_scheduler cosine-decay \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule linear --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise "${LABEL[@]}" \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU"
