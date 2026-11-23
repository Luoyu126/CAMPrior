#!/bin/bash
# DIFUSCO on minimum dominating set, BA-[200-300] (ba) or RB-[200-300] graphs (rb): the supervised side of the MDS
# SL-vs-UL comparison (UL side: 20261123_diffuco_mds_ul). Training exactly as the MIS RB-small DIFUSCO runs
# (20261119_difusco_rbsmall_reach): EncodeProcessDecode 8 x 64, 50 cosine diffusion steps, batch 40, AdamW 2e-4,
# cosine decay, wd 1e-4, 200 epochs; validation 64 graphs, 50 linear steps x 20 samples.
# Label modes (label sets under data/mds_labels/):
#   ilp1      one integer-programming optimum per graph (the baseline label, as KaMIS / Gurobi for MIS)
#   ilp8      up to 8 random optima per graph (BA only), one drawn per step   (+ _soft: posterior-mean target)
#   dif       all distinct smallest sets among DiffUCO samples per graph, one drawn per step (+ _soft)
# Sampler (validation, so it also selects best.pt): post = D3PM posterior (DIFUSCO), renoise (Fast T2T Alg. 1).
# Usage: bash experiments/20261123_difusco_mds_labels/launch.sh <gpu> <ba|rb> <label mode> <post|renoise> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; DS=$2; MODE=$3; SAMP=$4; SEED=${5:-0}
L=/usr0/home/shengyuf/yunyi/data/mds_labels
case "$DS" in
    ba) D=/usr0/home/shengyuf/yunyi/data/CAM_data/MDS; TR=$D/BA_small_train; VA=$D/BA_small_valid; TE=$D/BA_small_test ;;
    rb) D=/usr1/shengyuf/MIS; TR=$D/RB_small_train; VA=$D/RB_small_valid; TE=$D/RB_small_test ;;
    *) echo "unknown dataset $DS"; exit 1 ;;
esac
KIND=${MODE%_soft}; CHOICE=random; [[ "$MODE" == *_soft ]] && CHOICE=soft
case "$KIND" in
    ilp1) LABEL=(--label_set "$L/${DS}_small_train_ilp$([ $DS = ba ] && echo 8 || echo 1).pt" --label_choice first) ;;
    ilp8) [ $DS = ba ] || { echo "ilp8 only on ba"; exit 1; }; LABEL=(--label_set "$L/ba_small_train_ilp8.pt" --label_choice $CHOICE) ;;
    dif)  LABEL=(--label_set "$L/${DS}_small_train_diffuco.pt" --label_choice $CHOICE) ;;
    *) echo "unknown label mode $MODE"; exit 1 ;;
esac
case "$SAMP" in post) SAMPLER=posterior ;; renoise) SAMPLER=renoise ;; *) echo "unknown sampler $SAMP"; exit 1 ;; esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu
RUN=${EXP_NAME}_${DS}small_${MODE}_${SAMP}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 --main_process_port $((29800 + GPU)) main.py \
    --train_path $TR --valid_path $VA --test_path $TE --problem mds --valid_samples 64 --seed "$SEED" \
    --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 --num_tp 50 --diffusion_schedule cosine \
    --batch_size 40 --optimizer AdamW --lr 0.0002 --epochs 200 --weight_decay 0.0001 --lr_scheduler cosine-decay \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule linear --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler $SAMPLER "${LABEL[@]}" \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU"
