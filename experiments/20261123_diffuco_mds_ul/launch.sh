#!/bin/bash
# DiffUCO on minimum dominating set, BA-[200-300] (ba) or RB-[200-300] graphs (rb): the unsupervised side of the MDS
# SL-vs-UL comparison and the label source of the DIFUSCO runs (20261123_difusco_mds_labels).
# Hyperparameters: DiffUCO paper (Sanokowski et al. 2024, App. hyperparameter table, row "BA-small MDS"):
# lr 0.003, T_start (tau0) 0.3, 8 GNN layers, 5 diffusion steps, annealed noise, M_omega 20 graphs, M_KL 10 samples.
# Stability settings and epoch budget as the tuned RB-small MIS DiffUCO run of the CAM paper
# (20260513_diffuco_rbsmall_annealing/launch_original.sh): RAdam, grad clip 1, cosine decay, 1000 epochs of 4000
# graphs, temperature annealed over 1000 epochs. No early stop (patience 2000) and a checkpoint every epoch: on MDS the
# validation size first rises while the temperature is high (best at epoch ~6, then early stop at ~210 with patience 200).
# The RB graphs are the MIS RB-small graphs (MDS has no RB results in the literature; same hyperparameters).
# Usage: bash experiments/20261123_diffuco_mds_ul/launch.sh <gpus, e.g. 0,1,2,3> <ba|rb> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPUS=$1; DS=$2; SEED=${3:-0}
NP=$(echo "$GPUS" | tr ',' '\n' | grep -c .); BS=$((20 / NP))
[ $((BS * NP)) = 20 ] || { echo "20 graphs per update must split evenly over $NP GPUs"; exit 1; }
case "$DS" in
    ba) D=/usr0/home/shengyuf/yunyi/data/CAM_data/MDS; TR=$D/BA_small_train; VA=$D/BA_small_valid; TE=$D/BA_small_test ;;
    rb) D=/usr1/shengyuf/MIS; TR=$D/RB_small_train; VA=$D/RB_small_valid; TE=$D/RB_small_test ;;
    *) echo "unknown dataset $DS"; exit 1 ;;
esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS
cam_require_gpu
RUN=${EXP_NAME}_${DS}small_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=$NP --num_machines=1 --dynamo_backend=no \
    --main_process_port $((29700 + ${GPUS%%,*})) main.py \
    --train_path $TR --valid_path $VA --test_path $TE --problem mds --seed "$SEED" \
    --samples_per_epoch 4000 --valid_samples 64 --batch_size $BS --mini_batch_size 1 \
    --num_t 50 --num_k 20 --num_tp 5 --num_kp 10 --num_h 64 --num_l 8 --tau0 0.3 \
    --method DiffUCO --model EncodeProcessDecode --noise_energy annealing \
    --optimizer RAdam --grad_clip 1 --lr 0.003 --lr_scheduler cosine-decay \
    --epochs 1000 --N_anneal 1000 --patience 2000 --save_every_epoch --num_workers 4 \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpus $GPUS"
