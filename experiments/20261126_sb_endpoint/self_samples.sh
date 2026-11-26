#!/bin/bash
# Stage E (self-polished end distribution): 20 samples per TRAINING graph from a trained route-1 model (best.pt of <run
# dir>, renoise, 50 steps, sampler seed 1), via cam/scripts/mis/sl_samples.py; output {'files', 'samples'} for
# cam/scripts/sb/union_labels.py --ul. Training graphs only: nothing here touches validation or test graphs.
# Usage: bash experiments/20261126_sb_endpoint/self_samples.sh <gpu> <run dir> <omega> <out.pt>
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=$1; RUN=$2; W=$3; OUT=$4
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source "$REPO/cam/scripts/env.sh"
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs SL_SAMPLES_OUT="$OUT"
D=/usr1/shengyuf/MIS; mkdir -p "${OUT%.pt}_dir"
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 \
    --main_process_port $((29900 + GPU)) scripts/mis/sl_samples.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_train \
    --problem mis --valid_samples 64 --seed 1 --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 \
    --num_tp 50 --diffusion_schedule cosine --inference_schedule linear --batch_size 40 --num_t 50 --num_k 20 \
    --num_workers 4 --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise --sb_coupling --sb_omega "$W" \
    --checkpoint "$RUN/best.pt" --save_dir "${OUT%.pt}_dir" --log_file "${OUT%.pt}_result.txt" --wandb_name SLVUL --wandb_run_name self_samples
