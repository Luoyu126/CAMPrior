#!/bin/bash
# Fine-tune the round-1 DiffUCO-label DIFUSCO (20261118_difusco_rbsmall_ullabel diffuco_s<seed>/best.pt)
# on all 4000 training graphs, where the 53 graphs whose DiffUCO label has a larger independent set within
# Hamming 10 (warm-started RLSA, nearest +1) get that set as label and appear REP times (oversampled).
# Control: same oversampling, original labels. AdamW 5e-5 cosine over 20 epochs, 1 GPU, batch 40.
# near10_improved / near10_original: only those 53 graphs (train_near10), 100 epochs (2 steps each), validation
# on 8 graphs (the last epoch is tested, not the best).
# Usage: bash experiments/20261119_difusco_rbsmall_replay/launch.sh <gpu> <rep3_improved|rep10_improved|rep10_original|near10_improved|near10_original> <seed 0|1>
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; MODE=$2; SEED=$3; REP=${MODE%%_*}
EPOCHS=20; VALID=64; TRAIN=train_$REP
if [ "$REP" = near10 ]; then EPOCHS=100; VALID=8; TRAIN=train_near10; fi
D=/usr0/home/shengyuf/yunyi/data/rb_small_replay
CKPT="$REPO/runs/experiments/20261118_difusco_rbsmall_ullabel/20261118_difusco_rbsmall_ullabel_diffuco_s$SEED/best.pt"
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
export CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu
RUN=${EXP_NAME}_${MODE}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 --main_process_port $((29500 + GPU)) main.py \
    --train_path "$D/$TRAIN" --valid_path /usr1/shengyuf/MIS/RB_small_valid --test_path /usr1/shengyuf/MIS/RB_small_test \
    --problem mis --valid_samples $VALID --seed "$SEED" --method DIFUSCO --model EncodeProcessDecode \
    --num_h 64 --num_l 8 --num_tp 1000 --diffusion_schedule linear --batch_size 40 \
    --optimizer AdamW --lr 0.00005 --weight_decay 0.0001 --lr_scheduler cosine-decay --epochs $EPOCHS \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule cosine --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise --label_set "$D/$MODE.pt" --label_choice first \
    --checkpoint "$CKPT" --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU"
