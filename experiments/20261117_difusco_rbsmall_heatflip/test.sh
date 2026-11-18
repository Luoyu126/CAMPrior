#!/bin/bash
# Test a best.pt of the 11-17 no-early-stop DIFUSCO runs on all 1000 RB_small_test graphs, the CAM-paper protocol
# (same as experiments/20261116_difusco_rbsmall_heatflip/test.sh: 50 cosine steps x 20 samples, greedy
# decoding, best of 20 per graph), for the 2 x 2 table {training run} x {sampler}:
#   ckpt     randflip  -> runs/experiments/20261117_difusco_rbsmall_randflip/..._s0/best.pt (uniform forward)
#            lam0      -> heatflip lambda 0   (uniform forward, best.pt chosen with the re-noising sampler)
#            lam0p1    -> heatflip lambda 0.1 (heated forward)
#   sampler  posterior -> D3PM posterior q(x_{t-1} | x_t, x0_hat) of the uniform kernel
#            renoise   -> draw x0 from the prediction, run the training forward from it to the next step
# wandb stays off. Output: <run dir>/test_<sampler>_seed<N>{.log,_result.txt}.
# Usage: bash experiments/20261117_difusco_rbsmall_heatflip/test.sh <gpu> <ckpt> <sampler> [test_seed] [ckpt_dir]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
USAGE="usage: test.sh <gpu> <randflip|lam0|lam0p1> <posterior|renoise> [test_seed] [ckpt_dir]"
GPU=${1:?$USAGE}
CKPT=${2:?$USAGE}
SAMPLER=${3:?$USAGE}
TEST_SEED=${4:-0}
CKPT_DIR=${5:-}   # optional: a directory holding a best.pt snapshot (e.g. taken mid-training); outputs go there

case "$CKPT" in
    randflip) RUN=20261117_difusco_rbsmall_randflip_s0; EXP=20261117_difusco_rbsmall_randflip; LAM=0 ;;
    lam0)     RUN=20261117_difusco_rbsmall_heatflip_lam0_s0; EXP=20261117_difusco_rbsmall_heatflip; LAM=0 ;;
    lam0p1)   RUN=20261117_difusco_rbsmall_heatflip_lam0p1_s0; EXP=20261117_difusco_rbsmall_heatflip; LAM=0.1 ;;
    *) echo "$USAGE"; exit 1 ;;
esac
case "$SAMPLER" in posterior|renoise) ;; *) echo "$USAGE"; exit 1 ;; esac

source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"   # CAM_WANDB unset: wandb disabled
export CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu

OUT="$REPO/runs/experiments/$EXP/$RUN"
[ -n "$CKPT_DIR" ] && OUT="$(cd "$CKPT_DIR" && pwd)"
test -f "$OUT/best.pt"
DATA=/usr1/shengyuf/MIS
TAG="test_${SAMPLER}_seed${TEST_SEED}"

cd "$REPO/cam"
nohup "$CAM_PYTHON" main.py \
    --train_path "$DATA/RB_small_train" \
    --valid_path "$DATA/RB_small_valid" \
    --test_path "$DATA/RB_small_test" \
    --problem mis \
    --valid_samples 64 \
    --seed "$TEST_SEED" \
    --method DIFUSCO \
    --model EncodeProcessDecode \
    --num_h 64 \
    --num_l 8 \
    --num_tp 1000 \
    --diffusion_schedule linear \
    --batch_size 40 \
    --optimizer AdamW \
    --lr 0.0002 \
    --weight_decay 0.0001 \
    --lr_scheduler cosine-decay \
    --epochs 200 \
    --patience 1000 \
    --num_t 50 \
    --num_k 20 \
    --inference_schedule cosine \
    --num_workers 4 \
    --heat_lambda "$LAM" \
    --heat_until 0.5 \
    --difusco_sampler "$SAMPLER" \
    --save_dir "$OUT" \
    --log_file "$OUT/${TAG}_result.txt" \
    --wandb_name SLVUL \
    --wandb_run_name "$RUN" \
    > "$OUT/${TAG}.log" 2>&1 &
echo "$CKPT $SAMPLER seed $TEST_SEED pid $! gpu $GPU log $OUT/${TAG}.log"
