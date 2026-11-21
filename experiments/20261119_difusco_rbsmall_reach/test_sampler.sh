#!/bin/bash
# Test a trained run of this experiment with a given sampler on the 1000 RB_small_test graphs
# (50 steps x 20 chains, greedy decoding, best of 20; same protocol as test_after_train.sh). wandb off.
# Output: <run dir>/test_<sampler>_seed<N>{.log,_result.txt}.
# Usage: bash experiments/20261119_difusco_rbsmall_reach/test_sampler.sh <gpu> <run name> <sampler> <seed>
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; RUN=$2; SAMPLER=$3; SEED=$4
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"   # CAM_WANDB unset: wandb disabled
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; DATA=/usr1/shengyuf/MIS; TAG="test_${SAMPLER}_seed$SEED"
test -f "$OUT/best.pt" || { echo "no best.pt in $OUT"; exit 1; }
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU "$CAM_PYTHON" main.py \
    --train_path "$DATA/RB_small_train" --valid_path "$DATA/RB_small_valid" --test_path "$DATA/RB_small_test" \
    --problem mis --valid_samples 64 --seed $SEED --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 \
    --num_tp 50 --diffusion_schedule cosine --inference_schedule linear --batch_size 40 --num_t 50 --num_k 20 \
    --num_workers 4 --heat_lambda 0 --heat_until 0.5 --difusco_sampler $SAMPLER \
    --save_dir "$OUT" --log_file "$OUT/${TAG}_result.txt" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') $RUN $TAG $(grep -h 'Avg' "$OUT/${TAG}_result.txt" 2>/dev/null | tr '\n' ' ')"
