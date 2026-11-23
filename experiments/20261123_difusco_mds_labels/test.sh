#!/bin/bash
# Test a trained MDS DIFUSCO run on the 1000 test graphs: 50 linear steps x 20 samples, greedy decode, best of 20.
# Sampler: posterior (D3PM, DIFUSCO) | renoise (Fast T2T Alg. 1) | renoise_decode (decode x0 before renoising).
# Output: <run dir>/test_<sampler>_seed<N>_result.txt (+ .per_graph.csv). wandb off.
# Usage: bash experiments/20261123_difusco_mds_labels/test.sh <gpu> <run name> <sampler> [seed] [ckpt]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; RUN=$2; SAMPLER=$3; SEED=${4:-0}; CK=${5:-best.pt}
case "$RUN" in
    *basmall*) D=/usr0/home/shengyuf/yunyi/data/CAM_data/MDS; TE=$D/BA_small_test ;;
    *rbsmall*) D=/usr1/shengyuf/MIS; TE=$D/RB_small_test ;;
esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"; export WANDB_MODE=disabled
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; TAG=test_${SAMPLER}_seed${SEED}${CK:+_${CK%.pt}}; TAG=${TAG%_best}
test -f "$OUT/$CK" || { echo "no $CK in $OUT"; exit 1; }
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU "$CAM_PYTHON" main.py \
    --train_path $TE --valid_path $TE --test_path $TE --problem mds --seed $SEED \
    --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 --num_tp 50 --diffusion_schedule cosine \
    --num_t 50 --num_k 20 --inference_schedule linear --heat_lambda 0 --heat_until 0.5 --difusco_sampler $SAMPLER \
    --num_workers 2 --checkpoint "$OUT/$CK" --save_dir "$OUT" --log_file "$OUT/${TAG}_result.txt" \
    --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') $RUN $TAG $(grep -h 'Avg. obj' "$OUT/${TAG}_result.txt" 2>/dev/null)"
