#!/bin/bash
# Test a trained run on the 128 TSP-500 test instances: 50 steps, 16 samples, greedy merge, best of 16, with
# TWO_OPT iterations of 2-opt (0 = none; 1000 as DIFUSCO / CAM). wandb off.
# Output: <run dir>/test_<sampler>_2opt<TWO_OPT>_seed<N>{.log,_result.txt} (+ .per_graph.csv)
# env NUM_T=N inference steps (default 50; tag gets _tN otherwise).
# Sampler renoise_decode: env DECODE_2OPT=N iterations of 2-opt on each decoded x0 (default 0); tag gets _d<N>.
# Usage: bash experiments/20261121_difusco_tsp500_labels/test.sh <gpu> <run name> <sampler> <two_opt> [seed] [ckpt]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; RUN=$2; SAMPLER=$3; TWO=$4; SEED=${5:-0}; CK=${6:-best.pt}; D2=${DECODE_2OPT:-0}; NT=${NUM_T:-50}
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"   # CAM_WANDB unset: wandb disabled
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; T=/usr0/home/shengyuf/yunyi/data/CAM_data/TSP
TAG="test_${SAMPLER}$([ "$SAMPLER" = renoise_decode ] && echo _d$D2)_2opt${TWO}$([ "$NT" = 50 ] || echo _t$NT)_seed$SEED${CK:+_${CK%.pt}}"; TAG=${TAG%_best}
test -f "$OUT/$CK" || { echo "no $CK in $OUT"; exit 1; }
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU "$CAM_PYTHON" main.py \
    --train_path $T/tsp500_valid_concorde.txt --valid_path $T/tsp500_valid_concorde.txt \
    --test_path $T/tsp500_test_concorde.txt --test_samples 128 --problem tsp --seed $SEED \
    --method DIFUSCO --model AnGNN --num_h 256 --num_l 12 --sparse_factor 50 \
    --num_tp 50 --diffusion_schedule cosine --inference_schedule linear --num_t $NT --num_k 16 \
    --two_opt_iterations $TWO --difusco_sampler $SAMPLER --decode_two_opt $D2 --num_workers 2 --checkpoint "$OUT/$CK" \
    --save_dir "$OUT" --log_file "$OUT/${TAG}_result.txt" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') $RUN $TAG $(grep -h 'Avg' "$OUT/${TAG}_result.txt" 2>/dev/null | tr '\n' ' ')"
