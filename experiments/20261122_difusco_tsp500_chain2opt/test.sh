#!/bin/bash
# In-chain 2-opt sweep on TSP-500: renoise_decode with DECODE 2-opt moves on each step's decoded x0,
# final 2-opt 1000, 16 samples, 128 test instances, a model trained in 20261121_difusco_tsp500_labels. wandb off.
# Output: runs/experiments/<this exp>/<model run>/d<DECODE>_t<NUM_T>_seed<SEED>.{jsonl,log}
# Usage: bash experiments/20261122_difusco_tsp500_chain2opt/test.sh <gpu> <model run> <decode> [num_t] [seed] [n]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; RUN=$2; D2=$3; NT=${4:-50}; SEED=${5:-0}; N=${6:-128}
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"; export WANDB_MODE=disabled
CK="$REPO/runs/experiments/20261121_difusco_tsp500_labels/$RUN/best.pt"; T=/usr0/home/shengyuf/yunyi/data/CAM_data/TSP
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"; TAG=d${D2}_t${NT}_seed${SEED}$([ "$N" = 128 ] || echo _n$N)
rm -f "$OUT/$TAG.jsonl"
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU PROBE_OUT="$OUT/$TAG.jsonl" "$CAM_PYTHON" "$EXP_DIR/probe.py" \
    --train_path $T/tsp500_valid_concorde.txt --valid_path $T/tsp500_valid_concorde.txt \
    --test_path $T/tsp500_test_concorde.txt --test_samples $N --problem tsp --seed $SEED \
    --method DIFUSCO --model AnGNN --num_h 256 --num_l 12 --sparse_factor 50 \
    --num_tp 50 --diffusion_schedule cosine --inference_schedule linear --num_t $NT --num_k 16 \
    --two_opt_iterations 1000 --difusco_sampler renoise_decode --decode_two_opt $D2 --num_workers 2 --checkpoint "$CK" \
    --save_dir "$OUT" --log_file "$OUT/${TAG}_result.txt" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') $RUN $TAG $(grep -h 'Avg. obj' "$OUT/${TAG}_result.txt" 2>/dev/null)"
