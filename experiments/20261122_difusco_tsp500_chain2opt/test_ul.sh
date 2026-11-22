#!/bin/bash
# In-chain 2-opt for the released CAM TSP-500 checkpoint (UL flip sampler, --ul_chain_decode): after every step the
# edges are merged into a tour + DECODE 2-opt moves; final 2-opt 1000, 16 samples, 128 test instances. wandb off.
# DECODE=-1: plain CAM sampler (no in-chain decoding). CAM trains with 10 steps (num_tp 10); T = test steps.
# Output: runs/experiments/<this exp>/cam_ul/d<DECODE>_t<T>_seed<SEED>.{jsonl,log}
# Usage: bash experiments/20261122_difusco_tsp500_chain2opt/test_ul.sh <gpu> <decode> [T] [seed] [n]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; D2=$2; NT=${3:-10}; SEED=${4:-0}; N=${5:-128}
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"; export WANDB_MODE=disabled
CK=/usr0/home/shengyuf/yunyi/data/CAM_models/tsp_500_cam/cam/best.pt; T=/usr0/home/shengyuf/yunyi/data/CAM_data/TSP
OUT="$REPO/runs/experiments/$EXP_NAME/cam_ul"; mkdir -p "$OUT"
TAG=$([ "$D2" = -1 ] && echo plain || echo d$D2)_t${NT}_seed${SEED}$([ "$N" = 128 ] || echo _n$N)
FLAGS=$([ "$D2" = -1 ] && echo "--decode_two_opt 0" || echo "--ul_chain_decode --decode_two_opt $D2")
rm -f "$OUT/$TAG.jsonl"
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU PROBE_OUT="$OUT/$TAG.jsonl" "$CAM_PYTHON" "$EXP_DIR/probe.py" \
    --train_path $T/tsp500_test_concorde.txt --valid_path $T/tsp500_test_concorde.txt \
    --test_path $T/tsp500_test_concorde.txt --test_samples $N --problem tsp --seed $SEED \
    --sparse_factor 50 --num_t $NT --num_k 16 --num_tp 10 --num_kp 8 --num_h 256 --num_l 12 \
    --method CAM --model AnGNN --two_opt_iterations 1000 $FLAGS --checkpoint "$CK" \
    --save_dir "$OUT" --log_file "$OUT/${TAG}_result.txt" --wandb_name SLVUL --wandb_run_name cam_ul > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') cam_ul $TAG $(grep -h 'Avg. obj' "$OUT/${TAG}_result.txt" 2>/dev/null)"
