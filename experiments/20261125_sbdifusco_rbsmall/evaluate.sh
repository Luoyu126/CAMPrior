#!/bin/bash
# Evaluate best.pt of a run with the test protocol (num_t steps x 20 chains, greedy decoding, best of 20) on either
# the validation anchor (validation graphs 64-499 in loader order, 436 graphs; data/rb_small_valanchor holds symlinks)
# or the 1000 test graphs. Selection decisions use the anchor only. wandb off.
# Output: <run dir>/<set>_<sampler>_t<steps>_seed<N>{.log,_result.txt,_result.txt.per_graph.csv}.
# Usage: bash experiments/20261125_sbdifusco_rbsmall/evaluate.sh <gpu> <run dir> <anchor|test> <sampler> <steps> <seed> [extra flags]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; OUT=$2; SET=$3; SAMPLER=$4; STEPS=$5; SEED=$6; shift 6; EXTRA=("$@")
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"   # CAM_WANDB unset: wandb disabled
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
D=/usr1/shengyuf/MIS
case "$SET" in anchor) TP=/usr0/home/shengyuf/yunyi/data/rb_small_valanchor ;; test) TP=$D/RB_small_test ;; *) exit 1 ;; esac
TAG="${SET}_${SAMPLER}_t${STEPS}_seed$SEED"
test -f "$OUT/best.pt" || { echo "no best.pt in $OUT"; exit 1; }
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU "$CAM_PYTHON" main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path "$TP" \
    --problem mis --valid_samples 64 --seed $SEED --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 \
    --num_tp 50 --diffusion_schedule cosine --inference_schedule linear --batch_size 40 --num_t $STEPS --num_k 20 \
    --num_workers 4 --heat_lambda 0 --heat_until 0.5 --difusco_sampler $SAMPLER "${EXTRA[@]}" \
    --save_dir "$OUT" --log_file "$OUT/${TAG}_result.txt" --wandb_name SLVUL --wandb_run_name "$(basename "$OUT")" > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') $(basename "$OUT") $TAG $(grep -h 'Avg. size' "$OUT/${TAG}_result.txt" 2>/dev/null)"
