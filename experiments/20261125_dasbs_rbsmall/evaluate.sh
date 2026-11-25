#!/bin/bash
# Evaluate best.pt (the EMA weights) of a 2C run with the test protocol (num_t tau-leaping steps x 20 chains, greedy
# decoding of the last step's probabilities, best of 20) on the validation anchor (validation graphs 64-499 in loader
# order, data/rb_small_valanchor) or the 1000 test graphs. Selection decisions use the anchor only. wandb off.
# Also writes the sampler's raw end states per graph (CAM_DASBS_RAW): share without violated edges, raw size,
# decoded mean / best.
# Output: <run dir>/<set>_t<steps>_seed<N>{.log,_result.txt,_result.txt.per_graph.csv,_raw.jsonl}.
# CKPT=epoch_0099 in the environment evaluates that checkpoint instead of best.pt (training-efficiency curve, plan.md
# 2C); its outputs carry the suffix _<CKPT>.
# Usage: bash experiments/20261125_dasbs_rbsmall/evaluate.sh <gpu> <run dir> <mode as in launch.sh> <anchor|test> <steps> <seed>
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; OUT=$2; MODE=$3; SET=$4; STEPS=$5; SEED=$6
case "$MODE" in sb_a*) A=${MODE#sb_a} ;; ml) A=0 ;; *) echo "unknown mode $MODE"; exit 1 ;; esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"   # CAM_WANDB unset: wandb disabled
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
D=/usr1/shengyuf/MIS
case "$SET" in anchor) TP=/usr0/home/shengyuf/yunyi/data/rb_small_valanchor ;; test) TP=$D/RB_small_test ;; *) exit 1 ;; esac
TAG="${SET}_t${STEPS}_seed$SEED${CKPT:+_$CKPT}"
CK="$OUT/${CKPT:-best}.pt"
test -f "$CK" || { echo "no $CK"; exit 1; }
rm -f "${OUT:?}/${TAG:?}_raw.jsonl"
cd "$REPO/cam" && CAM_DASBS_RAW="$OUT/${TAG}_raw.jsonl" CUDA_VISIBLE_DEVICES=$GPU "$CAM_PYTHON" main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path "$TP" \
    --problem mis --seed $SEED --method DASBS --model EncodeProcessDecodeTime2 --num_h 64 --num_l 8 \
    --num_t $STEPS --num_k 20 --batch_size 1 --num_workers 4 --beta 1.2 \
    --dasbs_schedule log_linear --dasbs_gamma 1 --dasbs_alpha $A --dasbs_grid quadratic --dasbs_ctrl_scale tau --checkpoint "$CK" \
    --save_dir "$OUT" --log_file "$OUT/${TAG}_result.txt" --wandb_name SLVUL --wandb_run_name "$(basename "$OUT")" \
    > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') $(basename "$OUT") $TAG $(grep -h 'Avg. size' "$OUT/${TAG}_result.txt" 2>/dev/null)"
