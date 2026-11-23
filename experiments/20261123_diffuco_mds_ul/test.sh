#!/bin/bash
# Test a trained MDS DiffUCO run on the 1000 test graphs: 50 flip steps x 20 samples (as the DIFUSCO tests),
# greedy decode, best of 20. Default checkpoint: the last epoch (the temperature is annealed to 0 there; best.pt
# is chosen on validation over the whole run and can be an early high-temperature epoch).
# Output: <run dir>/test_<ckpt>_seed<N>_result.txt (+ .per_graph.csv). wandb off.
# Usage: bash experiments/20261123_diffuco_mds_ul/test.sh <gpu> <ba|rb> [ckpt, default epoch_0999.pt] [seed]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; DS=$2; CK=${3:-epoch_0999.pt}; SEED=${4:-0}
case "$DS" in
    ba) TE=/usr0/home/shengyuf/yunyi/data/CAM_data/MDS/BA_small_test ;;
    rb) TE=/usr1/shengyuf/MIS/RB_small_test ;;
    *) echo "unknown dataset $DS"; exit 1 ;;
esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"; export WANDB_MODE=disabled
RUN=${EXP_NAME}_${DS}small_s0; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; TAG=test_${CK%.pt}_seed${SEED}
test -f "$OUT/$CK" || { echo "no $CK in $OUT"; exit 1; }
cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU "$CAM_PYTHON" main.py \
    --train_path $TE --valid_path $TE --test_path $TE --problem mds --seed $SEED \
    --method DiffUCO --model EncodeProcessDecode --num_h 64 --num_l 8 --num_t 50 --num_k 20 --num_tp 5 \
    --noise_energy annealing --num_workers 2 --checkpoint "$OUT/$CK" --save_dir "$OUT" \
    --log_file "$OUT/${TAG}_result.txt" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/$TAG.log" 2>&1
echo "$(date '+%F %T') $RUN $TAG $(grep -h 'Avg. obj' "$OUT/${TAG}_result.txt" 2>/dev/null)"
