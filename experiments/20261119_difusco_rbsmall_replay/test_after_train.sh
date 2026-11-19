#!/bin/bash
# Wait for a fine-tuning run, then test its last checkpoint on the 1000 RB_small_test graphs
# ({posterior, renoise} x seeds {0, 1}, 50 cosine steps x 20 chains, best of 20; same protocol as
# 20261118_difusco_rbsmall_ullabel) and, renoise seed 0, on the 53 graphs with improved labels
# (data/rb_small_replay/train_near10). wandb off.
# Usage: bash experiments/20261119_difusco_rbsmall_replay/test_after_train.sh <run name> <train pid> <gpu>
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
RUN=$1; PID=$2; GPU=$3
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; DATA=/usr1/shengyuf/MIS
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
CK="${CKPT:-$(ls "$OUT"/epoch_0*.pt | sort | tail -1)}"; test -f "$CK" || { echo "no $CK"; exit 1; }
tst() {   # test_path sampler seed tag
    (cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$GPU "$CAM_PYTHON" main.py \
        --train_path "$DATA/RB_small_train" --valid_path "$DATA/RB_small_valid" --test_path "$1" \
        --problem mis --valid_samples 64 --seed $3 --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 \
        --num_tp 1000 --diffusion_schedule linear --batch_size 40 --num_t 50 --num_k 20 --inference_schedule cosine \
        --num_workers 4 --heat_lambda 0 --heat_until 0.5 --difusco_sampler $2 --checkpoint "$CK" \
        --save_dir "$OUT" --log_file "$OUT/${4}_result.txt" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/$4.log" 2>&1)
    echo "$(TZ=America/New_York date '+%F %T') $RUN $4 $(grep 'Avg. size' "$OUT/${4}_result.txt" 2>/dev/null)"
}
for s in 0 1; do for smp in posterior renoise; do tst "$DATA/RB_small_test" $smp $s test_${smp}_seed$s; done; done
tst /usr0/home/shengyuf/yunyi/data/rb_small_replay/train_near10 renoise 0 near10_renoise_seed0
