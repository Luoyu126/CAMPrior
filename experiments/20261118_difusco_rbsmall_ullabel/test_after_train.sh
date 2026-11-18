#!/bin/bash
# Wait for a training run of this experiment to finish, then test its best.pt on the 1000 RB_small_test
# graphs on the run's own two GPUs: {posterior, renoise} sampler x seeds {0, 1}, 50 cosine steps x 20
# chains, greedy decoding, best of 20 (same protocol as 20261117_difusco_rbsmall_heatflip/test.sh).
# wandb off. Output: <run dir>/test_<sampler>_seed<N>{.log,_result.txt}.
# Usage: bash experiments/20261118_difusco_rbsmall_ullabel/test_after_train.sh <run name> <train pid> <gpu> <gpu>
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
RUN=$1; PID=$2; G=($3 $4)
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"   # CAM_WANDB unset: wandb disabled
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; DATA=/usr1/shengyuf/MIS
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
test -f "$OUT/best.pt" || { echo "no best.pt in $OUT"; exit 1; }
tst() {   # gpu sampler seed
    local tag="test_$2_seed$3"
    (cd "$REPO/cam" && CUDA_VISIBLE_DEVICES=$1 "$CAM_PYTHON" main.py \
        --train_path "$DATA/RB_small_train" --valid_path "$DATA/RB_small_valid" --test_path "$DATA/RB_small_test" \
        --problem mis --valid_samples 64 --seed $3 --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 \
        --num_tp 1000 --diffusion_schedule linear --batch_size 40 --num_t 50 --num_k 20 --inference_schedule cosine \
        --num_workers 4 --heat_lambda 0 --heat_until 0.5 --difusco_sampler $2 \
        --save_dir "$OUT" --log_file "$OUT/${tag}_result.txt" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/$tag.log" 2>&1)
    echo "$(date -u '+%F %T') $RUN $tag $(grep 'Avg. size' "$OUT/${tag}_result.txt" 2>/dev/null)"
}
for s in 0 1; do tst ${G[0]} posterior $s & tst ${G[1]} renoise $s & wait; done
