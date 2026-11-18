#!/bin/bash
# DIFUSCO baseline on RB-[800-1200] MIS for 20261118_fm_rblarge_mcmcstart (user go-ahead 11-17 night).
# Same network and training budget as the FM / BM runs there: EncodeProcessDecode 8 x 64, 40 graphs per
# step (2 GPUs x 20), AdamW 2e-4, 200 epochs, no early stop, a checkpoint per epoch, same local KaMIS labels.
# Uniform forward, best.pt picked with the re-noising sampler (as 20261117_difusco_rbsmall_heatflip lam0).
#   1. wait for the FM pipeline's graph cache (it waits for the KaMIS labels first);
#   2. train on 2 idle GPUs (pool searched from GPU 9 down, the FM pipeline takes the low ones);
#   3. test best.pt on the 1000 test graphs: {posterior, renoise} x seeds {0, 1}, 50 cosine steps x 20 chains.
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
DATA=/usr0/home/shengyuf/yunyi/data/rb_large_kamis
FM_LOG="$REPO/runs/experiments/20261118_fm_rblarge_mcmcstart/pipeline.log"
RUN=${EXP_NAME}_s0
OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
GPU_POOL=(${GPU_POOL:-9 8 7 6 5 4 3 2 1 0})
log() { echo "$(date -u '+%F %T') $*"; }
idle_gpus() {
    while :; do
        local got=()
        for g in "${GPU_POOL[@]}"; do
            [ "$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && got+=("$g")
            [ ${#got[@]} -ge "$1" ] && { echo "${got[*]}"; return; }
        done
        sleep 120
    done
}
args() {   # common flags
    echo --train_path "$DATA/RB_large_train" --valid_path "$DATA/RB_large_valid" --test_path "$DATA/RB_large_test" \
        --problem mis --valid_samples 64 --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 \
        --num_tp 1000 --diffusion_schedule linear --optimizer AdamW --lr 0.0002 --weight_decay 0.0001 \
        --lr_scheduler cosine-decay --epochs 200 --patience 1000 --num_t 50 --num_k 20 --inference_schedule cosine \
        --num_workers 4 --heat_lambda 0 --wandb_name SLVUL --wandb_run_name "$RUN"
}

until grep -q "cache done" "$FM_LOG" 2>/dev/null; do sleep 300; done
G=($(idle_gpus 2)); log "train $RUN on GPUs ${G[*]}"
cd "$REPO/cam"
CUDA_VISIBLE_DEVICES=${G[0]},${G[1]} "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=2 \
    --main_process_port $((29800 + ${G[0]})) main.py $(args) --seed 0 --batch_size 20 --save_every_epoch \
    --use_activation_checkpoint --difusco_sampler renoise --do_train --save_dir "$OUT" > "$OUT/train.log" 2>&1
log "train exit $?"

export WANDB_MODE=disabled   # tests stay off wandb, as the RB-small tests
tst() {   # gpu sampler seed
    local tag="test_$2_seed$3"
    CUDA_VISIBLE_DEVICES=$1 "$CAM_PYTHON" main.py $(args) --seed $3 --batch_size 20 --difusco_sampler $2 \
        --save_dir "$OUT" --log_file "$OUT/${tag}_result.txt" > "$OUT/$tag.log" 2>&1
    log "$tag exit $? $(grep 'Avg. size' "$OUT/${tag}_result.txt" 2>/dev/null)"
}
for s in 0 1; do
    tst ${G[0]} posterior $s & tst ${G[1]} renoise $s & wait
done
log "all done"
