#!/bin/bash
# Flow / bridge matching from an MCMC start distribution on RB-[800-1200] MIS (UPSTREAM.md item 43).
# Unattended pipeline, user go-ahead 11-17 night:
#   1. wait until the KaMIS labels (data/rb_large_kamis/label.sh: test, valid, train) are written;
#   2. graph cache (CAM_GRAPH_CACHE) for the three splits;
#   3. RLSA start banks (scripts/mis/build_start_bank.py: 1000 steps, tau0 0.01, d 5, final chain states):
#      train 16 starts per graph, valid (first 64) and test (1000) 20 starts per graph = num_k;
#   4. two trainings in parallel, 2 GPUs each, 40 graphs per step, 200 epochs, no early stop:
#      ..._fm_..._lam0   = discrete flow matching, mixture path        (--fm_bridge_lambda 0)
#      ..._bm_..._lam0p5 = bridge matching, uniform-flip reference 0.5 (--fm_bridge_lambda 0.5)
#   5. test best.pt of both on the 1000 test graphs (50 steps x 20 chains, each chain from its own start).
# GPUs: each stage waits for idle GPUs (no compute process) among GPU_POOL.
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
DATA=/usr0/home/shengyuf/yunyi/data/rb_large_kamis      # graphs = symlinks to /usr1/shengyuf/MIS/RB_large_*
OUT="$REPO/runs/experiments/$EXP_NAME"; BANK="$OUT/banks"; mkdir -p "$BANK"
GPU_POOL=(${GPU_POOL:-0 1 2 3 4 5 6 7 8 9})
log() { echo "$(date -u '+%F %T') $*"; }

idle_gpus() {   # print up to $1 idle GPUs from the pool, waiting until that many are idle
    while :; do
        local got=()
        for g in "${GPU_POOL[@]}"; do
            [ "$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && got+=("$g")
            [ ${#got[@]} -ge "$1" ] && { echo "${got[*]}"; return; }
        done
        sleep 120
    done
}

# 1. labels
until grep -q "done train" "$DATA/label.log" 2>/dev/null; do sleep 300; done
log "labels: test $(ls $DATA/RB_large_test/anno | wc -l), valid $(ls $DATA/RB_large_valid/anno | wc -l), train $(ls $DATA/RB_large_train/anno | wc -l)"

# 2. graph cache
cd "$REPO/cam"
nice -n 10 "$CAM_PYTHON" -W ignore scripts/mis/build_graph_cache.py --workers 24 \
    "$DATA/RB_large_train" "$DATA/RB_large_valid" "$DATA/RB_large_test"
log "cache done"

# 3. start banks, 4 shards per split on 4 idle GPUs
bank() {   # split k limit
    local sp=$1 k=$2 lim=$3; local gs=($(idle_gpus 4)); local parts=()
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=${gs[$i]} "$CAM_PYTHON" -W ignore scripts/mis/build_start_bank.py \
            --graphs "$DATA/RB_large_$sp" --out "$BANK/${sp}_part$i.pt" --k $k --steps 1000 --tau0 0.01 --d 5 \
            --seed 0 --limit $lim --shard $i --num_shards 4 > "$BANK/${sp}_part$i.log" 2>&1 &
        parts+=("$BANK/${sp}_part$i.pt")
    done
    wait
    "$CAM_PYTHON" -W ignore scripts/mis/build_start_bank.py --merge "${parts[@]}" --out "$BANK/$sp.pt"
}
[ -f "$BANK/train.pt" ] || bank train 16 -1
[ -f "$BANK/valid.pt" ] || bank valid 20 64
[ -f "$BANK/test.pt" ]  || bank test 20 -1
log "banks done"

# 4. training
train() {   # run_name lambda gpu gpu
    local run=$1 lam=$2; local gs=($3 $4); local o="$OUT/$run"
    mkdir -p "$o"; log "train $run on GPUs ${gs[*]}"
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gs[*]}") "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=2 \
        --main_process_port $((29700 + ${gs[0]})) main.py \
        --train_path "$DATA/RB_large_train" --valid_path "$DATA/RB_large_valid" --test_path "$DATA/RB_large_test" \
        --problem mis --valid_samples 64 --seed 0 --method FM --model EncodeProcessDecode --num_h 64 --num_l 8 \
        --batch_size 20 --optimizer AdamW --lr 0.0002 --weight_decay 0.0001 --epochs 200 --patience 1000 \
        --save_every_epoch --use_activation_checkpoint --num_t 50 --num_k 20 --inference_schedule cosine --num_workers 4 \
        --fm_bank_train "$BANK/train.pt" --fm_bank_eval "$BANK/valid.pt" --fm_bridge_lambda $lam \
        --do_train --save_dir "$o" --wandb_name SLVUL --wandb_run_name "$run" > "$o/train.log" 2>&1
    log "train $run exit $?"
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gs[*]}") "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=2 \
        --main_process_port $((29700 + ${gs[0]})) main.py \
        --train_path "$DATA/RB_large_train" --valid_path "$DATA/RB_large_valid" --test_path "$DATA/RB_large_test" \
        --problem mis --seed 0 --method FM --model EncodeProcessDecode --num_h 64 --num_l 8 --batch_size 20 \
        --num_t 50 --num_k 20 --inference_schedule cosine --num_workers 4 \
        --fm_bank_eval "$BANK/test.pt" --fm_bridge_lambda $lam \
        --save_dir "$o" --log_file "$o/test_result.txt" --wandb_name SLVUL --wandb_run_name "$run" > "$o/test.log" 2>&1
    log "test $run exit $? $(grep 'Avg. size' "$o/test_result.txt" 2>/dev/null)"
}
G=($(idle_gpus 4))
train 20261118_fm_rblarge_mcmcstart_lam0_s0 0 ${G[0]} ${G[1]} &
train 20261118_bm_rblarge_mcmcstart_lam0p5_s0 0.5 ${G[2]} ${G[3]} &
wait
log "all done"
