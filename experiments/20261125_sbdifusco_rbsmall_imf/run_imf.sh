#!/bin/bash
# Stage 1D of docs/20261125/plan.md: the learned coupling (math_route1.md §12.4), discrete-time IMF with two
# networks trained at the same time on caches written by each other ("cross-feeding"):
#   forward  SB-DIFUSCO (start z -> solution), trained on pairs (z from the backward network, y ~ pi1 true labels)
#   backward SB-DIFUSCO --sb_reverse (solution -> start), trained on pairs (z ~ pi0 true, y from the forward network)
# Iteration 0: the independent coupling (scripts/sb/imf_cache.py independent); the forward network starts from the
# independent-coupling baseline (20261119 q5_mkamisfix_long_soft_s0), the backward network from scratch, E0 epochs.
# Iterations 1..L: both networks sample J chains on every training graph (forward: from uniform starts; backward:
# from uniformly drawn labels), the traces become the other network's cache, both train E more epochs from their
# last epoch. Everything else as experiments/20261125_sbdifusco_rbsmall/launch.sh (50 steps, cosine, batch 40,
# AdamW 2e-4, soft targets; here the soft target is the exact posterior under the cached pairs).
# Reports per iteration: mean Hamming cost of both caches vs the closed-form coupling (plan.md 1D criterion 2).
# Usage: bash experiments/20261125_sbdifusco_rbsmall_imf/run_imf.sh <gpu fwd> <gpu bwd> <omega> [seed]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GF=$1; GB=$2; W=$3; SEED=${4:-0}
E0=${E0:-100}; E=${E:-50}; L=${L:-3}; J=${J:-16}
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
D=/usr1/shengyuf/MIS
LS=/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt
LOGG=/usr0/home/shengyuf/yunyi/data/rb_small_sb/logg_mkamisfix_w${W}_K4194304.pt
BASE="$REPO/runs/experiments/20261119_difusco_rbsmall_reach/20261119_difusco_rbsmall_reach_q5_mkamisfix_long_soft_s0/best.pt"
CUDA_VISIBLE_DEVICES=$GF cam_require_gpu || exit 1
CUDA_VISIBLE_DEVICES=$GB cam_require_gpu || exit 1
RUN=${EXP_NAME}_w${W}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT/caches"
COMMON=(--train_path $D/RB_small_train --valid_path $D/RB_small_valid --problem mis --seed "$SEED" --method DIFUSCO
        --model EncodeProcessDecode --num_h 64 --num_l 8 --num_tp 50 --diffusion_schedule cosine --num_t 50
        --inference_schedule linear --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise --num_workers 4
        --sb_coupling --sb_omega "$W" --label_choice soft --wandb_name SLVUL)
cd "$REPO/cam"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/imf.log"; }

train() {   # <gpu> <fwd|bwd> <iteration> <cache> <epochs> [checkpoint]
    local gpu=$1 net=$2 it=$3 cache=$4 ep=$5 ck=${6:-}
    local dir="$OUT/${net}_it$it"; mkdir -p "$dir"
    local extra=(); [ "$net" = bwd ] && extra+=(--sb_reverse --valid_samples 2) || extra+=(--valid_samples 64)
    [ -n "$ck" ] && extra+=(--checkpoint "$ck")
    CUDA_VISIBLE_DEVICES=$gpu "$CAM_PYTHON" main.py "${COMMON[@]}" --test_path $D/RB_small_test "${extra[@]}" \
        --sb_pair_cache "$cache" --batch_size 40 --optimizer AdamW --lr 0.0002 --weight_decay 0.0001 \
        --lr_scheduler cosine-decay --epochs "$ep" --patience 1000 --save_every_epoch --num_k 20 \
        --do_train --save_dir "$dir" --wandb_run_name "${RUN}_${net}_it$it" > "$dir/train.log" 2>&1
}

trace() {   # <gpu> <fwd|bwd> <iteration> -> $OUT/caches/trace_<net>_it<iteration>.jsonl
    local gpu=$1 net=$2 it=$3
    local dir="$OUT/${net}_it$((it - 1))" t="$OUT/caches/trace_${net}_it$it.jsonl"
    local ck="$dir/epoch_$(printf %04d $(( (it == 1 ? E0 : E) - 1 ))).pt"
    local extra=(); [ "$net" = bwd ] && extra+=(--sb_reverse)
    [ -f "$t" ] && mv "$t" "$t.old"
    env CAM_SB_PAIR_TRACE="$t" CAM_SB_TRACE_NOSHARE=1 CAM_SB_STARTS="$LS" CUDA_VISIBLE_DEVICES=$gpu \
        "$CAM_PYTHON" main.py "${COMMON[@]}" --test_path $D/RB_small_train "${extra[@]}" --batch_size 1 --num_k "$J" \
        --checkpoint "$ck" --save_dir "$dir" --log_file "$OUT/caches/trace_${net}_it${it}_result.txt" \
        --wandb_run_name "${RUN}_${net}_trace$it" > "$OUT/caches/trace_${net}_it$it.log" 2>&1
}

log "start $RUN: E0 $E0, E $E, L $L, J $J"
if [ ! -f "$OUT/caches/f0.pt" ]; then
    "$CAM_PYTHON" scripts/sb/imf_cache.py independent --label_set "$LS" --J "$J" --seed "$SEED" \
        --out_fwd "$OUT/caches/f0.pt" --out_bwd "$OUT/caches/b0.pt" --logg "$LOGG" >> "$OUT/imf.log" 2>&1
fi
for it in $(seq 0 "$L"); do
    if [ "$it" -gt 0 ]; then
        log "iteration $it: sampling"
        trace "$GF" fwd "$it" & trace "$GB" bwd "$it" & wait
        "$CAM_PYTHON" scripts/sb/imf_cache.py from_trace --trace "$OUT/caches/trace_fwd_it$it.jsonl" --direction fwd \
            --out "$OUT/caches/b$it.pt" --label_set "$LS" --logg "$LOGG" >> "$OUT/imf.log" 2>&1
        "$CAM_PYTHON" scripts/sb/imf_cache.py from_trace --trace "$OUT/caches/trace_bwd_it$it.jsonl" --direction bwd \
            --out "$OUT/caches/f$it.pt" --label_set "$LS" --logg "$LOGG" >> "$OUT/imf.log" 2>&1
        PF="$OUT/fwd_it$((it - 1))/epoch_$(printf %04d $(( (it == 1 ? E0 : E) - 1 ))).pt"
        PB="$OUT/bwd_it$((it - 1))/epoch_$(printf %04d $(( (it == 1 ? E0 : E) - 1 ))).pt"
        log "iteration $it: training"
        train "$GF" fwd "$it" "$OUT/caches/f$it.pt" "$E" "$PF" & train "$GB" bwd "$it" "$OUT/caches/b$it.pt" "$E" "$PB" & wait
    else
        log "iteration 0: training on the independent coupling"
        train "$GF" fwd 0 "$OUT/caches/f0.pt" "$E0" "$BASE" & train "$GB" bwd 0 "$OUT/caches/b0.pt" "$E0" & wait
    fi
    log "iteration $it done: fwd best $(grep -a 'eval/mis_size' "$OUT/fwd_it$it/train.log" | awk '{print $NF}' | sort -g | tail -1)"
done
log "done"
