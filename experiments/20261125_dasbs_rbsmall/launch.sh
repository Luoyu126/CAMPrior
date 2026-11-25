#!/bin/bash
# Stage 2C of docs/20261125/plan.md: cross-graph DASBS (cam/solvers/dasbs.py) on RB-small MIS. One GPU per run.
# Aligned with DiffUCO RB-small (experiments/20260513_diffuco_rbsmall_annealing): EncodeProcessDecode width 64,
# 8 layers (here the time-input variant EncodeProcessDecodeTime2 for the controller), 40 graphs per parameter update
# (there 4 per GPU x 10 GPUs), 10 chains per graph, one update per rollout (100 updates per epoch of 4000 graphs),
# RAdam lr 1e-3, gradient clipping 1, cosine decay, penalty beta 1.2, validation on the first 64 validation graphs.
# Differences, with the reason (docs/20261125/status.md, 2C):
#   - temperature (round 2, default): geometric (TEMP=geometric) from TAU0 = 20 to TAU1 = 0.05 over N_ANNEAL epochs,
#     then constant. TAU0 = 20 is the mean |dH| at Bernoulli(0.5) states on RB-small train graphs (21.0; median 20.6,
#     p90 33.8, beta 1.2), so the ctrl-AM ratios exp(-dH / tau) are ~e^(+-1) at the start. Round 0/1 (TEMP=linear
#     TAU0=1): at tau = 1 the ratios are already up to e^28 and on-policy training settled in a degenerate fixed point
#     (every flip probability 1, end states stay Bernoulli(0.5); docs/20261125/status.md, 2C round 2).
#     Earlier note: linear from TAU0 = 1 to TAU1 = 0.05 over N_ANNEAL epochs, then constant. DiffUCO anneals 0.1 -> 0;
#     here the regression target is exp(-dH / tau) and with random early rollouts on RB-small (degree up to ~70,
#     -dH up to ~80) every controller batch overflows at tau = 0.1 (CPU check, 16 of 16 batches), so training must
#     start hot; tau = 0 is not defined for this target.
#   - training rollouts: NUM_TP tau-leaping steps (default 10; DiffUCO trains with 6 steps of a time-free flip kernel);
#     evaluation 50 steps (and 10). With 50 training steps a parameter update took 4.1 s (~46 GPU-h for 400 epochs,
#     over the 24 GPU-h cap of plan.md 2C): the rollouts are ~80% of the cost.
#   - REPLICAS bridge time points per (x0, x1) pair (default 2): with 4, the controller pass over 10 chains x 4
#     replicas of a 40-graph batch ran out of the 10.5 GB GPU memory. Only the variance of the loss changes.
#     Every mini-batch is one optimizer step (cam/solvers/meta_solver.py train), so the mini-batch stays at 10 chains:
#     one update per rollout, 100 per epoch.
#   - controller output scaled by 1 / tau, loss terms divided by max(1, Phi), parameter EMA 0.999 for evaluation
#     (cam/solvers/dasbs.py docstring).
#   - OVERFLOW (default rescale): controller batches whose per-element gradient is not representable in float32 are
#     rescaled by one common constant (--dasbs_rescale_overflow; with --grad_clip 1 the step is the exact one).
#     OVERFLOW=skip (--dasbs_skip_overflow) was round 0 (runs renamed *_round0): from epoch ~60 (tau ~0.8) nearly every
#     batch was skipped and training stopped (docs/20261125/status.md, 2C round 1).
#   - TAG: optional run-name suffix (diagnostic runs).
# Modes:
#   sb_a<A>   log-linear reference gamma 1, alpha A (signal at t = 1: A / (1 + A)), with corrector, stages of 200
#             controller + 100 corrector iterations (the full SB)
#   ml        memoryless log-linear (alpha 0), no corrector, 300 controller iterations per stage (discrete adjoint
#             matching without SB)
# Usage: bash experiments/20261125_dasbs_rbsmall/launch.sh <gpu> <mode> [seed]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; MODE=$2; SEED=${3:-0}; OVERFLOW=${OVERFLOW:-rescale}
EPOCHS=${EPOCHS:-400}; NUM_TP=${NUM_TP:-10}; REPLICAS=${REPLICAS:-2}; N_ANNEAL=${N_ANNEAL:-300}; TAU0=${TAU0:-20}; TAU1=${TAU1:-0.05}; TEMP=${TEMP:-geometric}
case "$MODE" in
    sb_a*) A=${MODE#sb_a}; REF=(--dasbs_alpha "$A" --dasbs_stage 200,100) ;;
    ml) REF=(--dasbs_alpha 0 --dasbs_stage 300,0) ;;
    *) echo "unknown mode $MODE"; exit 1 ;;
esac
case "$OVERFLOW" in rescale|skip) ;; *) echo "OVERFLOW must be rescale or skip"; exit 1 ;; esac
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu
RUN=${EXP_NAME}_${MODE}_s${SEED}${TAG:+_$TAG}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
D=/usr1/shengyuf/MIS
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 --main_process_port $((29650 + GPU)) main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_test \
    --problem mis --valid_samples 64 --seed "$SEED" --method DASBS --model EncodeProcessDecodeTime2 \
    --num_h 64 --num_l 8 --samples_per_epoch 4000 --batch_size 40 --num_kp 10 --mini_batch_size 10 \
    --inner_loop_steps 1 --num_tp $NUM_TP --num_t 50 --num_k 20 --num_workers 4 \
    --optimizer RAdam --lr 0.001 --grad_clip 1 --lr_scheduler cosine-decay --beta 1.2 \
    --epochs $EPOCHS --N_anneal $N_ANNEAL --temp_schedule $TEMP --tau0 $TAU0 --tau1 $TAU1 --patience 1000 --save_every_epoch \
    --dasbs_schedule log_linear --dasbs_gamma 1 "${REF[@]}" --dasbs_replicas $REPLICAS --dasbs_corrector_time zero \
    --dasbs_grid quadratic --dasbs_ctrl_scale tau --dasbs_loss_norm pred --dasbs_${OVERFLOW}_overflow --dasbs_ema 0.999 \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU"
