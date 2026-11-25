#!/bin/bash
# Stage 2B of docs/20261125/plan.md: cross-graph DASBS (cam/solvers/dasbs.py) on small random graphs (n = 8-12,
# data/sb_small from cam/scripts/sb/gen_small_graphs.py), temperature 1, MIS penalty beta = 2. CPU, 8 threads per run.
# 48 epochs x 125 iterations (16 graphs x 16 chains each, 50 tau-leaping steps on the grid denser near t = 1,
# 4 gradient steps per rollout, 4 bridge points per pair) = 6000 iterations = 20 stages of 300 iterations.
# Modes:
#   sb       log-linear reference gamma 1, alpha 0.5 (not memoryless), stages of 200 controller + 100 corrector its
#   ml       memoryless log-linear (alpha 0), no corrector (300 controller its per stage): no SB, adjoint matching only
#   nocorr   reference of sb but no corrector: wrong fixed point (math_route2.md §6 table), control
# Diagnosis round 1 (status.md 11-25 13:30): validation TV oscillated strongly along training for all three modes;
# environment knobs for the fix (defaults = the original runs):
#   EMA=d          parameter EMA (--dasbs_ema d; checkpoints hold the averaged weights), as DASBS App. C always does
#   SB_STAGE=C,R   stage split of mode sb (default 200,100)
#   EPOCHS=E       epochs (default 48; 125 iterations each)
#   TAG=x          run name suffix (_x)
# Usage: bash experiments/20261125_dasbs_crossgraph/launch.sh <mode> [seed]
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
MODE=$1; SEED=${2:-0}
case $MODE in
  sb) REF="--dasbs_alpha 0.5 --dasbs_stage ${SB_STAGE:-200,100}" ;;
  ml) REF="--dasbs_alpha 0 --dasbs_stage 300,0" ;;
  nocorr) REF="--dasbs_alpha 0.5 --dasbs_stage 300,0" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
EPOCHS=${EPOCHS:-48}
[ -n "${EMA:-}" ] && REF="$REF --dasbs_ema $EMA"
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
source "$REPO/cam/scripts/env.sh"     # CAM_WANDB unset: wandb disabled
export WANDB_MODE=disabled OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
D=/usr0/home/shengyuf/yunyi/data/sb_small
NAME=20261125_dasbs_crossgraph_${MODE}_s$SEED${TAG:+_$TAG}
OUT=/usr0/home/shengyuf/yunyi/Schrödinger\ Bridge/CAMPrior/runs/experiments/20261125_dasbs_crossgraph/$NAME
mkdir -p "$OUT"
cd "$REPO/cam" && nohup "$CAM_PYTHON" main.py --train_path $D/train --valid_path $D/valid --test_path $D/test \
  --problem mis --valid_samples 16 --seed $SEED --method DASBS --model EncodeProcessDecodeTime2 --num_h 64 --num_l 8 \
  --num_tp 50 --num_kp 16 --num_t 50 --num_k 16 --batch_size 16 --mini_batch_size 16 --inner_loop_steps 4 \
  --samples_per_epoch 2000 --epochs $EPOCHS --N_anneal $EPOCHS --patience 1000 --tau0 1 --tau1 1 --beta 2 --lr 1e-3 \
  --dasbs_schedule log_linear --dasbs_gamma 1 --dasbs_replicas 4 --dasbs_corrector_time zero --dasbs_grid quadratic \
  $REF --num_workers 2 --save_every_epoch --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name $NAME \
  > "$OUT/train.log" 2>&1 &
echo "$NAME pid $!"
