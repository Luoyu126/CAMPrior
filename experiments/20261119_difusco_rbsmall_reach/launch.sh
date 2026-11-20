#!/bin/bash
# Reachable labels: labels built from a DIFUSCO student's own samples at the test step budget (50 steps), so that
# every label is reachable by the 50-step sampler (cf. the learnability diagnostic: the gap to a label is predicted
# by how often a few-step sampler reaches its size, Spearman 0.70, not by the one-step denoising loss, 0.18).
# Training as 20261119_difusco_rbsmall_fewstep tp50 (50 diffusion steps, cosine noise, linear 50-step test schedule).
# Labels (data/rb_small_reach/, runs/scratch/learnability_1119/build_reach.py) from round R's student samples
# (200 per training graph, 50 steps):
#   rR_self    medoid of the largest samples of each graph
#   rR_union   rR_self, or the DiffUCO k200 label where that one is strictly larger
#   rR_toprand all unique largest samples, one drawn uniformly per step
#   pS_dselM{1,4,16}  probe-selected DiffUCO labels: among each graph's largest DiffUCO samples (train_diffuco_all200.pt)
#              the M most predictable by a cross-fitted one-step probe (cam/scripts/mis/probe_select.py), one drawn
#              uniformly per step; pS_dselbot4 the 4 least predictable (control); pS_dtop all of them (control).
#   qR_dNall / qR_dNfpsM  diverse DiffUCO labels (cam/scripts/mis/diverse_labels.py): all unique largest of N DiffUCO
#              samples per graph, or M of them by farthest-point sampling in Hamming distance; _long: 400 epochs.
#   *_long     400 epochs instead of 200 (any label mode)
#   *_soft     the same label set, but the target is the posterior mean over the set given x_t (--label_choice soft,
#              cam/utils/label_sets.py posterior_mean): same optimum as random, lower-variance targets
#   *_ft       fine-tune the round-R student (seed s) instead of training from scratch (AdamW 5e-5, 30 epochs)
# Usage: bash experiments/20261119_difusco_rbsmall_reach/launch.sh <gpu> <rR_self|rR_union|rR_toprand>[_ft] [seed] [init ckpt for _ft]
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
GPU=$1; MODE=$2; SEED=${3:-0}; INITCK=${4:-}
L=/usr0/home/shengyuf/yunyi/data/rb_small_reach
R=${MODE%%_*}; KIND=${MODE#*_}; KIND=${KIND%_ft}
SOFT=0; if [[ "$KIND" == *_soft ]]; then SOFT=1; KIND=${KIND%_soft}; fi
LONG=0; if [[ "$KIND" == *_long ]]; then LONG=1; KIND=${KIND%_long}; fi
OPT=(--lr 0.0002 --epochs 200); INIT=()
case "$KIND" in
    self) LABEL=(--label_set "$L/${R}_self.pt" --label_choice first) ;;
    union) LABEL=(--label_set "$L/${R}_union.pt" --label_choice first) ;;
    toprand) LABEL=(--label_set "$L/${R}_top.pt" --label_choice random) ;;
    dselM1|dselM4|dselM16|dselbot4) P=/usr0/home/shengyuf/yunyi/data/rb_small_probe/dsel
        LABEL=(--label_set "${P}_${KIND#dsel}.pt" --label_choice random) ;;
    d[0-9]*all|d[0-9]*fps[0-9]*) F=${KIND/all/_all}; F=${F/fps/_fps}
        LABEL=(--label_set "/usr0/home/shengyuf/yunyi/data/rb_small_div/$F.pt" --label_choice random)
        ;;
    t1000tau*) LABEL=(--label_set "/usr0/home/shengyuf/yunyi/data/rb_small_typ/t1000_tau${KIND#t1000tau}.pt" --label_choice random) ;;
    s1000*) LABEL=(--label_set "/usr0/home/shengyuf/yunyi/data/rb_small_strong/s1000_${KIND#s1000}.pt" --label_choice random) ;;
    c1far*|c1near*) LABEL=(--label_set "/usr0/home/shengyuf/yunyi/data/rb_small_cov/c1_${KIND#c1}.pt" --label_choice random) ;;
    dtop) LABEL=(--label_set /usr0/home/shengyuf/yunyi/data/rb_small_ullabels/train_diffuco_all200_top.pt --label_choice random) ;;
    *) echo "unknown mode $MODE"; exit 1 ;;
esac
if [[ $LONG == 1 ]]; then OPT=(--lr 0.0002 --epochs 400); fi
if [[ $SOFT == 1 ]]; then LABEL=("${LABEL[@]/random/soft}"); fi
if [[ "$MODE" == *_ft ]]; then OPT=(--lr 0.00005 --epochs 30); INIT=(--checkpoint "${INITCK:?_ft needs an init checkpoint}"); fi
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh
export CAM_WANDB=1
source "$REPO/cam/scripts/env.sh"
export WANDB_DIR="$EXP_DIR" NCCL_IB_DISABLE=1 CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs CUDA_VISIBLE_DEVICES=$GPU
cam_require_gpu
RUN=${EXP_NAME}_${MODE}_s${SEED}; OUT="$REPO/runs/experiments/$EXP_NAME/$RUN"; mkdir -p "$OUT"
D=/usr1/shengyuf/MIS
cd "$REPO/cam"
nohup "$CAM_ACCELERATE" launch --mixed_precision=no --num_processes=1 --main_process_port $((29600 + GPU)) main.py \
    --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_test \
    --problem mis --valid_samples 64 --seed "$SEED" --method DIFUSCO --model EncodeProcessDecode \
    --num_h 64 --num_l 8 --num_tp 50 --diffusion_schedule cosine --batch_size 40 \
    --optimizer AdamW "${OPT[@]}" --weight_decay 0.0001 --lr_scheduler cosine-decay \
    --patience 1000 --save_every_epoch --num_t 50 --num_k 20 --inference_schedule linear --num_workers 4 \
    --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise "${LABEL[@]}" "${INIT[@]}" \
    --do_train --save_dir "$OUT" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train.log" 2>&1 &
echo "$RUN pid $! gpu $GPU"
