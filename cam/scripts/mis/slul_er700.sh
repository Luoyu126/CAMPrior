#!/bin/bash
# Controlled SL vs UL comparison on ER-[700-800] with the small GNN (local addition, see
# cam/UPSTREAM.md and docs/20261116/handoff.md). Follows the unified setup of the CAM paper,
# appendix B.2: EncodeProcessDecode with 8 layers and width 64 (no time input), the same 16,284
# training graphs (SL also reads the KaMIS labels in er_train/anno), 50 inference steps and 20
# samples. Only the learning method differs:
#   METHOD=DIFUSCO  supervised discrete diffusion (cross-entropy to the KaMIS label)
#   METHOD=DiffUCO  unsupervised discrete diffusion (energy, entropy, temperature annealing)
# The CAM e100 runs (runs/cam/mis_er700/CAM_tau0.1_e100_s*) use the same network, data,
# validation and per-epoch protocol and serve as a third (UL) arm.
#
#   CUDA_VISIBLE_DEVICES=0 METHOD=DIFUSCO EXP=DIFUSCO_e100_s0 bash cam/scripts/mis/slul_er700.sh --seed 0
#   CUDA_VISIBLE_DEVICES=1 METHOD=DiffUCO EXP=DiffUCO_bern_e100_s0 bash cam/scripts/mis/slul_er700.sh --seed 0
#   MODE=test ...                                     (evaluate best.pt or --checkpoint X.pt)
# Extra arguments are passed to main.py and override the defaults below (argparse keeps the last).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cam_require_gpu

export PRECISION=no
# Parsed-graph cache (datasets/mis.py); fill it first with scripts/mis/build_graph_cache.py.
export CAM_GRAPH_CACHE="${CAM_GRAPH_CACHE:-$_CAM_CACHE/graphs}"
export PROJECT="${PROJECT:-slul_er700}"
export EXP="${EXP:?set EXP}"
METHOD="${METHOD:?set METHOD=DIFUSCO or METHOD=DiffUCO}"
MODE="${MODE:-train}"
DATA="${CAM_ER700_DIR:-/data/shengyuf/MIS/data_er}"
SAVE_DIR="$CAM_RUNDIR/$PROJECT/$EXP"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
PORT=$((29500 + RANDOM % 1000))

case "$MODE" in
    train) MODE_ARGS=(--do_train) ;;
    test)  MODE_ARGS=(--log_file "$SAVE_DIR/test_log.txt") ;;
    *) echo "error: MODE must be train or test" >&2; exit 1 ;;
esac

case "$METHOD" in
    # DIFUSCO: T=1000 training noise levels (Sun & Yang 2023, sec. 4.1), batch 32 (their appendix E
    # for ER), AdamW; one epoch is one pass over the 16,284 training graphs (509 steps).
    DIFUSCO) METHOD_ARGS=(--method DIFUSCO --num_tp 1000 --batch_size 32 --optimizer AdamW
                          --diffusion_schedule linear --inference_schedule cosine --epochs 100
                          --save_every_epoch) ;;
    # DiffUCO: CAM's per-epoch protocol (1000 graphs, 20 trajectories of 50 steps, temperature
    # annealed to 0 over all epochs); RAdam with gradient clipping 1.0 as in the original DiffUCO
    # (CAM paper appendix B.2). --noise_energy bernoulli or annealing (CAM reports the better one).
    DiffUCO) METHOD_ARGS=(--method DiffUCO --num_tp 50 --num_kp 20 --samples_per_epoch 1000
                          --batch_size 1 --mini_batch_size 512 --optimizer RAdam --grad_clip 1.0
                          --tau0 0.1 --noise_energy bernoulli --epochs 100 --N_anneal 100
                          --save_every_epoch) ;;
    *) echo "error: METHOD must be DIFUSCO or DiffUCO" >&2; exit 1 ;;
esac

cd "$CAM_ROOT"
nice -n 10 "$CAM_ACCELERATE" launch --mixed_precision=$PRECISION \
    --num_processes=$NPROC --num_machines=1 --dynamo_backend=no --main_process_port $PORT main.py \
    --train_path "$DATA/er_train" \
    --valid_path "$DATA/er_valid" \
    --test_path "$DATA/er_test" \
    --problem mis \
    --valid_samples 64 \
    --num_t 50 \
    --num_k 20 \
    --num_h 64 \
    --num_l 8 \
    --model EncodeProcessDecode \
    --lr 0.0005 \
    --patience 1000 \
    --save_dir "$SAVE_DIR" \
    --wandb_name "$PROJECT" \
    --wandb_run_name "$EXP" \
    "${METHOD_ARGS[@]}" "${MODE_ARGS[@]}" "$@"
