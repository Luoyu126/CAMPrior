#!/bin/bash
# After the DiffUCO MDS run (20261123_diffuco_mds_ul) finishes its 1000 epochs: test it, then make DiffUCO labels
# and train DIFUSCO on them.
#   1. test the last epoch and best.pt on the 1000 test graphs (20 samples, as the DIFUSCO tests);
#   2. sample the last epoch 200 times on every training graph (scripts/mis/ul_samples.py) and keep, per graph,
#      all distinct smallest sets (scripts/mds/ul_label_sets.py) -> data/mds_labels/<ds>_small_train_diffuco.pt;
#   3. launch DIFUSCO with these labels, one set drawn per step (dif) and soft (dif_soft), renoise validation.
# Usage (detached): nohup bash experiments/20261123_difusco_mds_labels/dif_labels.sh <ba|rb> <gpu a> <gpu b> > LOG &
set -uo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$EXP_DIR/../.." && pwd)"
DS=$1; GA=$2; GB=$3
UL=20261123_diffuco_mds_ul; RUN=${UL}_${DS}small_s0; OUT="$REPO/runs/experiments/$UL/$RUN"
L=/usr0/home/shengyuf/yunyi/data/mds_labels
case "$DS" in
    ba) D=/usr0/home/shengyuf/yunyi/data/CAM_data/MDS; TR=$D/BA_small_train ;;
    rb) D=/usr1/shengyuf/MIS; TR=$D/RB_small_train ;;
esac
until [ -f "$OUT/epoch_0999.pt" ] && ! pgrep -f "wandb_run_name $RUN" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') $RUN finished training"
bash "$REPO/experiments/$UL/test.sh" $GA $DS epoch_0999.pt &
bash "$REPO/experiments/$UL/test.sh" $GB $DS best.pt &
wait
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source "$REPO/cam/scripts/env.sh"; export WANDB_MODE=disabled
cd "$REPO/cam"
UL_SAMPLES_OUT="$L/${DS}_small_train_diffuco_samples200.pt" CUDA_VISIBLE_DEVICES=$GA "$CAM_PYTHON" scripts/mis/ul_samples.py \
    --train_path $TR --valid_path $TR --test_path $TR --problem mds --seed 1 \
    --method DiffUCO --model EncodeProcessDecode --num_h 64 --num_l 8 --num_t 50 --num_k 200 --num_tp 5 \
    --noise_energy annealing --num_workers 4 --checkpoint "$OUT/epoch_0999.pt" --save_dir "$OUT" \
    --log_file "$OUT/train_samples200_result.txt" --wandb_name SLVUL --wandb_run_name "$RUN" > "$OUT/train_samples200.log" 2>&1
"$CAM_PYTHON" scripts/mds/ul_label_sets.py "$L/${DS}_small_train_diffuco_samples200.pt" "$L/${DS}_small_train_diffuco.pt" \
    --ilp "$L/${DS}_small_train_ilp$([ $DS = ba ] && echo 8 || echo 1).pt" || exit 1
cd "$REPO"
PORT=29830 bash experiments/20261123_difusco_mds_labels/launch.sh $GA $DS dif renoise 0
PORT=29831 bash experiments/20261123_difusco_mds_labels/launch.sh $GB $DS dif_soft renoise 0
