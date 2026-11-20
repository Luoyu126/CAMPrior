#!/bin/bash
# Per-sample sizes of a checkpoint on the first 1000 TRAINING graphs (50 steps x 20, renoise, seed 0).
# Usage: bash train_eval.sh <gpu> <tag> <ckpt>      -> tr_<tag>.pt (dict idx -> 20 sizes), tr_<tag>.txt
cd "/home/shengyuf/yunyi/Schrödinger Bridge/CAMPrior/cam"
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source scripts/env.sh
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
S="/home/shengyuf/yunyi/Schrödinger Bridge/CAMPrior/runs/scratch/diag_1120"; D=/usr1/shengyuf/MIS
OUTF="$S/tr_$2.pt" CUDA_VISIBLE_DEVICES=$1 "$CAM_PYTHON" "$S/../dist_1119/per_sample.py" \
  --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_train --test_samples 1000 \
  --problem mis --valid_samples 64 --num_workers 4 --seed 0 --method DIFUSCO --model EncodeProcessDecode --num_h 64 --num_l 8 \
  --num_tp 50 --diffusion_schedule cosine --batch_size 40 --num_t 50 --num_k 20 --inference_schedule linear \
  --heat_lambda 0 --heat_until 0.5 --difusco_sampler renoise \
  --checkpoint "$3" --save_dir "$S/save" --log_file "$S/tr_$2.txt" --wandb_run_name tr_$2 > "$S/tr_$2.log" 2>&1
echo "$2 $(grep "Avg. size" "$S/tr_$2.txt")"
