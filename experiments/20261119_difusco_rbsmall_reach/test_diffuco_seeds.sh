#!/bin/bash
cd "/home/shengyuf/yunyi/Schrödinger Bridge/CAMPrior/cam"
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source scripts/env.sh
export CAM_GRAPH_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam/graphs
S="/home/shengyuf/yunyi/Schrödinger Bridge/CAMPrior/runs/scratch/dist_1119"; D=/usr1/shengyuf/MIS
go() { local gpu=$1 nt=$2; KEEP_SETS=0 UL_SAMPLES_OUT="$S/test_diffuco_k20_t${nt}_s${SEED:-0}.pt" CUDA_VISIBLE_DEVICES=$gpu "$CAM_PYTHON" scripts/mis/ul_samples.py \
  --train_path $D/RB_small_train --valid_path $D/RB_small_valid --test_path $D/RB_small_test --problem mis --valid_samples 64 --num_workers 4 --seed ${SEED:-0} \
  --method DiffUCO --model EncodeProcessDecode --samples_per_epoch 4000 --batch_size 4 --mini_batch_size 1 --num_t $nt --num_k 20 --num_tp 6 --num_kp 10 \
  --num_h 64 --num_l 8 --tau0 0.1 --noise_energy annealing --optimizer RAdam --grad_clip 1 --beta 1.2 --lr 0.001 --epochs 1000 --N_anneal 1000 \
  --patience 200 --lr_scheduler cosine-decay --save_dir ../runs/experiments/20260513_diffuco_rbsmall_annealing --log_file "$S/du_t${nt}_s${SEED}.txt" --wandb_run_name du_t${nt}_s${SEED} > "$S/du_t${nt}_s${SEED}.log" 2>&1; echo "du t$nt done"; }
SEED=0 go 6 50; SEED=1 go 6 50 & SEED=2 go 7 50 & wait; SEED=3 go 7 50
