#!/bin/bash
# Original launch of the CAM paper's RB-[200-300] DiffUCO run with bernoulli noise (not the one in
# Table 1, which used annealing), as recorded in
# wandb/run-20260513_180501-biczvw9h/files/wandb-metadata.json. Imported for the record; we do not
# rerun it (user, 2026-11-16). It ran on sa in Shengyu's RiemannCO checkout (commit ebae8b8,
# entry point train.py, later renamed main.py) with 10 processes on 10 RTX 2080 Ti, DATADIR=/usr1/shengyuf.
# Identical to the annealing run (experiments/20260513_diffuco_rbsmall_annealing) except
# --noise_energy, --save_dir and --wandb_run_name.
#
# Test runs (load save_dir/best.pt):
#   2026-05-13 21:04 UTC, same arguments without --do_train: 19.701
#   2026-05-14 00:47 UTC, same but --noise_energy annealing (unused at test time): 19.739

accelerate launch --mixed_precision=no --num_processes=10 train.py \
    --train_path /usr1/shengyuf/MIS/RB_small_train \
    --valid_path /usr1/shengyuf/MIS/RB_small_valid \
    --test_path /usr1/shengyuf/MIS/RB_small_test \
    --problem mis \
    --samples_per_epoch 4000 \
    --valid_sample 64 \
    --batch_size 4 \
    --mini_batch_size 1 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 6 \
    --num_kp 10 \
    --num_h 64 \
    --num_l 8 \
    --tau0 0.1 \
    --method DiffUCO \
    --model EncodeProcessDecode \
    --do_train \
    --noise_energy bernoulli \
    --optimizer RAdam \
    --grad_clip 1 \
    --beta 1.2 \
    --lr 0.001 \
    --epochs 1000 \
    --N_anneal 1000 \
    --patience 200 \
    --lr_scheduler cosine-decay \
    --save_dir /usr1/shengyuf/RiemannCO/mis_rb_small_diffuco/diffuco_bernoulli \
    --wandb_name mis_rb_small_diffuco \
    --wandb_run_name diffuco_bernoulli
