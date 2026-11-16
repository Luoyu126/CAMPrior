#!/bin/bash
# Original launch of the CAM paper's RB-[200-300] DiffUCO run (Table 1: 19.88), as recorded in
# wandb/run-20260513_214606-gcgve1vl/files/wandb-metadata.json. Imported for the record; we do not
# rerun it (user, 2026-11-16). It ran on sa in Shengyu's RiemannCO checkout (commit ebae8b8,
# entry point train.py, later renamed main.py) with 10 processes on 10 RTX 2080 Ti, DATADIR=/usr1/shengyuf.
# The process count is inferred from the logs: 4000 // P // 4 = 100 steps per epoch gives P = 10.
#
# Test runs (same arguments without --do_train; they load save_dir/best.pt):
#   2026-05-14 00:30 UTC, commit ebae8b8: 19.872
#   2026-05-22 15:27 UTC, commit 7b03f86: 19.876 (reported in Table 1 as 19.88)

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
    --noise_energy annealing \
    --optimizer RAdam \
    --grad_clip 1 \
    --beta 1.2 \
    --lr 0.001 \
    --epochs 1000 \
    --N_anneal 1000 \
    --patience 200 \
    --lr_scheduler cosine-decay \
    --save_dir /usr1/shengyuf/RiemannCO/mis_rb_small_diffuco/diffuco_annealing \
    --wandb_name mis_rb_small_diffuco \
    --wandb_run_name diffuco_annealing
