#!/bin/bash

# Run the main training script for TinyImageNet
echo "Starting TinyImageNet training..."
python main.py \
    --dataset TinyImageNet \
    --data_split_file TinyImageNet_split_cn10_tn6_cet30_s42.pkl \
    --num_glob_iters 60 \
    --local_epochs 50 \
    --lr 5e-4 \
    --flow_lr 1e-3 \
    --k_loss_flow 0.3 \
    --k_flow_lastflow 0.1 \
    --flow_explore_theta 0.1 \
    --fedprox_k 0.001 \
    --batch_size 32 \
    --c-channel-size 128 \
    --target_dir_name output_tinyimagenet