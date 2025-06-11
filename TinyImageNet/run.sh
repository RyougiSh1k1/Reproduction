#!/bin/bash

# # First, create the data split for TinyImageNet
# echo "Creating TinyImageNet data split..."
# python split_dataset_tinyimagenet.py \
#     --dataset TinyImageNet \
#     --datadir ./datasets/PreciseFCL/ \
#     --data_split_file data_split/TinyImageNet_split_cn10_tn6_cet30_s42.pkl \
#     --client_num 10 \
#     --task_num 6 \
#     --class_each_task 30 \
#     --seed 42

# # Check if the data split was created successfully
# if [ $? -ne 0 ]; then
#     echo "Data split creation failed. Exiting."
#     exit 1
# fi

# Check if the pickle file was created
if [ ! -f "data_split/TinyImageNet_split_cn10_tn6_cet30_s42.pkl" ]; then
    echo "Data split file not found. Exiting."
    exit 1
fi

echo "Data split created successfully!"

# Run the main training script for TinyImageNet
echo "Starting TinyImageNet training..."
python main.py \
    --dataset TinyImageNet \
    --data_split_file data_split/TinyImageNet_split_cn10_tn6_cet30_s42.pkl \
    --num_glob_iters 60 \
    --local_epochs 200 \
    --lr 5e-4 \
    --flow_lr 1e-3 \
    --k_loss_flow 0.3 \
    --k_flow_lastflow 0.1 \
    --flow_explore_theta 0.1 \
    --fedprox_k 0.001 \
    --batch_size 32 \
    --c-channel-size 128 \
    --target_dir_name output_tinyimagenet