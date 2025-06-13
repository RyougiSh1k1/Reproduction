#!/bin/bash

# Script to run PreciseFCL on TinyImageNet

# # First, download TinyImageNet if not already downloaded
# if [ ! -d "./datasets/PreciseFCL/tiny-imagenet-200" ]; then
#     echo "Downloading TinyImageNet-200..."
#     mkdir -p ./datasets/PreciseFCL/
#     cd ./datasets/PreciseFCL/
#     wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
#     unzip tiny-imagenet-200.zip
#     rm tiny-imagenet-200.zip
#     cd ../..
# fi

# # Generate data split file if not exists
# if [ ! -f "data_split/TinyImageNet_split_cn10_tn5_cet40_s42.pkl" ]; then
#     echo "Generating data split file..."
#     python split_dataset.py \
#         --dataset TinyImageNet \
#         --datadir ./datasets/PreciseFCL/ \
#         --data_split_file data_split/TinyImageNet_split_cn10_tn5_cet40_s42.pkl \
#         --client_num 10 \
#         --task_num 5 \
#         --class_each_task 40 \
#         --class_split 5 \
#         --seed 42
# fi

# Run the main experiment
echo "Running PreciseFCL on TinyImageNet..."
python main.py \
    --dataset TinyImageNet \
    --data_split_file data_split/TinyImageNet_split_cn10_tn5_cet40_s42.pkl \
    --num_glob_iters 50 \
    --local_epochs 100 \
    --lr 1e-3 \
    --flow_lr 5e-3 \
    --k_loss_flow 0.5 \
    --k_flow_lastflow 0.1 \
    --flow_explore_theta 0.1 \
    --fedprox_k 0.001 \
    --batch_size 32 \
    --c-channel-size 64 \
    --target_dir_name output_tinyimagenet