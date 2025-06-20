#!/bin/bash
# run_ili.sh

# # First preprocess the data
# echo "Preprocessing ILI data..."
# python preprocess_ili_data.py --input_path data/state360.txt --output_dir data/processed

# Run AF-FCL on ILI dataset
echo "Running AF-FCL on ILI dataset..."
python main.py \
    --dataset ILI \
    --datadir ./data/processed/ \
    --data_split_file ili_afcl_data.pkl \
    --algorithm PreciseFCL \
    --num_glob_iters 60 \
    --local_epochs 50 \
    --lr 1e-3 \
    --flow_lr 5e-4 \
    --k_loss_flow 0.3 \
    --k_flow_lastflow 0.2 \
    --flow_explore_theta 0.1 \
    --fedprox_k 0.001 \
    --batch_size 16 \
    --target_dir_name output_ili