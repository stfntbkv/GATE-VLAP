#!/bin/bash

# Define the input lists
train_data_root="/data/oxe/"
list_data="fractal20220817_data kuka bridge_orig taco_play jaco_play berkeley_cable_routing roboturk viola berkeley_autolab_ur5 language_table stanford_hydra_dataset_converted_externally_to_rlds austin_buds_dataset_converted_externally_to_rlds nyu_franka_play_dataset_converted_externally_to_rlds furniture_bench_dataset_converted_externally_to_rlds ucsd_kitchen_dataset_converted_externally_to_rlds austin_sailor_dataset_converted_externally_to_rlds austin_sirius_dataset_converted_externally_to_rlds dlr_edan_shared_control_converted_externally_to_rlds iamlab_cmu_pickup_insert_converted_externally_to_rlds utaustin_mutex cmu_stretch bc_z"
list_weight="1.0 0.8341046294 1.0 1.0 1.0 1.0 2.0 1.0 3.0 0.1 1.0 1.0 3.0 0.1 3.0 1.0 1.0 1.0 1.0 1.0 1.0 0.2"


# Initialize empty strings for the concatenated results
all_train_data=""
all_weights=""
all_shards="/*.tar"

# Read the string list into positional parameters
set -- $list_data
for str in "$@"; do
    data_path="${train_data_root}${str}${all_shards}"
    if [ -z "$all_train_data" ]; then
        all_train_data="$data_path"
    else
	all_train_data="${all_train_data}::${data_path}"
    fi
done


# Read the weight list into positional parameters
set -- $list_weight
for num in "$@"; do
    if [ -z "$all_weights" ]; then
        all_weights="$num"
    else
        all_weights="${all_weights}::${num}"
    fi
done

# Print the results
echo "$all_train_data"
echo "$all_val_data"
echo "$all_weights"

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node 4 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 -m training.main \
    --batch-size 32 \
    --precision amp \
    --workers 4 \
    --save-frequency 1 \
    --train-num-samples 1000000 \
    --dataset-resampled \
    --train-data="$all_train_data" \
    --train-data-upsampling-factors="$all_weights" \
    --dataset-type webdataset \
    --csv-separator="," \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-supervision-key supervision \
    --csv-label-key label \
    --warmup 10000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=100 \
    --model="ViT-H-14-378-quickgelu"
    
