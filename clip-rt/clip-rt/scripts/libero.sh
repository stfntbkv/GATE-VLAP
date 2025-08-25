#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 -m training.main \
    --batch-size 32 \
    --precision amp \
    --workers 1 \
    --save-frequency 1 \
    --train-data="/home/dimitar/Robotics/Robotic-Simulation-GATE/VLABench/VLABench/downloads/VLA_dataset/modified_raw_files/000000.tar" \ # should specify the path of tar files         
    --train-num-samples 2000 \                               # if the tar file ends with 000067.tar, write 67*1000 
    --dataset-type webdataset \
    --csv-separator="," \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-label-key label \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=256 \
    --local-loss \
    --gather-with-grad \
    --use-action-decoder \
    --model="ViT-H-14-378-quickgelu" \
    --pretrained="./weights/clip-rt-finetuned-libero-object/weights/cliprt_libero_object.pt"
