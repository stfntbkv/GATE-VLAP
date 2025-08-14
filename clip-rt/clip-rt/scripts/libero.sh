#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node 4 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 -m training.main \
    --batch-size 32 \
    --precision amp \
    --workers 4 \
    --save-frequency 1 \
    --train-data="./libero_object_tar/{000000..000067}.tar" \ # should specify the path of tar files         
    --train-num-samples 67000 \                               # if the tar file ends with 000067.tar, write 67*1000 
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
    --pretrained="./cliprt-oxe-pretrained.pt"
