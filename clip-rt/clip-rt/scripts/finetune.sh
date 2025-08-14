export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 -m training.main \
    --batch-size 16 \
    --precision amp \
    --workers 4 \
    --save-frequency 1 \
    --dataset-type csv \
    --csv-separator="," \
    --train-data="./clip_rt_in_domain_data.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-supervision-key supervision \
    --csv-label-key label \
    --warmup 100 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=100 \
    --model="ViT-H-14-378-quickgelu" \
    --pretrained="./cliprt-oxe-pretrained.pt"

