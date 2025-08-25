#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
cd /home/dimitar/Robotics/Robotic-Simulation-GATE/clip_rt
export PYTHONPATH="$(pwd):$(pwd)/open_clip/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --batch-size 16
  --precision amp
  --workers 4
  --save-frequency 3
  --train-data "/home/dimitar/Robotics/Robotic-Simulation-GATE/clip_rt/action_tasks_tar/{000000..000076}.tar"  
  --train-num-samples 20000
  --dataset-type webdataset
  --warmup 1000
  --lr 5e-6  
  --wd 0.1
  --epochs 20
  --local-loss
  --gather-with-grad
  --use-action-decoder
  --model ViT-H-14-378-quickgelu
  --pretrained /home/dimitar/Robotics/Robotic-Simulation-GATE/clip_rt/clip-rt-oxe-pretrained/cliprt-oxe-pretrained.pt
  --lock-image
  --lock-text
  --freeze-vision-text
  --log-every-n-steps 10
  --name libero_object_pick_place_capabilities
)

# Run training with error handling
if torchrun --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 -m training.main "${ARGS[@]}" 2>&1 | tee training.log; then
    echo "Training completed successfully at: $(date)"
else
    echo "Training failed at: $(date)"
    exit 1
fi