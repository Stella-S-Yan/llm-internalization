#!/bin/bash
set +e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Combined
runs=(
    "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --LORA_RATIO 2 --LORA_DROPOUT 0.2 --TOTAL_STEPS 80000"
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_thinking_sft.py $params
  echo "Finished run with: $params"
done