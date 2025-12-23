#!/bin/bash
set +e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Combined
runs=(
    "--LR 5e-6 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 2 --LORA_RANK 16 --LORA_RATIO 1 --LORA_DROPOUT 0.0 --TOTAL_STEPS 1000"  # Best result. Keep
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_thinking_grpo_unsloth.py $params
  echo "Finished run with: $params"
done