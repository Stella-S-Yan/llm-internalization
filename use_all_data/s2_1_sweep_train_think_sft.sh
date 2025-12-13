#!/bin/bash
set +e

# use all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# pick on good setting and explore more
runs=(
  # Best for Beauty, but LR is high for Toys
  # "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RATIO 2 --TOTAL_STEPS 50001 --LORA_DROPOUT 0.2 "

  # Toy. at least not overfit, can train longer
  # "--LR 9e-5 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RATIO 2 --TOTAL_STEPS 50001 --LORA_DROPOUT 0.2 "
  "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 8 --LORA_RATIO 2 --TOTAL_STEPS 50000 --LORA_DROPOUT 0.2 "
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_thinking_sft.py $params
  echo "Finished run with: $params"
done