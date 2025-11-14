#!/bin/bash
set +e

# pick on good setting and explore more
runs=(
  # "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RATIO 2 --TOTAL_STEPS 40000 --LORA_DROPOUT 0.2 --ADAPTOR_SAVE_DIR train_thinking "

  "--LR 6e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RATIO 2 --TOTAL_STEPS 50000 --LORA_DROPOUT 0.2 --ADAPTOR_SAVE_DIR train_thinking "
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_thinking_sft.py $params
  echo "Finished run with: $params"
done