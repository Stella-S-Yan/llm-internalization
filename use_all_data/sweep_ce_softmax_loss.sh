#!/bin/bash
set +e

# pick on good setting and explore more
runs=(
  "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 1200 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2 --TOTAL_STEPS 28000 --LORA_DROPOUT 0.2 "
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_ce_softmax_loss.py $params
  echo "Finished run with: $params"
done