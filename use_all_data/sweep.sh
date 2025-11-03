#!/bin/bash
set +e

runs=(
    "--LR 4e-4 --WARMUP_STEPS 1000 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2"
    "--LR 4e-4 --WARMUP_STEPS 1000 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 1"
    "--LR 5e-4 --WARMUP_STEPS 1500 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 1"
    "--LR 5e-4 --WARMUP_STEPS 1500 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2"
    "--LR 5e-4 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 20 --LORA_RATIO 1"
)

for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py $params
  echo "Finished run with: $params"
done