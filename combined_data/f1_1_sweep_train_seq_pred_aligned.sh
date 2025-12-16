#!/bin/bash
set +e


# Combined
runs=(
    # "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --LORA_RANK 16 --LORA_RATIO 2 --LORA_DROPOUT 0.2 --TOTAL_STEPS 40000"
    "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --LORA_RANK 16 --LORA_RATIO 2 --LORA_DROPOUT 0.1 --TOTAL_STEPS 40000"
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py $params
  echo "Finished run with: $params"
done