#!/bin/bash
set +e


# Combined
runs=(
    # "--SOURCE Toys_and_Games --LR 1e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --ACC_STEP 1 --LORA_RANK 32 --LORA_RATIO 1 --LORA_DROPOUT 0.2 --TOTAL_STEPS 80000"
    # "--SOURCE Toys_and_Games --LR 6e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --ACC_STEP 4 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.2 --TOTAL_STEPS 80000"
    "--LR 2e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 16 --ACC_STEP 1 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.2 --TOTAL_STEPS 60000"
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py $params
  echo "Finished run with: $params"
done