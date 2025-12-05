#!/bin/bash
set +e


# Good bench mark comparisons
# Beauty
# runs=(
#     "--LR 4e-4 --WARMUP_STEPS 1000 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2"
#     "--LR 4e-4 --WARMUP_STEPS 1000 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 1"
#     "--LR 5e-4 --WARMUP_STEPS 1500 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 1"
#     "--LR 5e-4 --WARMUP_STEPS 1500 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2"
#     "--LR 5e-4 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 20 --LORA_RATIO 1"
# )

# Sports
# 4e-3 too high
runs=(
    # "--LR 4e-4 --WARMUP_STEPS 1200 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2 --TOTAL_STEPS 20000"
    # "--LR 4e-4 --WARMUP_STEPS 200 --TRAIN_BATCH_SIZE 32 --LORA_RATIO 2 --TOTAL_STEPS 20000" 
    "--LR 4e-4 --WARMUP_STEPS 200 --TRAIN_BATCH_SIZE 32 --LORA_RATIO 2 --TOTAL_STEPS 20000 --WEIGHT_DECAY 0.05 --POLY_POW 3.0"
    # "--LR 4e-3 --WARMUP_STEPS 1000 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2"
)




# pick on good setting and explore more
# runs=(
#   # Beauty
#   "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 1200 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2 --TOTAL_STEPS 22000 --LORA_DROPOUT 0.2 "
# )


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py $params
  echo "Finished run with: $params"
done