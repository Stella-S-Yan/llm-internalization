#!/bin/bash
set +e


# Good bench mark comparisons
# Beauty
# runs=(
#     "--LR 4e-4 --WARMUP_STEPS 1000 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2"
# )

# Combined
runs=(
    "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --LORA_RATIO 2 --LORA_DROPOUT 0.2 --TOTAL_STEPS 40000"
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