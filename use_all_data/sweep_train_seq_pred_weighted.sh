#!/bin/bash
set +e

# pick on good setting and explore more
runs=(
  # "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 1200 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2 --TOTAL_STEPS 30000 --LORA_DROPOUT 0.2 "
  
  # over fit for [5, 1. ]
  "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 1200 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2 --TOTAL_STEPS 27000 --LORA_DROPOUT 0.20 "  

  # result is bad
  # "--LR 9e-5 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 1200 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2 --TOTAL_STEPS 28000 --LORA_DROPOUT 0.2 "  
  
  # lr = 2e-4 underfilt, total steps = 30000 overfit
  # "--LR 2e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 1200 --TRAIN_BATCH_SIZE 16 --LORA_RATIO 2 --TOTAL_STEPS 28000 --LORA_DROPOUT 0.2 "    
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=8 train_seq_pred_weighted_loss.py $params
  echo "Finished run with: $params"
done