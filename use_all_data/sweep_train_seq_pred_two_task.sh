#!/bin/bash
set +e

# Enable PyTorch Elastic error logging
export TORCHELASTIC_ERROR_FILE=errors.log

# pick on good setting and explore more
runs=(
  # "--LR 4e-4 --WEIGHT_DECAY 0.05 --WARMUP_STEPS 1050 --TRAIN_BATCH_SIZE 8 --ACCUMULATION_STEPS 2 --LORA_RATIO 2 --TOTAL_STEPS 54000 --LORA_DROPOUT 0.1 "  # 800
  "--LR 4e-4 --WEIGHT_DECAY 0.1 --WARMUP_STEPS 1500 --TRAIN_BATCH_SIZE 8 --ACCUMULATION_STEPS 2 --LORA_RATIO 2 --TOTAL_STEPS 54000 --LORA_DROPOUT 0.2 " 
  # "--LR 4e-4 --WEIGHT_DECAY 0.05 --WARMUP_STEPS 1500 --TRAIN_BATCH_SIZE 8 --ACCUMULATION_STEPS 2 --LORA_RATIO 2 --TOTAL_STEPS 54000 --LORA_DROPOUT 0.1 "  # 1200
)



for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --standalone --nproc_per_node=8 train_seq_pred_two_task.py $params
  echo "Finished run with: $params"
done