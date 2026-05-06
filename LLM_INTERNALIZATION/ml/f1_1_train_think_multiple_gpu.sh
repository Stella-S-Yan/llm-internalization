#!/bin/bash
set +e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Best for 20m
# params="--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
#         --TRAIN_BATCH_SIZE 16 --ACC_STEP 1 \
#         --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.05 \
#         --TOTAL_STEPS 200000 --RUN_NUM 0 --CHECK_POINT 0"

# For 1m
params="--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
        --TRAIN_BATCH_SIZE 4 --ACC_STEP 1 \
        --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.05 \
        --TOTAL_STEPS 200000 --RUN_NUM 0 --CHECK_POINT 0"


torchrun \
  --nproc_per_node=7 \
  train_thinking_sft.py $params

echo "Finished run on GPU $GPU_INDEX"
