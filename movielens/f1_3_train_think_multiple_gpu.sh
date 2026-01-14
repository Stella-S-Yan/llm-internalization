#!/bin/bash
set +e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

params="--LR 2e-4 --WEIGHT_DECAY 0.05 --WARMUP_STEPS 5000 \
        --TRAIN_BATCH_SIZE 16 --ACC_STEP 1 \
        --LORA_RANK 16 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
        --TOTAL_STEPS 150000 "


torchrun \
  --nproc_per_node=7 \
  train_thinking_sft.py $params

echo "Finished run on GPU $GPU_INDEX"
