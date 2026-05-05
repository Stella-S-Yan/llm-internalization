#!/bin/bash
set +e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Beauty 
params="--LR 1e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
        --TRAIN_BATCH_SIZE 1 --ACC_STEP 1 \
        --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
        --TOTAL_STEPS 300000 --RUN_NUM 0 --CHECK_POINT 0"


torchrun \
  --nproc_per_node=8 \
  train_thinking_sft.py $params

echo "Finished run on GPU $GPU_INDEX"
