#!/bin/bash
set +e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Better 0.051 on Toys 
# params="--LR 5e-5 --WEIGHT_DECAY 0.02 --WARMUP_STEPS 2000 \
#         --TRAIN_BATCH_SIZE 8 --ACC_STEP 1 \
#         --LORA_RANK 16 --LORA_RATIO 1 --LORA_DROPOUT 0.4 \
#         --TOTAL_STEPS 200000 --RUN_NUM 0 --CHECK_POINT 0"

params="--LR 5e-5 --WEIGHT_DECAY 0.02 --WARMUP_STEPS 2000 \
        --TRAIN_BATCH_SIZE 16 --ACC_STEP 1 \
        --LORA_RANK 16 --LORA_RATIO 1 --LORA_DROPOUT 0.4 \
        --TOTAL_STEPS 200000 --RUN_NUM 0 --CHECK_POINT 28000"





torchrun \
  --nproc_per_node=8 \
  train_thinking_sft.py $params

echo "Finished run on GPU $GPU_INDEX"
