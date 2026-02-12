#!/bin/bash
set +e

export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7

# Combined
runs=(
    # best
    # "--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.25 --TOTAL_STEPS 300000 --ACC_STEP 2 --RUN_NUM 0 --CHECK_POINT 194000"   # 64 out of memory
    
    # 50k
    "--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.25 --TOTAL_STEPS 300000 --ACC_STEP 2 --RUN_NUM 0 --CHECK_POINT 210000"   # 64 out of memory
)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  torchrun --nproc_per_node=7 train_thinking_sft.py $params
  echo "Finished run with: $params"
done