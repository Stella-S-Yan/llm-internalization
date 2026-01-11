#!/bin/bash
set +e

if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_index>"
    exit 1
fi

GPU_INDEX=$1
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

# Select params based on GPU index
case $GPU_INDEX in
  0)
    params="--LR 1e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 8 --ACC_STEP 1 \
            --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  1)
    params="--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 16 --ACC_STEP 1 \
            --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  2)
    params="--LR 4e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 16 --ACC_STEP 2 \
            --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  3)
    params="--LR 8e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 16 --ACC_STEP 4 \
            --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  4)
    params="--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 16 --ACC_STEP 1 \
            --LORA_RANK 16 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  5)
    params="--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 16 --ACC_STEP 1 \
            --LORA_RANK 8 --LORA_RATIO 2 --LORA_DROPOUT 0.25 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  6)
    params="--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 32 --ACC_STEP 1 \
            --LORA_RANK 8 --LORA_RATIO 1 --LORA_DROPOUT 0.25 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  7)
    params="--LR 2e-4 --WEIGHT_DECAY 0.005 --WARMUP_STEPS 2000 \
            --TRAIN_BATCH_SIZE 32 --ACC_STEP 1 \
            --LORA_RANK 8 --LORA_RATIO 1 --LORA_DROPOUT 0.05 \
            --TOTAL_STEPS 100000 --RUN_NUM $GPU_INDEX"
    ;;
  *)
    echo "Invalid GPU index: $GPU_INDEX"
    exit 1
    ;;
esac

RDZV_PORT="2950${GPU_INDEX}"

echo "Starting run on GPU $GPU_INDEX"
echo "Params: $params"

torchrun \
  --nproc_per_node=1 \
  --rdzv_endpoint=localhost:$RDZV_PORT \
  train_thinking_sft.py $params

echo "Finished run on GPU $GPU_INDEX"
