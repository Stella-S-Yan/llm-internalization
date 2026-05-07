#!/bin/bash
set +e

GPU_INDEX=$1
CHECK_POINT=$2 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# Unique rendezvous port per GPU
RDZV_PORT="2950${GPU_INDEX}"

python merge_save_sft_think_model.py --RUN_NUM $GPU_INDEX --CHECK_POINT $CHECK_POINT

echo "Starting run on GPU $GPU_INDEX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RDZV_PORT=$RDZV_PORT"

torchrun \
  --nproc_per_node=7 \
  --rdzv_endpoint=localhost:$RDZV_PORT \
  eval_sft_think.py --RUN_NUM $GPU_INDEX

echo "Finished run on GPU $GPU_INDEX"