#!/bin/bash
set +e

# Start vLLM server first
# $ trl vllm-serve --model think_model_sft/ --dtype half

# pick on good setting and explore more
runs=(
  # "--LR 4e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RATIO 2 --TOTAL_STEPS 40000 --LORA_DROPOUT 0.2 --ADAPTOR_SAVE_DIR train_thinking "

  # maximum can handel batchsize = 8  
  # Policy collapse around step 600~800. LR too high, beta = 0
  # "--LR 6e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RATIO 2 --TOTAL_STEPS 50000 --LORA_DROPOUT 0.2 --ADAPTOR_SAVE_DIR train_thinking "

  # beta = 0.2
  "--LR 5e-6 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RANK 8 --LORA_RATIO 4 --LORA_DROPOUT 0.2"

  # "--LR 3e-4 --WEIGHT_DECAY 0.01 --WARMUP_STEPS 800 --TRAIN_BATCH_SIZE 8 --LORA_RANK 8 --LORA_RATIO 4 --LORA_DROPOUT 0.2"

)


for params in "${runs[@]}"; do
  echo "Starting run with: $params"
  # torchrun fails. The trainer uses accelerate internally and run on all gpus
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_thinking_grpo.py $params 
  echo "Finished run with: $params"
done