#!/usr/bin/env bash
set -e

# ----------------------------------------
# SAFETY: kill all children on Ctrl+C
# ----------------------------------------
trap 'echo "Stopping all experiments..."; kill 0' SIGINT SIGTERM

# ----------------------------------------
# CONFIG
# ----------------------------------------
GPUS=(0 1 2 3 4 5 6 7)

# Each experiment = one line
# 52000 / 32 * 3 ≈ 4875 steps
EXPERIMENTS=(
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 100 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 200 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 300 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 400 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 2 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 100 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 1 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 200 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 1 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 300 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 1 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"
  "--LR 4e-4 --WEIGHT_DECAY 0.001 --WARMUP_STEPS 400 --TRAIN_BATCH_SIZE 32 --LORA_RANK 32 --LORA_RATIO 1 --LORA_DROPOUT 0.05 --TOTAL_STEPS 5000"

)

# ----------------------------------------
# LAUNCH
# ----------------------------------------
NUM_GPUS=${#GPUS[@]}
NUM_EXPS=${#EXPERIMENTS[@]}

for i in "${!EXPERIMENTS[@]}"; do
  GPU_ID=${GPUS[$((i % NUM_GPUS))]}
  CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python select_param_sft.py ${EXPERIMENTS[$i]}"

  echo "Launching on GPU ${GPU_ID}: ${CMD}"
  eval "${CMD}" &

  # Prevent launching more jobs than GPUs
  if (( (i + 1) % NUM_GPUS == 0 )); then
    wait
  fi
done

wait
echo "All experiments completed."
