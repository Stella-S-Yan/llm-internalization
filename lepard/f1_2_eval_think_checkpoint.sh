#!/bin/bash
set +e

STEP=$1   # first argument passed to the script
DATA_TYPE=$2

# python merge_save_sft_think_model.py --checkpoint_step $STEP

PARAMS="--DATA_TYPE $DATA_TYPE"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 eval_think_sft_ddp.py $PARAMS


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
torchrun --nproc_per_node=7 eval_think_sft_ddp.py $PARAMS