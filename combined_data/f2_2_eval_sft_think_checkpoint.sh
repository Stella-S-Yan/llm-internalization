#!/bin/bash
set +e

STEP=$1   # first argument passed to the script

python merge_save_sft_think_model.py --checkpoint_step $STEP

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 eval_sft_think_ddp.py