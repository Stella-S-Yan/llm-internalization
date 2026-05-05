#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate torch_think_test
python p1_1_t5_emb.py
python p1_2_llama_emb.py

conda activate jax
python p2_train_rqvae.py
python p3_gen_sid.py

conda activate torch_think_test
python p4_train_sid_alignment.py
python p5_train_eval_data.py






