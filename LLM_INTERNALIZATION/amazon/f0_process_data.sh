#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch_exp

python p0_process_data.py
python p1_t5_emb.py
python p2_llama_embedding.py

conda activate jax_exp
python p3_train_rqvae.py
python p4_gen_sid.py

conda activate torch_exp
python p5_gen_sequence.py
python p6_fixed_grain_dataset.py
python p7_emb_alignment.py
python p8_build_reasoning_data.py





