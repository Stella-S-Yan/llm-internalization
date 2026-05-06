#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda create -n jax_exp python=3.11 -y
source activate jax_exp

pip install -U "jax[cuda12]"

pip install flax==0.10.6 
pip install optax==0.2.4 
pip install tensorflow-cpu==2.20.0 

pip install pandas==3.0.2 
pip install pyarrow==24.0.0
pip install matplotlib==3.10.9
pip install bagz