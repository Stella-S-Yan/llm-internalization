#!/bin/bash

conda create -n jax_exp python=3.11 -y
source activate jax_exp

pip install -U "jax[cuda12]"
pip install flax==0.10.6 optax==0.2.4 tensorflow-cpu pandas matplotlib bagz pyarrow