#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda create -n torch_exp python=3.11 -y
conda activate torch_exp

echo "Current env: $CONDA_DEFAULT_ENV"

python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

python -m pip install transformers==4.57.3
python -m pip install peft==0.18.0
python -m pip install datasets==4.8.5

python -m pip install tensorboard==2.20.0
python -m pip install setuptools==80.9.0
python -m pip install matplotlib==3.10.9
python -m pip install pandas==3.0.2
python -m pip install sentence-transformers==5.4.1
python -m pip install bagz

python -m pip install huggingface-hub==0.36.2
