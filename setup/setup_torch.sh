#!/bin/bash

conda create -n torch_exp python=3.11 -y
source activate torch_exp

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft pandas datasets trl tensorboard bagz matplotlib
pip install -U sentence-transformers
pip install -U huggingface-hub==0.35.3