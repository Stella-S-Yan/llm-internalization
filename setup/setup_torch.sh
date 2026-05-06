#!/bin/bash

conda create -n torch_exp python=3.11 -y
conda activate torch_exp

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install transformers==4.57.3
pip install huggingface-hub==0.36.2
pip install peft==0.19.1
pip install datasets==4.8.5

pip install tensorboard==2.20.0
pip install matplotlib==3.10.9
pip install pandas==3.0.2
pip install sentence-transformers==5.4.1
pip install bagz