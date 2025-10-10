
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import config


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / "sid_aligned_model"
LR = 5e-5

def load_full_model(save_dir):
    model = AutoModelForCausalLM.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    optimizer = torch.optim.Adam(model.get_input_embeddings().parameters(), lr=LR)
    optimizer.load_state_dict(torch.load(os.path.join(save_dir, "optimizer.pt")))

    return model, tokenizer


model, tokenizer = load_full_model(MODEL_SAVE_DIR)
print(len(tokenizer))