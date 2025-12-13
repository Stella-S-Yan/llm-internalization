
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from utils import bagz_utils
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import os
import itertools
from transformers import get_cosine_schedule_with_warmup, get_inverse_sqrt_schedule, get_polynomial_decay_schedule_with_warmup
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set seeds for reproducibility
seed = 411
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_all_sid_alignment"
LOG_DIR = config.RUN_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_all_sid_alignment"
BATCH_SIZE = 4096
TOTAL_STEPS = 20_000     # plateau at step 2k
LR =  4e-3         #  1e-3 best, but then overfit 
TEMP = 0.1     # high temperature: smoother distribution, softer gradients

SCALE = 0.01    # Best
WARMUP_UP = 200
POLY_POW = 2.0
POLY_END_LR = 1e-7  # better than 1e-6 for Toys_and_Games





def save_model(model, tokenizer, old_vocab_size):
    save_dir = MODEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Save ONLY the newly trained embedding matrix (NOT full model)
    emb = model.get_input_embeddings().weight.data
    new_emb = emb[old_vocab_size:].detach().cpu()
    torch.save(new_emb, os.path.join(save_dir, "new_embeddings.pt"))

    print(f"Saved tokenizer + new embeddings at: {save_dir}")



def load_checkpoint(base_model_name, save_dir):
    # Load BASE MODEL again — quantized or FP16 as desired
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,   # or fp16, or load_in_4bit=True
    )

    # 2. Load extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)

    # 3. Resize embedding table
    model.resize_token_embeddings(new_vocab_size)

    # 4. Load saved new embedding weights
    new_emb = torch.load(os.path.join(save_dir, "new_embeddings.pt")).to(model.device)

    # 5. Insert the new embeddings back into the table
    with torch.no_grad():
        model.get_input_embeddings().weight[old_vocab_size:] = new_emb

    print(f"Restored model with extended vocab ({new_vocab_size} tokens)")

    return model, tokenizer
   


base_model_name = MODEL_NAME
save_dir = MODEL_SAVE_DIR
load_checkpoint(base_model_name, save_dir)