""" 
Generate product embeddings with llama
"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import math
from utils import bagz_utils
import config
import numpy as np
from multiprocessing import Process
import torch.multiprocessing as mp


MODEL_NAME = "meta-llama/Llama-3.2-1B"  #"meta-llama/Llama-3.2-1B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_MODEL_DIR = config.MODEL_DIR / "alignment"


def load_model_tokenizer(run_test=False):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)  
    # model = AutoModelForCausalLM.from_pretrained("/usr/local/google/home/stellasyan/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6", dtype=torch.bfloat16)  
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"           

    return model, tokenizer


# Embedding function
@torch.no_grad()
def get_embedding_mean_pooling(texts, model, tokenizer, device="cuda", normalize=True):
    """
    texts: list of strings
    Returns tensor: [len(texts), hidden_dim]
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # adjust as needed
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Forward pass (FP16)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

    # Mean pooling
    mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    embedding = summed / counts

    if normalize:
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu()  # move back to CPU to save GPU memory


@torch.no_grad()
def get_embedding_last_token(texts, model, tokenizer, device, normalize=True):
    """
    texts: list of strings
    Returns tensor: [len(texts), hidden_dim]
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

    # Last token embedding (ignore pads)
    last_indices = attention_mask.sum(dim=1) - 1  # [batch]
    embedding = hidden_states[torch.arange(hidden_states.size(0)), last_indices]

    if normalize:
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu()


# Batch processing over dataframe
def embed_col(model, tokenizer, df, col_name, new_col_name, device, batch_size=128):
    embeddings = []
    num_batches = math.ceil(len(df) / batch_size)

    for i in tqdm(range(num_batches), desc="Generating embeddings"):
        batch_texts = df[col_name].iloc[i*batch_size : (i+1)*batch_size].tolist()
        emb_batch = get_embedding_last_token(batch_texts, model, tokenizer, device)
        embeddings.extend(emb_batch)

        # free memory
        del emb_batch
        torch.cuda.empty_cache()

    # Convert to list for saving
    df[new_col_name] = [e.tolist() for e in embeddings]
    return df


def gen_embedding():
    model, tokenizer = load_model_tokenizer()
    model.eval()
    model.to(DEVICE)
    
    fname = config.META_W_SID
    df = bagz_utils.read_parquet(fname)

    df_out = embed_col(model, tokenizer, df, "formatted_text", "llama_embedding", DEVICE)
    bagz_utils.save_parquet(df_out,  config.META_W_EMB_SID)
    print(f"Embedding generation finished. Saved to {config.META_W_EMB_SID}")

if __name__=="__main__":
    gen_embedding()