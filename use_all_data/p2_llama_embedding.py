""" 
Generate product embeddings with llama

# Use all gpus

$ python llama_embedding.py
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


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  

def run_on_gpu(gpu_id, df_split):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Running on {device}")

    # Load model directly to the correct GPU
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map={"": device}) 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    df_split = embed_col(model, tokenizer, df_split, "formatted_text", "llama_embedding", device)

    fname = config.META_W_ALL_TWO_EMB
    bagz_utils.save_parquet(df_split,  f"{fname}_{gpu_id}")
    print(f"[GPU {gpu_id}] Finished. Saved to {fname}_{gpu_id}")


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
# Bigger batch_size will OOM
def embed_col(model, tokenizer, df, col_name, new_col_name, device, batch_size=256):
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

    fname = config.META_W_ALL_EMBEDDING
    df = bagz_utils.read_parquet(fname)

    n_gpus = torch.cuda.device_count()
    splits = np.array_split(df, n_gpus)

    procs = []
    for gpu_id, df_split in enumerate(splits):
        p = Process(target=run_on_gpu, args=(gpu_id, df_split))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    

if __name__=="__main__":
    gen_embedding()