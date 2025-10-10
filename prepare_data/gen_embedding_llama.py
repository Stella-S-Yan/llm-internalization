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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_tokenizer():
    # Load model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)  # FP16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Original vocab size:", len(tokenizer))

    model.eval()
    model.to(DEVICE)
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
def get_embedding_last_token(texts, model, tokenizer,  normalize=True):
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
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

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
def embed_col(model, tokenizer, df, col_name, new_col_name, batch_size=8):
    embeddings = []
    num_batches = math.ceil(len(df) / batch_size)

    for i in tqdm(range(num_batches), desc="Generating embeddings"):
        batch_texts = df[col_name].iloc[i*batch_size : (i+1)*batch_size].tolist()
        emb_batch = get_embedding_last_token(batch_texts, model, tokenizer, device=DEVICE)
        embeddings.extend(emb_batch)

        # free memory
        del emb_batch
        torch.cuda.empty_cache()

    # Convert to list for saving
    df[new_col_name] = [e.tolist() for e in embeddings]
    return df



def do_the_work():
    model, tokenizer = load_model_tokenizer()

    df = bagz_utils.read_parquet(config.META_W_EMBEDDING)
        
    df = embed_col(model, tokenizer, df, "formatted_text", "llama_embedding", batch_size=16)
     
    bagz_utils.save_parquet(df, config.META_W_LLAMA_EMBEDDING)

    print(df.head())


if __name__=="__main__":
    do_the_work()