import torch
import math
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from multiprocessing import Process, set_start_method
from tqdm import tqdm
import config
from utils import bagz_utils

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# ---------------------------------------------------
# Embedding functions
# ---------------------------------------------------
@torch.no_grad()
def get_embedding_last_token(texts, model, tokenizer, device, normalize=True):
    """
    Returns embeddings for the last token of each text.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

    last_indices = attention_mask.sum(dim=1) - 1
    embedding = hidden_states[torch.arange(hidden_states.size(0)), last_indices]

    if normalize:
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu()


def embed_col(model, tokenizer, df, col_name, new_col_name, device, batch_size=256):
    """
    Embed a column in batches on a single GPU.
    Truncates destination_context to last 512 characters.
    """
    embeddings = []

    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"Embedding {col_name}"):
        batch_texts = df[col_name].iloc[start_idx:start_idx+batch_size].fillna("").astype(str)
        if col_name == "destination_context":
            batch_texts = batch_texts.apply(lambda x: x[-512:]).tolist()
        else:
            batch_texts = batch_texts.tolist()

        emb_batch = get_embedding_last_token(batch_texts, model, tokenizer, device)
        embeddings.extend(emb_batch)

        del emb_batch
        torch.cuda.empty_cache()

    df[new_col_name] = [e.tolist() for e in embeddings]
    return df

# ---------------------------------------------------
# GPU worker
# ---------------------------------------------------
def run_on_gpu(gpu_id, df_split, quote_split):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Running on {device}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device}, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 1️ Embed destination_context
    df_split = embed_col(model, tokenizer, df_split, "destination_context", "destination_context_llama", device)

    # 2️ Embed quote (unique passage_id rows)
    quote_split = embed_col(model, tokenizer, quote_split, "quote", "quote_llama", device)

    # 3️ Save partial results
    fname = config.LEPARD_TWO_EMB
    bagz_utils.save_parquet(df_split, f"{fname}_dest_{gpu_id}")
    bagz_utils.save_parquet(quote_split, f"{fname}_quote_{gpu_id}")

    print(f"[GPU {gpu_id}] Finished. Saved splits")

# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------
def gen_embedding():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 1️ Load full dataframe
    df = bagz_utils.read_parquet(config.LEPARD_OUTSIDE_EMB)

    # 2️ Deduplicate quotes by passage_id
    quote_df = df[["passage_id", "quote"]].drop_duplicates(subset="passage_id", keep="first").reset_index(drop=True)

    # 3️ Split both dataframes for multi-GPU
    n_gpus = torch.cuda.device_count()
    dest_splits = np.array_split(df, n_gpus)
    quote_splits = np.array_split(quote_df, n_gpus)

    # 4️ Launch GPU processes
    procs = []
    for gpu_id in range(n_gpus):
        p = Process(target=run_on_gpu, args=(gpu_id, dest_splits[gpu_id], quote_splits[gpu_id]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # 5️ Merge partial results
    # Destination_context
    dest_dfs = [bagz_utils.read_parquet(f"{config.LEPARD_TWO_EMB}_dest_{i}") for i in range(n_gpus)]
    full_df = pd.concat(dest_dfs, ignore_index=True)

    # Quote embeddings
    quote_dfs = [bagz_utils.read_parquet(f"{config.LEPARD_TWO_EMB}_quote_{i}") for i in range(n_gpus)]
    quote_df_full = pd.concat(quote_dfs, ignore_index=True)

    # 6️ Merge quote embeddings back
    full_df = full_df.merge(quote_df_full[["passage_id", "quote_llama"]], on="passage_id", how="left")

    # 7️ Save final dataframe
    bagz_utils.save_parquet(full_df, config.LEPARD_TWO_EMB)
    print("All embeddings completed and saved!")

# ---------------------------------------------------
# Entrypoint
# ---------------------------------------------------
if __name__ == "__main__":
    gen_embedding()
