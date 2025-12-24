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


# Embedding functions
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

    return embedding.cpu().numpy()


def embed_col(model, tokenizer, df, col_name, device, batch_size=256):
    """
    Embed a column in batches on a single GPU.
    Truncates destination_context to last 512 characters.
    Returns embeddings as a numpy array.
    """
    embeddings = []

    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"Embedding {col_name}"):
        batch_texts = df[col_name].iloc[start_idx:start_idx+batch_size].fillna("").astype(str)
        if col_name == "destination_context":
            batch_texts = batch_texts.apply(lambda x: x[-512:]).tolist()
        else:
            batch_texts = batch_texts.tolist()

        emb_batch = get_embedding_last_token(batch_texts, model, tokenizer, device)
        embeddings.append(emb_batch)

        del emb_batch
        torch.cuda.empty_cache()

    return np.vstack(embeddings)  # shape [num_rows, hidden_dim]


# GPU worker
def run_on_gpu(gpu_id, dest_df, quote_df, out_prefix):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Running on {device}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device}, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 1️ Embed destination_context
    dest_texts = dest_df["destination_context"]
    dest_emb = embed_col(model, tokenizer, dest_df, "destination_context", device)
    np.save(f"{out_prefix}_dest_emb_{gpu_id}.npy", dest_emb)
    np.save(f"{out_prefix}_dest_row_ids_{gpu_id}.npy", dest_df["row_id"].to_numpy())

    # 2️ Embed quote (unique passage_id)
    quote_texts = quote_df["quote"]
    quote_emb = embed_col(model, tokenizer, quote_df, "quote", device)
    np.save(f"{out_prefix}_quote_emb_{gpu_id}.npy", quote_emb)
    np.save(f"{out_prefix}_quote_row_ids_{gpu_id}.npy", quote_df["row_id"].to_numpy())

    print(f"[GPU {gpu_id}] Finished embedding and saved shards")


# Merge shards into single .npy files
def merge_shards(out_prefix, n_gpus, col_prefix):
    """
    col_prefix: "dest" or "quote"
    """
    all_embs = []
    all_ids = []

    for gpu_id in range(n_gpus):
        emb = np.load(f"{out_prefix}_{col_prefix}_emb_{gpu_id}.npy")
        ids = np.load(f"{out_prefix}_{col_prefix}_row_ids_{gpu_id}.npy")
        all_embs.append(emb)
        all_ids.append(ids)

    all_embs = np.vstack(all_embs)
    all_ids = np.concatenate(all_ids)

    # reorder by row_id
    order = np.argsort(all_ids)
    all_embs = all_embs[order]
    all_ids = all_ids[order]

    np.save(f"{out_prefix}_{col_prefix}_emb.npy", all_embs.astype(np.float32))
    np.save(f"{out_prefix}_{col_prefix}_row_ids.npy", all_ids)
    print(f"Merged {col_prefix} embeddings saved: {out_prefix}_{col_prefix}_emb.npy")


# Main pipeline
def gen_embedding():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Load data (already has row_id)
    meta_df = pd.read_parquet(config.LEPARD_DEST_DF)
    quote_df = meta_df[["passage_id", "quote", "row_id"]].drop_duplicates(subset="passage_id", keep="first").reset_index(drop=True)


    n_gpus = torch.cuda.device_count()
    dest_splits = np.array_split(meta_df, n_gpus)
    quote_splits = np.array_split(quote_df, n_gpus)

    out_prefix = config.LEPARD_LLM_EMB

    # Launch GPU processes
    procs = []
    for gpu_id in range(n_gpus):
        p = Process(target=run_on_gpu, args=(gpu_id, dest_splits[gpu_id], quote_splits[gpu_id], out_prefix))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Merge shards
    merge_shards(out_prefix, n_gpus, "dest")
    merge_shards(out_prefix, n_gpus, "quote")

# Entrypoint
if __name__ == "__main__":
    gen_embedding()



#  Use the embeddings later, save for quote
# meta_df = pd.read_parquet(...)
# dest_emb = np.load("llama_dest_emb.npy", mmap_mode="r")

# row_id = meta_df.loc[i, "row_id"]
# vector = dest_emb[row_id]
