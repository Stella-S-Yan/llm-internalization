import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"


import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from multiprocessing import Process, set_start_method
from tqdm import tqdm
import config

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


def embed_col(model, tokenizer, df, device, batch_size=128):
    """
    Embed a column in batches on a single GPU.
    Truncates destination_context to last 512 characters.
    Returns embeddings as a numpy array.
    """
    embeddings = []

    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"Embedding formatted_text"):
        batch_texts = df["formatted_text"].iloc[start_idx:start_idx+batch_size].fillna("").astype(str)
        batch_texts = batch_texts.tolist()

        emb_batch = get_embedding_last_token(batch_texts, model, tokenizer, device)
        embeddings.append(emb_batch)

        del emb_batch
        torch.cuda.empty_cache()

    return np.vstack(embeddings)  # shape [num_rows, hidden_dim]


# GPU worker
def run_on_gpu(gpu_id, meta_df, out_prefix):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Running on {device}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device}, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    emb = embed_col(model, tokenizer, meta_df, device, batch_size=128)
    np.save(f"{out_prefix}_{gpu_id}.npy", emb)
    np.save(f"{out_prefix}_row_ids_{gpu_id}.npy", meta_df["row_id"].to_numpy())

    print(f"[GPU {gpu_id}] Finished embedding and saved shards")


# Merge shards into single .npy files
def merge_shards(out_prefix, n_gpus):
    """
    col_prefix: "dest" or "quote"
    """
    all_embs = []
    all_ids = []

    for gpu_id in range(n_gpus):
        emb = np.load(f"{out_prefix}_{gpu_id}.npy")
        ids = np.load(f"{out_prefix}_row_ids_{gpu_id}.npy", allow_pickle=True)
        all_embs.append(emb)
        all_ids.append(ids)

    all_embs = np.vstack(all_embs)
    all_ids = np.concatenate(all_ids)

    # reorder by row_id
    order = np.argsort(all_ids)
    all_embs = all_embs[order]
    all_ids = all_ids[order]

    np.save(f"{out_prefix}_emb.npy", all_embs.astype(np.float32))
    np.save(f"{out_prefix}_row_ids.npy", all_ids)
    print(f"Merged embeddings saved: {out_prefix}_emb.npy")

    # cleanup shard files
    for gpu_id in range(n_gpus):
        os.remove(f"{out_prefix}_{gpu_id}.npy")
        os.remove(f"{out_prefix}_row_ids_{gpu_id}.npy")
    print("Cleaned up shard files.")


# Main pipeline
def gen_embedding():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Load data (already has row_id)
    file_name = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_row_id_df.parquet")
    movies_df = pd.read_parquet(str(file_name))
    
    n_gpus = torch.cuda.device_count()
    df_splits = np.array_split(movies_df, n_gpus)
    
    out_prefix = str(os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_llm_emb"))

    # Launch GPU processes
    procs = []
    for gpu_id in range(n_gpus):
        p = Process(target=run_on_gpu, args=(gpu_id, df_splits[gpu_id], out_prefix))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Merge shards
    merge_shards(out_prefix, n_gpus)


# Entrypoint
if __name__ == "__main__":
    gen_embedding()