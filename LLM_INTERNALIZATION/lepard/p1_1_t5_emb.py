import os
import torch
import logging
import psutil
import multiprocessing
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

from LLM_INTERNALIZATION import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gen_embedding(text, ttype):
    # 1. Hardware Setup
    num_gpus = torch.cuda.device_count()
    target_devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else ["cpu"]
    logger.info(f"Detected {num_gpus} GPUs. Using devices: {target_devices}")

    # 2. Load and Optimize Model
    model_name = "sentence-transformers/sentence-t5-base"
    model = SentenceTransformer(model_name)
    
    # Enable FP16 (Half Precision) to double throughput and halve VRAM usage
    if num_gpus > 0:
        model.half()

    # 3. Initialize Multi-GPU Pool
    # This creates a separate process per GPU to bypass Python's GIL
    pool = model.start_multi_process_pool(target_devices=target_devices)

    logger.info(f"Starting embedding of {len(text)} items...")

    # 4. Embed text
    embeddings = model.encode(
        sentences=text,
        pool=pool,                # Triggers multi-GPU parallel workers
        batch_size=2048,          # High batch size for b200
        chunk_size=50000,         # Large chunks for high-throughput distribution
        normalize_embeddings=True, # L2 normalization for cosine similarity
        show_progress_bar=True
    )

    # 5. Cleanup Pool (Critical to free up GPU memory)
    model.stop_multi_process_pool(pool)

    # 6. Save results
    np.save(
        f"{config.LEPARD_OUTSIDE_EMB}_{config.REVIEW_TYPE}_{ttype}_t5.npy",
        embeddings
    )
    

def do_the_work():
    # 1. Load data
    if config.REVIEW_TYPE == "10k":
        df = pd.read_csv(config.DATA_DIR / config.DATA_SOURCE / "top_10000_data.csv")
    elif config.REVIEW_TYPE == "20k":
        df = pd.read_csv(config.DATA_DIR / config.DATA_SOURCE / "top_20000_data.csv")
    elif config.REVIEW_TYPE == "50k":
        df = pd.read_csv(config.DATA_DIR / config.DATA_SOURCE / "top_50000_data.csv")
    df = df.reset_index(drop=True)

    # Stable row id (CRITICAL)
    df["row_id"] = df.index.astype("int64")
    logger.info(f"Loaded dataframe: {df.shape}")

    df.to_parquet(
        config.LEPARD_DEST_DF,
        engine="pyarrow",
        compression="zstd", 
        index=False
    )

    # 2️ Prepare texts
    dest_texts = (
        df["destination_context"]
        .fillna("")
        .astype(str)
        .apply(lambda x: x[-512:])  # keep last 512 chars
        .tolist()
    )

    quote_df = (
        df[["passage_id", "quote"]]
        .drop_duplicates("passage_id", keep="first")
        .reset_index(drop=True)
    )

    quote_texts = quote_df["quote"].fillna("").astype(str).tolist()

    logger.info(f"Destination texts: {len(dest_texts)}")
    logger.info(f"Unique quotes: {len(quote_texts)}")


    # 6️ Save metadata ONLY (no embeddings)
    df.to_parquet(
        config.LEPARD_DEST_DF,
        engine="pyarrow",
        compression="zstd", 
        index=False
    )

    quote_df[["passage_id"]].to_parquet(
        config.LEPARD_QUOTE_DF,
        engine="pyarrow",
        index=False
    )
    
    # Hardware/RAM Logging
    ram_gb = psutil.virtual_memory().total / (2048 ** 3)
    logger.info(f"System RAM: {ram_gb:.2f} GB | CPU Cores: {multiprocessing.cpu_count()}")

    gen_embedding(quote_texts, "quote")
    gen_embedding(dest_texts, "dest")

if __name__ == "__main__":
    # if __name__ protection is mandatory for multi-process CUDA spawning
    do_the_work()