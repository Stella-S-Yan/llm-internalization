import os
import torch
import logging
import psutil
import multiprocessing
from sentence_transformers import SentenceTransformer
from utils import bagz_utils
import config
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gen_embedding(meta_df):
    # 1. Hardware Setup
    # With 80GB, we want to ensure we utilize all available GPUs on the system
    num_gpus = torch.cuda.device_count()
    target_devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else ["cpu"]
    logger.info(f"Detected {num_gpus} GPUs. Using devices: {target_devices}")

    # 2. Load and Optimize Model
    # Sentence-T5-Base is efficient; Large/XL versions would also fit easily in 80GB
    model_name = "sentence-transformers/sentence-t5-xl"
    model = SentenceTransformer(model_name)
    
    # Enable FP16 (Half Precision) to double throughput and halve VRAM usage
    if num_gpus > 0:
        model.half()

    # 3. Initialize Multi-GPU Pool
    # This creates a separate process per GPU to bypass Python's GIL
    pool = model.start_multi_process_pool(target_devices=target_devices)

    logger.info(f"Starting embedding of {len(meta_df)} items...")

    # 4. Generate Embeddings with 80GB-Optimized Parameters
    # batch_size: 1024-2048 is safe for T5-Base on 80GB VRAM
    # chunk_size: Large chunks (50k+) reduce communication overhead
    embeddings = model.encode(
        sentences=meta_df["formatted_text"].tolist(),
        pool=pool,                # Triggers multi-GPU parallel workers
        batch_size=1024,          # High batch size for 80GB A100
        chunk_size=50000,         # Large chunks for high-throughput distribution
        normalize_embeddings=True, # L2 normalization for cosine similarity
        show_progress_bar=True
    )

    # 5. Cleanup Pool (Critical to free up GPU memory)
    model.stop_multi_process_pool(pool)

    # 6. Save Results
    meta_df["t5_embed"] = list(embeddings)
    bagz_utils.save_parquet(meta_df, config.META_OUTSIDE_EMB)
    logger.info("Embeddings saved successfully.")

def do_the_work():
    
    column_names = ['MovieID', 'Title', 'Genre']
    if config.REVIEW_TYPE == "1m":
        movies_file = os.path.join(config.DATA_DIR, config.DATA_SOURCE, f'ml-{config.REVIEW_TYPE}', 'movies.dat')
        movies_df = pd.read_csv(movies_file, sep='::', names=column_names, engine='python', header=None, encoding='ISO-8859-1')
    elif config.REVIEW_TYPE == "20m":
        movies_file = os.path.join(config.DATA_DIR, config.DATA_SOURCE, f'ml-{config.REVIEW_TYPE}', 'movies.csv')
        movies_df = pd.read_csv(movies_file, engine='python', encoding='ISO-8859-1')
        movies_df.columns = column_names

    print(movies_df.shape)
    
    # Stable row id (CRITICAL)
    movies_df["row_id"] = movies_df.index.astype("int64")

    movies_df["formatted_text"] = movies_df["Title"].fillna("") + "  " + movies_df["Genre"].fillna("")

    # Save metadata ONLY (no embeddings)
    movies_df.to_parquet(
        os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_row_id_df.parquet"),
        engine="pyarrow",
        compression="zstd", 
        index=False
    )

    # Hardware/RAM Logging
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    logger.info(f"System RAM: {ram_gb:.2f} GB | CPU Cores: {multiprocessing.cpu_count()}")
    
    gen_embedding(movies_df)

if __name__ == "__main__":
    # if __name__ protection is mandatory for multi-process CUDA spawning
    do_the_work()