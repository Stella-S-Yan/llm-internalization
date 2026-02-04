import os
import torch
import logging
import psutil
import multiprocessing
from sentence_transformers import SentenceTransformer
from utils import bagz_utils
import config

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
    model_name = "sentence-transformers/sentence-t5-base"
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
        batch_size=2048,          # High batch size for b200
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
    # Load data
    meta_df = bagz_utils.read_parquet(config.META_NORMALIZED)
    
    # Deduplicate and reset index to prevent merging errors
    meta_df = meta_df.drop_duplicates(subset=["formatted_text"]).reset_index(drop=True)
    
    # Hardware/RAM Logging
    ram_gb = psutil.virtual_memory().total / (2048 ** 3)
    logger.info(f"System RAM: {ram_gb:.2f} GB | CPU Cores: {multiprocessing.cpu_count()}")
    
    gen_embedding(meta_df)

if __name__ == "__main__":
    # if __name__ protection is mandatory for multi-process CUDA spawning
    do_the_work()