"""
Generate product embeddings using SentenceBert. 
"""

import pandas as pd
import logging
import multiprocessing
import psutil
from sentence_transformers import SentenceTransformer
import config
import pickle
from utils import bagz_utils

logger = logging.getLogger(__name__)

def gen_embedding():
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("all-mpnet-base-v2")

    # Read in meta data that has formatted text
    meta_df = bagz_utils.read_parquet(config.META_W_TEXT)
    logger.debug(meta_df.head(3))

    # Check cpu to set resource for embedding generation
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    logger.debug(f"Total RAM: {ram_gb:.2f} GB")
    logger.debug(multiprocessing.cpu_count())  # Total logical cores

    # Generate item embeddings
    logger.debug(f"Num of unique asins to be embedded: {meta_df.shape[0]}")
    embeddings = model.encode(
        meta_df["formatted_text"].tolist(),
        batch_size=2048,
        show_progress_bar=True
    )

    # Save the data
    # Convert to list of 1D arrays (or lists)
    meta_df["embedding"] = [emb.tolist() for emb in embeddings]

    bagz_utils.save_parquet(meta_df, config.META_W_EMBEDDING)
    logger.debug(meta_df.head(3))

    # Export iid -> embedding lookup
    iid_to_embedding = {row['IID']: row['embedding'] for _, row in meta_df.iterrows()}
    bagz_utils.save_object(iid_to_embedding, config.IID2EMBEDDING)

if __name__=="__main__":
    gen_embedding()