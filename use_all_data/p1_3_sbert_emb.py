"""
Generate product embeddings using SentenceBert. 

$ python sbert_embedding.py
"""

import pandas as pd
import logging
import multiprocessing
import psutil
from sentence_transformers import SentenceTransformer
import config
from utils import bagz_utils
import ast
import os
import numpy as np

logger = logging.getLogger(__name__)
SEED = 411


def gen_embedding(meta_df):
    # Restrict to GPU #2
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    # Check cpu to set resource for embedding generation
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    logger.debug(f"Total RAM: {ram_gb:.2f} GB")
    logger.debug(multiprocessing.cpu_count())  # Total logical cores

    # Generate item embeddings
    logger.debug(f"--- Num of unique asins to be embedded: {meta_df.shape[0]}")
    # Only keep unique formatted_text
    embeddings = model.encode(
        meta_df["formatted_text"].tolist(),
        batch_size=2048,
        show_progress_bar=True
    )

    # Save the data
    # Convert to list of 1D arrays (or lists)
    meta_df["sbert_embed"] = [emb.tolist() for emb in embeddings]

    bagz_utils.save_parquet(meta_df, config.META_OUTSIDE_EMB)
    logger.debug(meta_df.head(3))

    

def do_the_work():
    meta_df = bagz_utils.read_parquet(config.META_NORMALIZED)
    gen_embedding(meta_df)


if __name__=="__main__":

    do_the_work()