"""
Store embeddings outside of dataframe using multi-GPU SentenceTransformer.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



MODEL_NAME = "thenlper/gte-large"

# ---------------------------------------------------
# Multi-GPU embedding helper
# ---------------------------------------------------
def encode_texts_multi_gpu(texts, model, pool, batch_size=1024):
    """
    texts: List[str]
    returns: np.ndarray [N, D] float32
    """
    embeddings = model.encode(
        texts,
        pool=pool,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")


# Main embedding pipeline
def gen_embedding():
    # 1️ Load data
    column_names = ['MovieID', 'Title', 'Genre']

    # 1m
    movies_1m_file = os.path.join(config.DATA_DIR, 'MovieLens', 'ml-1m', 'movies.dat')
    movies_1m_df = pd.read_csv(movies_1m_file, sep='::', names=column_names, engine='python', header=None, encoding='ISO-8859-1')
    print(movies_1m_df.shape)

    # 20m
    movies_20m_file = os.path.join(config.DATA_DIR, 'MovieLens', 'ml-20m', 'movies.csv')
    movies_20m_df = pd.read_csv(movies_20m_file, engine='python', encoding='ISO-8859-1')
    movies_20m_df.columns = column_names
    print(movies_20m_df.shape)

    movies_df = (
        pd.concat([movies_20m_df, movies_1m_df], ignore_index=True)
        .drop_duplicates(subset=['Title'], keep='first')
    )
    
    assert movies_df.shape[0] == 27805
    
    # Stable row id (CRITICAL)
    movies_df["row_id"] = movies_df.index.astype("int64")

    logger.info(f"Merged dataframe: {movies_df.shape}")

    movies_df["formatted_text"] = movies_df["Title"].fillna("") + "  " + movies_df["Genre"].fillna("")

    formatted_text = (
        movies_df["formatted_text"]
        .astype(str)
        .tolist()
    )

    # 3️ Load model + start multi-GPU pool
    model = SentenceTransformer(MODEL_NAME)
    pool = model.start_multi_process_pool()  # uses all visible GPUs

    try:
        # 4️ Encode formatted test
        logger.info("Encoding formatted with GTE")
        dest_embeddings = encode_texts_multi_gpu(
            texts=formatted_text,
            model=model,
            pool=pool
        )

        np.save(
            os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_outside_emb") + "_gte.npy",
            dest_embeddings
        )

        
    finally:
        model.stop_multi_process_pool(pool)

    # 6️ Save metadata ONLY (no embeddings)
    movies_df.to_parquet(
        os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_row_id_df.parquet"),
        engine="pyarrow",
        compression="zstd", 
        index=False
    )

    logger.info("GTE embedding pipeline finished successfully")


# Entrypoint
if __name__ == "__main__":
    gen_embedding()
