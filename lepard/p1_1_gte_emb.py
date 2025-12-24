"""
Store embeddings outside of dataframe using multi-GPU SentenceTransformer.
"""


import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
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
    df = pd.read_csv(config.DATA_DIR / config.DATA_SOURCE / "top_10000_data.csv")
    df = df.reset_index(drop=True)

    # Stable row id (CRITICAL)
    df["row_id"] = df.index.astype("int64")

    logger.info(f"Loaded dataframe: {df.shape}")

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

    # 3️ Load model + start multi-GPU pool
    model = SentenceTransformer(MODEL_NAME)
    pool = model.start_multi_process_pool()  # uses all visible GPUs

    try:
        # 4️ Encode destination_context
        logger.info("Encoding destination_context with GTE")
        dest_embeddings = encode_texts_multi_gpu(
            texts=dest_texts,
            model=model,
            pool=pool
        )

        np.save(
            config.LEPARD_OUTSIDE_EMB + "_dest_gte.npy",
            dest_embeddings
        )

        # 5️ Encode quote (unique passage_id)
        logger.info("Encoding quote with GTE")
        quote_embeddings = encode_texts_multi_gpu(
            texts=quote_texts,
            model=model,
            pool=pool
        )

        np.save(
            config.LEPARD_OUTSIDE_EMB + "_quote_gte.npy",
            quote_embeddings
        )

    finally:
        model.stop_multi_process_pool(pool)

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

    logger.info("GTE embedding pipeline finished successfully")


# Entrypoint
if __name__ == "__main__":
    gen_embedding()


# Use the embeddings later
# destination_context

# df = pd.read_parquet(config.LEPARD_DEST_DF)
# dest_emb = np.load(config.LEPARD_OUTSIDE_EMB + "_dest_gte.npy", mmap_mode="r")

# row_id = meta.loc[i, "row_id"]
# vec = dest_emb[row_id]

# Quote

# quote_meta = pd.read_parquet(config.LEPARD_QUOTE_DF)
# quote_emb = np.load(config.LEPARD_OUTSIDE_EMB + "_quote_gte.npy", mmap_mode="r")

# pid_to_idx = dict(zip(quote_meta.passage_id, quote_meta.index))
# vec = quote_emb[pid_to_idx[passage_id]]