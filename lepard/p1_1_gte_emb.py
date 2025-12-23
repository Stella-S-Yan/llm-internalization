import torch
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
import config
from utils import bagz_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Multi-GPU embedding helper (modern ST usage)
# ---------------------------------------------------
def embed_column_multi_gpu(df, text_col, output_col, model, pool):
    # Convert all entries to string to avoid "unhashable type: dict"
    if text_col == "destination_context":
        # Keep only the last 512 characters of each text
        texts = df[text_col].fillna("").astype(str).apply(lambda x: x[-512:]).tolist()
    else:
        texts = df[text_col].fillna("").astype(str).tolist()

    # Use encode() with pool argument for multi-GPU / multi-process
    embeddings = model.encode(
        texts,
        pool=pool,
        batch_size=1024,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    df[output_col] = embeddings.tolist()


# ---------------------------------------------------
# Main embedding pipeline (MULTI-GPU)
# ---------------------------------------------------
def gen_embedding(meta_df):
    logger.info(f"Total rows: {len(meta_df)}")

    # Load model ONCE
    model_name = "thenlper/gte-large"
    model = SentenceTransformer(model_name)

    # Start multi-GPU / multi-process pool
    pool = model.start_multi_process_pool()

    try:
        # 1. Embed destination_context (ALL rows)
        logger.info(f"Embedding destination_context ({len(meta_df)})")
        embed_column_multi_gpu(
            meta_df,
            text_col="destination_context",
            output_col="destination_context_gte",
            model=model,
            pool=pool
        )

        # 2. Deduplicate quotes
        quote_df = (
            meta_df[["passage_id", "quote"]]
            .drop_duplicates("passage_id", keep="first")
            .reset_index(drop=True)
        )

        logger.info(f"Embedding quote ({len(quote_df)})")
        # 3. Embed quote (UNIQUE passage_id)
        embed_column_multi_gpu(
            quote_df,
            text_col="quote",
            output_col="quote_gte",
            model=model,
            pool=pool
        )

        # 4. Merge quote embeddings back
        meta_df = meta_df.merge(
            quote_df[["passage_id", "quote_gte"]],
            on="passage_id",
            how="left"
        )

        # 5. Save embeddings
        bagz_utils.save_parquet(meta_df, config.LEPARD_OUTSIDE_EMB)
        logger.info("Embedding completed successfully")

    finally:
        # Clean up the pool
        model.stop_multi_process_pool(pool)


# ---------------------------------------------------
# Entrypoint
# ---------------------------------------------------
def do_the_work():
    df = pd.read_csv(config.DATA_DIR / config.DATA_SOURCE / "top_10000_data.csv")
    gen_embedding(df)


if __name__ == "__main__":
    do_the_work()
