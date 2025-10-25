
import pandas as pd
import logging
import multiprocessing
import psutil
from sentence_transformers import SentenceTransformer
import config
import pickle
from utils import bagz_utils
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def add_formatted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two formatted columns:
    - formatted_source: combines source_court, source_cite, and quote
    - formatted_destination: combines dest_court, dest_cite, and destination_context
    """
    df = df.copy()

    # Create formatted source text
    df["formatted_source"] = (
        df["source_court"].fillna("") + " | " +
        df["source_cite"].fillna("") + " | " +
        df["quote"].fillna("")
    )

    # Create formatted destination text
    df["formatted_destination"] = (
        df["dest_court"].fillna("") + " | " +
        df["dest_cite"].fillna("") + " | " +
        df["destination_context"].fillna("")
    )

    return df



def gen_embedding(split, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    # Check cpu to set resource for embedding generation
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    logger.debug(f"Total RAM: {ram_gb:.2f} GB")
    logger.debug(multiprocessing.cpu_count())  # Total logical cores


    file_path = f"/usr/local/google/home/stellasyan/Documents/LePaRD/data/{split}set_top_10000.csv"
    df = pd.read_csv(file_path)
    df_formatted = add_formatted_columns(df)


    # Generate item embeddings
    logger.debug(f"Num of unique asins to be embedded: {df_formatted.shape[0] * 2}")
    embeddings = model.encode(
        df_formatted["formatted_source"].tolist(),
        batch_size=2048,
        show_progress_bar=True, 
        device=device,
        convert_to_numpy=True,   # avoids slow .tolist() later
    )
    df_formatted["source_embedding"] = [emb.tolist() for emb in embeddings]

    embeddings = model.encode(
        df_formatted["formatted_destination"].tolist(),
        batch_size=2048,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True,   # avoids slow .tolist() later
    )
    df_formatted["destination_embedding"] = [emb.tolist() for emb in embeddings]


    bagz_utils.save_parquet(df_formatted, save_path)
    logger.info(df_formatted.head(3))

    

if __name__=="__main__":
    # gen_embedding("train", config.LEPARD_W_EMBEDDING_TRAIN)
    gen_embedding("dev", config.LEPARD_W_EMBEDDING_DEV)
    # gen_embedding("test", config.LEPARD_W_EMBEDDING_TEST)