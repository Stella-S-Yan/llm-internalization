"""
Generate product embeddings using SentenceBert. 
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

logger = logging.getLogger(__name__)


# Function to format the text
def format_text(row):
    title = str(row["title"]) if pd.notna(row["title"]) else " "
    fine_category = str(row["fine_category"]) if pd.notna(row["fine_category"]) else " "
    brand = str(row["brand"]) if pd.notna(row["brand"]) else " "
    price = str(row["price"]) if pd.notna(row["price"]) else " "
    description = str(row["description"]) if pd.notna(row["description"]) else " "

    return f"{brand}\n{fine_category}\n{price}\n{title}\n{description}"


def gen_embedding(meta_df):
    # Restrict to GPU #2
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

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

    bagz_utils.save_parquet(meta_df, config.META_W_ALL_EMBEDDING)
    logger.debug(meta_df.head(3))

    

def read_in_data():
    # Read in meta data
    # The meta_{review_type}.json doesn't conform to the JSON spec and has inconsistent
    # use of quotes (' vs "). We use eval here as a workaround.
    good_rows = []
    with open(config.AMAZON_META_DATASET, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                row = ast.literal_eval(line)
                good_rows.append(row)
            except Exception as e:
                logger.info(f"Skipping line {i}: {e}")

    meta_df = pd.DataFrame(good_rows)
    num_items = meta_df.shape[0]
    logger.debug(f"Items: {num_items}")

    meta_df['fine_category'] = meta_df['categories'].apply(lambda x: x[-1][-1] if x and x[-1] else None)
    meta_df["formatted_text"] = meta_df.apply(format_text, axis=1)
    print(meta_df.head(3))

    return meta_df



def do_the_work():
    meta_df = read_in_data()
    gen_embedding(meta_df)


if __name__=="__main__":

    do_the_work()