"""
Generate user_sequence data for MovieLens data. All user history is kept, no filtering is done at this step.
Plot and save user_sequence length distribution into a figure. 
"""

import pandas as pd
import logging
import config
from utils import bagz_utils
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_preprocessing():

    meta_df = pd.read_parquet(config.ML_SID)  
    logger.info(meta_df.head(3))

    column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    reviews_df = pd.read_csv(config.MOVIELES_REVEIW_DATASET, sep='::', names=column_names, engine='python', header=None)

    # Generate user sequences
    merged_df = reviews_df.merge(meta_df[['MovieID', 'sid']], on='MovieID', how='left')
    sorted_df = merged_df.sort_values(by=['UserID', 'Timestamp'])
    num_items = sorted_df['MovieID'].nunique()
    logger.info(f"---Items: {num_items}")

    # Aggregate both sid and MovieID sequences
    user_sequences = sorted_df.groupby('UserID').agg({
        'sid': list,
        'MovieID': list
    }).reset_index()

    logger.info(user_sequences.head(3))

    # ------- Get sequence length information -------------
    lengths = user_sequences["MovieID"].apply(len)

    min_len = lengths.min()
    max_len = lengths.max()

    logger.info(f"Min MovieID length: {min_len}")
    logger.info(f"Max MovieID length: {max_len}")


    lengths = user_sequences["MovieID"].apply(len)

    plt.figure()
    plt.hist(lengths, bins=50)
    plt.xlabel("Length of MovieID list")
    plt.ylabel("Number of users")
    plt.title("Distribution of MovieID List Lengths")
    plt.savefig(os.path.join(config.PROCESSED_DATA_DIR, "ml1m_user_seq_length_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # -----------------------------------------------------

    records = []
    for idx, (_, row) in enumerate(user_sequences.iterrows(), start=1):
        record = {
            "id": idx,  # sequential ID
            "reviewerID": row["UserID"],
            "sequence": row["sid"],  # existing formatted SID sequence
            "MovieID_sequence": row["MovieID"],      # new ASIN sequence
        }
        records.append(record)

    bagz_utils.save_record(records, config.USER_SEQUENCE)


if __name__=="__main__":
    run_preprocessing()

