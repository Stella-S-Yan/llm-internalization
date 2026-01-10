import pandas as pd
import logging
import config
from utils import bagz_utils
import random

logger = logging.getLogger(__name__)


def run_preprocessing():

    meta_df = pd.read_parquet(config.ML_SID)  
    logger.info(meta_df.head(3))

    column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    reviews_df = pd.read_csv(config.MOVIELES_REVEIW_DATASET, sep='::', names=column_names, engine='python', header=None)

    # Generate user sequences
    merged_df = reviews_df.merge(meta_df[['MovieID', 'sid']], on='MovieID', how='left')
    sorted_df = merged_df.sort_values(by=['UserID', 'Timestamp'])
    num_items = sorted_df['MovieID'].nunique()
    logger.debug(f"---Items: {num_items}")

    # Aggregate both sid and MovieID sequences
    user_sequences = sorted_df.groupby('UserID').agg({
        'sid': list,
        'MovieID': list
    }).reset_index()

    logger.info(user_sequences.head(3))

    records = []
    for idx, (_, row) in enumerate(user_sequences.iterrows(), start=1):
        record = {
            "id": idx,  # sequential ID
            "reviewerID": row["UserID"],
            "sequence": row["sid"],  # existing formatted SID sequence
            "MovieID_sequence": row["MovieID"],      # new ASIN sequence
        }
        records.append(record)

    rng = random.Random(411)  # fixed seed
    rng.shuffle(records)    

    bagz_utils.save_record(records, config.USER_SEQUENCE)


if __name__=="__main__":
    run_preprocessing()

