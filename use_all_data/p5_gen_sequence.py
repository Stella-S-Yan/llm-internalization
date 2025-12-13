import pandas as pd
import logging
import config
from utils import bagz_utils

logger = logging.getLogger(__name__)


def run_preprocessing():
    meta_df = bagz_utils.read_parquet(config.META_ALL_SID) 
    logger.info(meta_df.head(3))

    review_df = pd.read_json(config.AMAZON_REVIEW_DATASET, lines=True)
    num_users = review_df["reviewerID"].nunique()
    logger.debug(f"---Users: {num_users}")

    # Generate user sequences
    merged_df = review_df.merge(meta_df[['asin', 'formatted_sid']], on='asin', how='left')
    sorted_df = merged_df.sort_values(by=['reviewerID', 'unixReviewTime'])
    num_items = sorted_df['asin'].nunique()
    logger.debug(f"---Items: {num_items}")

    # Extract prefix: first 3 SID tokens
    # sorted_df["sid_prefix_lst"] = sorted_df["formatted_sid"].str.split().str[:3]

    # Aggregate both formatted_sid and asin sequences
    user_sequences = sorted_df.groupby('reviewerID').agg({
        'formatted_sid': list,
        'asin': list
    }).reset_index()

    logger.info(user_sequences.head(3))

    records = []
    for idx, (_, row) in enumerate(user_sequences.iterrows(), start=1):
        record = {
            "id": idx,  # sequential ID
            "reviewerID": row["reviewerID"],
            "sequence": row["formatted_sid"],  # existing formatted SID sequence
            "asin_sequence": row["asin"],      # new ASIN sequence
        }
        records.append(record)

    bagz_utils.save_record(records, config.USER_SEQUENCE)


if __name__=="__main__":
    run_preprocessing()

