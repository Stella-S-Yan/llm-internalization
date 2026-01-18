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

    # Read in df with sid
    file_name = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_sid_df.parquet")
    meta_df = pd.read_parquet(file_name) 
    assert meta_df.shape[0] == 27805

    movies_column_names = ['MovieID', 'Title', 'Genre']
    ratings_column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    
    # Generate user sequences
    for review_type in ['1m', "20m"]:
        if review_type == "1m":
            
            movies_1m_file = os.path.join(config.DATA_DIR, 'MovieLens', 'ml-1m', 'movies.dat')
            movies_1m_df = pd.read_csv(movies_1m_file, sep='::', names=movies_column_names, engine='python', header=None, encoding='ISO-8859-1')
            movies_sid_df = movies_1m_df.merge(
                meta_df.drop(columns=['MovieID', 'Genre']),
                on='Title',
                how='left'
            )
            assert movies_sid_df.shape[0] == movies_1m_df.shape[0]

            ratings_1m_file = os.path.join(config.DATA_DIR, 'MovieLens', 'ml-1m', 'ratings.dat')
            ratings_1m_df = pd.read_csv(ratings_1m_file, sep='::', names=ratings_column_names, engine='python', header=None)
            merged_df = ratings_1m_df.merge(movies_sid_df[['MovieID', 'sid']], on='MovieID', how='left')
            assert merged_df.shape[0] == ratings_1m_df.shape[0]
            
        elif review_type == "20m":
            movies_20m_file = os.path.join(config.DATA_DIR, 'MovieLens', 'ml-20m', 'movies.csv')
            movies_20m_df = pd.read_csv(movies_20m_file, engine='python', encoding='ISO-8859-1')
            movies_20m_df.columns = movies_column_names
            movies_sid_df = movies_20m_df.merge(
                meta_df.drop(columns=['MovieID', 'Genre']),
                on='Title',
                how='left'
            )
            assert movies_sid_df.shape[0] == movies_20m_df.shape[0]
            
            ratings_20m_file = os.path.join(config.DATA_DIR, 'MovieLens', 'ml-20m', 'ratings.csv')
            ratings_20m_df = pd.read_csv(ratings_20m_file, engine='python')
            ratings_20m_df.columns = ratings_column_names
            merged_df = ratings_20m_df.merge(movies_sid_df[['MovieID', 'sid']], on='MovieID', how='left')
            assert merged_df.shape[0] == ratings_20m_df.shape[0]
        
        sorted_df = merged_df.sort_values(by=['UserID', 'Timestamp'])
        num_items = sorted_df['MovieID'].nunique()
        logger.info(f"---{review_type} Items: {num_items}")

        # Aggregate both sid and MovieID sequences
        user_sequences = sorted_df.groupby('UserID').agg({
            'sid': list,
            'MovieID': list
        }).reset_index()

        logger.info(user_sequences.head(3))

        rows_with_nan = user_sequences[user_sequences['sid'].apply(lambda lst: any(x != x for x in lst))]
        assert rows_with_nan.shape[0] == 0

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
        plt.title(f"Distribution of MovieID List Lengths {review_type}")
        plt.savefig(os.path.join(config.PROCESSED_DATA_DIR, f"ml-{review_type}_user_seq_length_distribution.png"), dpi=300, bbox_inches="tight")
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

        sequence_file = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{review_type}_user_sequence.bagz" )
        bagz_utils.save_record(records, sequence_file)


if __name__=="__main__":
    run_preprocessing()

