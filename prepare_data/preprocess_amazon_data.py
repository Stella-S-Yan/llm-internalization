import pandas as pd
import ast
import logging
import config
from utils import tokenizer_utils
from utils import bagz_utils
import os

logger = logging.getLogger(__name__)


# Function to format the text
def format_text(row):
    description = str(row["description"]) if pd.notna(row["description"]) else " "
    title = str(row["title"]) if pd.notna(row["title"]) else " "
    price = str(row["price"]) if pd.notna(row["price"]) else " "
    brand = str(row["brand"]) if pd.notna(row["brand"]) else " "
    fine_category = str(row["fine_category"]) if pd.notna(row["fine_category"]) else " "

    return f"{brand}\n{fine_category}\n{price}\n{title}\n{description}"


def run_preprocessing():
    # Read in review data
    review_df = pd.read_json(config.AMAZON_REVIEW_DATASET, lines=True)
    num_users = review_df.shape[0]

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

    # Keep only core items
    meta_df = meta_df[meta_df['asin'].isin(review_df['asin'])].copy()
    num_items = meta_df.shape[0]

    logger.debug(f"Users: {num_users}, Items: {num_items}")

    # Add sequential UIDs starting from 1
    # user_id_map = {uid: i for i, uid in enumerate(review_df['reviewerID'].unique(), start=1)}
    # review_df['UID'] = review_df['reviewerID'].map(user_id_map)

    # review_df = review_df.sort_values(by=['UID'])

    # Add sequential IIDs starting from 1
    item_id_map = {iid: i for i, iid in enumerate(review_df['asin'].unique(), start=1)}
    review_df['IID'] = review_df['asin'].map(item_id_map)

    # # Apply hash trick to reviwerID and generate 2k userID
    # review_df['hashed_userID'] = review_df['reviewerID'].apply(format_sid.hash_user_id)
    # uid_to_hashed = dict(zip(review_df["UID"], review_df["hashed_userID"]))
    # bagz_utils.save_object(uid_to_hashed, config.USER2HASHED)
    

    # Build a tokenizer
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    tokenizer_utils.build_tokenizer()

    # Add the IID to meta_df
    review_dedup = review_df.drop_duplicates(subset='asin', keep='first')
    meta_df['IID'] = meta_df['asin'].map(review_dedup.set_index('asin')['IID'])

    # Generate user sequences
    sorted_df = review_df.sort_values(by=['reviewerID', 'unixReviewTime'])
    user_sequences = sorted_df.groupby('reviewerID')['IID'].agg(list).reset_index()

    records = []
    for _, row in user_sequences.iterrows():
        record = {
            "reviewerID": row["reviewerID"],
            "sequence": row["IID"],  
        }
        records.append(record)
    bagz_utils.save_record(records, config.USER_SEQUENCE)


    # Only keep the finest cateogy
    meta_df['fine_category'] = meta_df['categories'].apply(lambda x: x[-1][-1] if x and x[-1] else None)
    meta_df["formatted_text"] = meta_df.apply(format_text, axis=1)
    bagz_utils.save_parquet(meta_df, config.META_W_TEXT)


if __name__=="__main__":
    run_preprocessing()

