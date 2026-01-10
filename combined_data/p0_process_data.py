"""
Generate product embeddings using SentenceBert. 

$ python sbert_embedding.py
"""

import pandas as pd
import logging
import config
from utils import bagz_utils
import ast
import numpy as np

logger = logging.getLogger(__name__)

# Function to format the text
def format_text(row):
    def val(col):
        v = row[col]

        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""

        if isinstance(v, (list, tuple, np.ndarray)):
            return " > ".join(map(str, v)) if len(v) > 0 else ""

        return str(v)

    # Handle price separately to add $
    price_val = row['price']
    if isinstance(price_val, float) and not pd.isna(price_val):
        price_str = f"${price_val}"
    else:
        price_str = ""

    formatted = [
        f"title: {val('title')}",
        f"category: {val('fine_category')}",
        f"description: {val('description')}",
        f"brand: {val('brand')}",
        f"price: {price_str}",
    ]

    return "\n".join(formatted)


def normalize_meta_data():
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

    # Normalize description: set empty-like values to ""
    desc = meta_df['description'].astype(str).str.strip().str.lower()
    empty_mask = desc.isin(["", "nan", "none"])
    # Replace these rows with empty string
    meta_df.loc[empty_mask, 'description'] = ""

    # Normalize title
    title = meta_df['title'].astype(str).str.strip().str.lower()
    empty_mask = title.isin(["", "nan", "none"])
    # Replace these rows with empty string
    meta_df.loc[empty_mask, 'title'] = ""

    # Normalize brand
    brand = meta_df['brand'].astype(str).str.strip().str.lower()
    empty_mask = brand.isin(["", "nan", "none"])
    # Replace these rows with empty string
    meta_df.loc[empty_mask, 'brand'] = ""

    # filter bad xml descriptions
    mask_xml = meta_df['description'].str.contains(r'<[^>]+>', regex=True, na=False)
    meta_df.loc[mask_xml, 'description'] = ""
    print(f"--- Filtered {mask_xml.sum()} items with bad xml descriptions. Set description to empty string.")

    # Mark if the item is in review set
    review_df = pd.read_json(config.AMAZON_REVIEW_DATASET, lines=True)
    num_users = review_df["reviewerID"].nunique()
    print(f"---Users: {num_users}")
    meta_df["has_review"] = meta_df["asin"].isin(review_df["asin"]).astype(int)

    # Mark description length
    meta_df['desc_length'] = meta_df['description'].str.len()

    # Truncate too long description
    MAX_LENGTH = 1000
    meta_df['description'] = meta_df['description'].astype(str).str[:MAX_LENGTH]

    # Generate formatted_text
    meta_df["formatted_text"] = meta_df.apply(format_text, axis=1)

    bagz_utils.save_parquet(meta_df, config.META_NORMALIZED)
    print(meta_df.head(3))


if __name__=="__main__":

    normalize_meta_data()