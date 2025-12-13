"""
Remove near-duplicate embeddings from rqvae training set.
"""


import faiss
import numpy as np
import config
from utils import bagz_utils
import pandas as pd

df_all = []
for group_id in range(8):
    print(f"get file {config.META_W_ALL_TWO_EMB}_{group_id}")
    df = bagz_utils.read_parquet(f"{config.META_W_ALL_TWO_EMB}_{group_id}")
    df_all.append(df)
meta_df = pd.concat(df_all, ignore_index=True)

print("All data shape: ", meta_df.shape)

# Drop rows with no description
drop_cnt = meta_df['description'].isnull().sum()
print(f"To drop {drop_cnt} rows with no description, which is {drop_cnt/meta_df.shape[0]:.2%} of total")
meta_df = meta_df.dropna(subset=['description'])
print("After dropping no description: ", meta_df.shape)

raw_item_embeddings = meta_df['embedding'].tolist()

# Ensure all arrays are writable
raw_item_embeddings = [np.array(emb, dtype=np.float32, copy=True) for emb in raw_item_embeddings]
