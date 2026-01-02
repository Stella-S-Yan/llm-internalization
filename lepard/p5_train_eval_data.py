"""
90%  training, 5% eval, 5% test
"""


import config
import pandas as pd
import importlib
from utils import bagz_utils
import numpy as np



SEED = 411

def split_sets(df, dataset_type="50k", fraction=(0.9, 0.05, 0.05)):
    """Splits the dataframe into training, eval, and test sets based on the given fraction."""
    # Shuffle indices
    rng = np.random.default_rng(seed=SEED)
    indices = rng.permutation(len(df))

    # Compute split sizes
    n = len(df)
    train_end = int(fraction[0] * n)
    eval_end = train_end + int(fraction[1] * n)

    # Split indices
    train_idx = indices[:train_end]
    eval_idx = indices[train_end:eval_end]
    test_idx = indices[eval_end:]

    # Create splits
    train_df = df.iloc[train_idx]
    eval_df = df.iloc[eval_idx]
    test_df = df.iloc[test_idx]

    if dataset_type == "50k":
        train_df.to_parquet(config.LEPARD_50k_TRAIN, index=False)
        eval_df.to_parquet(config.LEPARD_50k_EVAL, index=False)
        test_df.to_parquet(config.LEPARD_50k_TEST, index=False)
    elif dataset_type == "20k":
        eval_df.to_parquet(config.LEPARD_20k_EVAL, index=False)
        test_df.to_parquet(config.LEPARD_20k_TEST, index=False)
    elif dataset_type == "10k":
        eval_df.to_parquet(config.LEPARD_10k_EVAL, index=False)
        test_df.to_parquet(config.LEPARD_10k_TEST, index=False)


# 50k data
df = pd.read_parquet(config.LEPARD_SID)
split_sets(df, dataset_type="50k")

cols = [
    "dest_id", "source_id", "dest_date", "dest_court", "dest_name",
    "dest_cite", "source_date", "source_court", "source_name",
    "source_cite", "passage_id", "quote", "destination_context"
]
# 20k data
df2 = pd.read_csv(config.DATA_DIR / config.DATA_SOURCE / "top_20000_data.csv")
df_20k = df.merge(df2[cols], on=cols, how="inner")
split_sets(df_20k, dataset_type="20k")

df2 = pd.read_csv(config.DATA_DIR / config.DATA_SOURCE / "top_10000_data.csv")
df_10k = df.merge(df2[cols], on=cols, how="inner")
split_sets(df_10k, dataset_type="10k")

