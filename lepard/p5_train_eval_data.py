"""
90%  training, 5% eval, 5% test
"""


import config
import pandas as pd
import importlib
from utils import bagz_utils
import numpy as np


SEED = 411

df = pd.read_parquet(config.LEPARD_SID)

# Shuffle indices
rng = np.random.default_rng(seed=SEED)
indices = rng.permutation(len(df))

# Compute split sizes
n = len(df)
train_end = int(0.9 * n)
eval_end = int(0.95 * n)

# Split indices
train_idx = indices[:train_end]
eval_idx = indices[train_end:eval_end]
test_idx = indices[eval_end:]

# Create splits
train_df = df.iloc[train_idx]
eval_df = df.iloc[eval_idx]
test_df = df.iloc[test_idx]

train_df.to_parquet(config.LEPARD_TRAIN, index=False)
eval_df.to_parquet(config.LEPARD_EVAL, index=False)
test_df.to_parquet(config.LEPARD_TEST, index=False)

