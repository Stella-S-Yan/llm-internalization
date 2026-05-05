import numpy as np
import pandas as pd

from LLM_INTERNALIZATION import config

SEED = 411


def split_data(df, fraction=(0.9, 0.05, 0.05)):
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(df))

    n = len(df)
    train_end = int(fraction[0] * n)
    eval_end = train_end + int(fraction[1] * n)

    train_df = df.iloc[indices[:train_end]]
    eval_df = df.iloc[indices[train_end:eval_end]]
    test_df = df.iloc[indices[eval_end:]]

    train_df.to_parquet(config.LEPARD_TRAIN, index=False)
    eval_df.to_parquet(config.LEPARD_EVAL, index=False)
    test_df.to_parquet(config.LEPARD_TEST, index=False)

    return train_df, eval_df, test_df


SUBSET_COLS = [
    "dest_id", "source_id", "dest_date", "dest_court", "dest_name",
    "dest_cite", "source_date", "source_court", "source_name",
    "source_cite", "passage_id", "quote", "destination_context"
]


# Load full dataset
df = pd.read_parquet(config.LEPARD_SID)

# Split once
train_split, eval_split, test_split = split_data(df)

print(f"{config.REVIEW_TYPE} split: train={len(train_split)}, eval={len(eval_split)}, test={len(test_split)}")


