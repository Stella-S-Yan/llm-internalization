import numpy as np
import pandas as pd
import config

SEED = 411


# -------------------------------------------------
# 1. Split 50k ONCE
# -------------------------------------------------
def split_50k(df, fraction=(0.9, 0.05, 0.05)):
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(df))

    n = len(df)
    train_end = int(fraction[0] * n)
    eval_end = train_end + int(fraction[1] * n)

    train_df = df.iloc[indices[:train_end]]
    eval_df = df.iloc[indices[train_end:eval_end]]
    test_df = df.iloc[indices[eval_end:]]

    train_df.to_parquet(config.LEPARD_50k_TRAIN, index=False)
    eval_df.to_parquet(config.LEPARD_50k_EVAL, index=False)
    test_df.to_parquet(config.LEPARD_50k_TEST, index=False)

    return train_df, eval_df, test_df


# -------------------------------------------------
# 2. Columns defining subset identity
# -------------------------------------------------
SUBSET_COLS = [
    "dest_id", "source_id", "dest_date", "dest_court", "dest_name",
    "dest_cite", "source_date", "source_court", "source_name",
    "source_cite", "passage_id", "quote", "destination_context"
]


# -------------------------------------------------
# 3. Create subset eval/test by FILTERING ONLY
# -------------------------------------------------
def derive_subset(eval_50k, test_50k, subset_csv, eval_out, test_out):
    subset_df = (
        pd.read_csv(subset_csv)
        .drop_duplicates()
        [SUBSET_COLS]
    )
    print(f"Deriving subset from {len(subset_df)} entries in {subset_csv}")

    eval_subset = eval_50k.merge(subset_df, on=SUBSET_COLS, how="inner")
    test_subset = test_50k.merge(subset_df, on=SUBSET_COLS, how="inner")

    eval_subset.to_parquet(eval_out, index=False)
    test_subset.to_parquet(test_out, index=False)

    return eval_subset, test_subset


# -------------------------------------------------
# 4. Main workflow
# -------------------------------------------------

# Load full 50k dataset
df_50k = pd.read_parquet(config.LEPARD_SID)

# Split once
train_50k, eval_50k, test_50k = split_50k(df_50k)

print(f"50k split: train={len(train_50k)}, eval={len(eval_50k)}, test={len(test_50k)}")

# ---- 20k ----
top_20k_csv = config.DATA_DIR / config.DATA_SOURCE / "top_20000_data.csv"

eval_20k, test_20k = derive_subset(
    eval_50k,
    test_50k,
    top_20k_csv,
    config.LEPARD_20k_EVAL,
    config.LEPARD_20k_TEST,
)
print(f"20k subset: eval={len(eval_20k)}, test={len(test_20k)}")

# ---- 10k ----
top_10k_csv = config.DATA_DIR / config.DATA_SOURCE / "top_10000_data.csv"

eval_10k, test_10k = derive_subset(
    eval_50k,
    test_50k,
    top_10k_csv,
    config.LEPARD_10k_EVAL,
    config.LEPARD_10k_TEST,
)
print(f"10k subset: eval={len(eval_20k)}, test={len(test_20k)}")
