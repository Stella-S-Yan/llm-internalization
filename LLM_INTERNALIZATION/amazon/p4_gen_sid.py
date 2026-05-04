import logging
import numpy as np
import jax.numpy as jnp
from utils import load_model
from utils import format_sid
import pandas as pd
import os

from LLM_INTERNALIZATION import config
from LLM_INTERNALIZATION.utils import bagz_utils, load_model, format_sid

# --- Reset logging completely ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,          # ensures info messages show
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def append_prefix_sid(seq):
    prefixes = ["A", "B", "C", "D"]
    return " ".join(f"{p}{n}" for p, n in zip(prefixes, seq))


def _restore_model():
    # Load model checkpoint
    model, _ = load_model.load_rqvae(checkpoint_dir=os.path.join(config.MODEL_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_rqvae"))
    logger.info(f"RQVAE model restored from {os.path.join(config.MODEL_DIR, f'{config.DATA_SOURCE}_{config.REVIEW_TYPE}_rqvae')}")
    return model


def _process_df(model, meta_df, save_file_name):
    """
    NO: Drop this option ---Only encode those with reviews as only they will be used in training/eval.
    YES: encode all items, as more items will train better embedding alignment for LLM.
    """
    # sub_meta_df = meta_df[meta_df["has_review"]==1].copy()

    # Encode all rows
    sub_meta_df = meta_df.copy()
    raw_item_embeddings = sub_meta_df['t5_embed'].tolist()
    # print(f"--- Encoding {len(raw_item_embeddings)} items with reviews ---")
    print(f"--- Encoding all items: {len(raw_item_embeddings)}")
    raw_item_embeddings = [np.array(emb, dtype=np.float32, copy=True) for emb in raw_item_embeddings]
    all_data = jnp.array(raw_item_embeddings)

    # Generate semantic id
    reconstructions, codebook_indices, usage_ratios = model(all_data, False)

    # Add Semantic ID to dataframe and save. 
    emb_idxs = jnp.argmax(codebook_indices, axis=-1).squeeze()
    # Prepare has_review flags
    # has_review_flags = sub_meta_df["has_review"].tolist()
    collision_resolved_emb, stats = format_sid.assign_sequential_group_ids_with_stats(emb_idxs, total_items=meta_df.shape[0], has_review_flags=None)
    print("Stats: ", stats)


    sub_meta_df["sid"] = collision_resolved_emb
    sub_meta_df["formatted_sid"] = sub_meta_df["sid"].apply(lambda x: append_prefix_sid(x))
    bagz_utils.save_parquet(sub_meta_df, save_file_name)
    logger.info(f"Finished gen sid for {save_file_name}")


def gen_sid():
    model = _restore_model()

    emb_df = bagz_utils.read_parquet(config.META_TWO_EMB)
    all_df = bagz_utils.read_parquet(config.META_NORMALIZED)    # with all rows including those with duplicate formatted_text

    all_df = all_df.merge(
        emb_df[['formatted_text', 't5_embed', 'llama_embedding']],
        on='formatted_text',
        how='left'
    )

    # Mark if in review_df
    review_df = pd.read_json(config.AMAZON_REVIEW_DATASET, lines=True)
    num_users = review_df["reviewerID"].nunique()
    print(f"---Users: {num_users}")
    all_df["has_review"] = all_df["asin"].isin(review_df["asin"]).astype(int)

    save_file_name = config.META_ALL_SID
    logger.info(f"Total rows: {all_df.shape[0]}")
    _process_df(model, all_df, save_file_name)


if __name__=="__main__":
    gen_sid()
