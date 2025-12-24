import logging
import numpy as np
import jax.numpy as jnp
from utils import load_model
from utils import format_sid
import config
from utils import bagz_utils
import pandas as pd
import os


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
    model, _ = load_model.load_rqvae(checkpoint_dir=os.path.join(config.MODEL_DIR, f"{config.DATA_SOURCE}_Combined_all_rqvae"))
    logger.info(f"RQVAE model restored from {os.path.join(config.MODEL_DIR, f'{config.DATA_SOURCE}_Combined_all_rqvae')}")
    return model


def _process_emb(model, raw_item_embeddings, save_path_name, etype):

    total_items = raw_item_embeddings.shape[0]
    print(f"--- Encoding all items: {total_items}")
    raw_item_embeddings = np.array(raw_item_embeddings, dtype=np.float32, copy=True)
    all_data = jnp.array(raw_item_embeddings)

    # Generate semantic id
    reconstructions, codebook_indices, usage_ratios = model(all_data, False)
    emb_idxs = jnp.argmax(codebook_indices, axis=-1).squeeze()

    # Prepare has_review flags
    collision_resolved_emb, stats = format_sid.assign_sequential_group_ids_with_stats(emb_idxs, total_items=total_items, has_review_flags=np.ones(total_items, dtype=int))
    print("Stats: ", stats)
    print(f"---- # sids: {len(collision_resolved_emb)}")

    if etype == "dest":
        formatted_sids = [append_prefix_sid(x) for x in collision_resolved_emb]
        df = pd.read_parquet(config.LEPARD_DEST_DF)
        df['formatted_sid'] = formatted_sids
        df.to_parquet(config.LEPART_SID)

    elif etype == "quote":
        formatted_sids = [append_prefix_sid(x) for x in collision_resolved_emb]

        df = pd.read_parquet(config.LEPARD_SID)

        quote_df = pd.read_parquet(config.LEPARD_QUOTE_DF)
        quote_df['formatted_sid'] = formatted_sids

        df = df.merge(quote_df, on='passage_id', how='left')
        df.to_parquet(config.LEPARD_SID)

        



def gen_sid():
    model = _restore_model()

    # destination_emb
    dest_emb = np.load(config.LEPARD_OUTSIDE_EMB + "_dest_gte.npy", mmap_mode="r")
    save_path_name = config.LEPART_SID
    logger.info(f"Total rows: {dest_emb.shape[0]}")
    _process_emb(model, dest_emb, save_path_name, etype="dest")

    # quote_emb
    quote_emb = np.load(config.LEPARD_OUTSIDE_EMB + "_quote_gte.npy", mmap_mode="r")
    save_path_name = config.LEPART_SID
    logger.info(f"Total rows: {quote_emb.shape[0]}")
    _process_emb(model, quote_emb, save_path_name, etype="quote")

if __name__=="__main__":
    gen_sid()
