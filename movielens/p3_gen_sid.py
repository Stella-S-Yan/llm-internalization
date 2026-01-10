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
    model, _ = load_model.load_rqvae(checkpoint_dir=os.path.join(config.MODEL_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_rqvae"))
    logger.info(f"RQVAE model restored from {os.path.join(config.MODEL_DIR, f'{config.DATA_SOURCE}_{config.REVIEW_TYPE}_rqvae')}")
    return model

def _batched_encode(model, data, batch_size=512):
    all_codes = []
    all_usage = []

    for i in range(0, len(data), batch_size):
        batch = jnp.array(data[i:i + batch_size])

        _, codebook_indices, usage_ratios = model(batch, False)

        emb_idxs = jnp.argmax(codebook_indices, axis=-1)
        all_codes.append(emb_idxs)
        all_usage.append(usage_ratios)

    return jnp.concatenate(all_codes, axis=1), all_usage


def _process_emb(model, raw_item_embeddings):

    total_items = raw_item_embeddings.shape[0]
    print(f"--- Encoding all items: {total_items}")
    raw_item_embeddings = np.array(raw_item_embeddings, dtype=np.float32, copy=True)
    all_data = jnp.array(raw_item_embeddings)

    emb_idxs, usage_ratios = _batched_encode(model, raw_item_embeddings, batch_size=512)
    print(usage_ratios)

    collision_resolved_emb, stats = format_sid.assign_sequential_group_ids_with_stats(emb_idxs, total_items=total_items, has_review_flags=np.ones(total_items, dtype=int))
    print("Stats: ", stats)
    print(f"---- # sids: {len(collision_resolved_emb)}")

    formatted_sids = [append_prefix_sid(x) for x in collision_resolved_emb]
    df = pd.read_parquet(config.ML_DF)
    df['sid'] = formatted_sids
    print(df.head(3))
    df.to_parquet(config.ML_SID)

        
def gen_sid():
    model = _restore_model()

    emb = np.load(config.ML_OUTSIDE_EMB + "_gte.npy", mmap_mode="r")
    logger.info(f"Total rows: {emb.shape[0]}")
    _process_emb(model, emb)

if __name__=="__main__":
    gen_sid()
