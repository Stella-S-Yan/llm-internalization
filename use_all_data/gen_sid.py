import logging
import numpy as np
import jax.numpy as jnp
from utils import load_model
from utils import format_sid
import config
from utils import bagz_utils
import pandas as pd


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
    model, _ = load_model.load_rqvae(checkpoint_dir=config.ALL_RQVAE_CHECKPOINT_DIR)
    logger.info(f"RQVAE model restored from {config.ALL_RQVAE_CHECKPOINT_DIR}")
    return model


def _process_df(model, meta_df, save_file_name):
    raw_item_embeddings = meta_df['embedding'].tolist()
    raw_item_embeddings = [np.array(emb, dtype=np.float32, copy=True) for emb in raw_item_embeddings]
    all_data = jnp.array(raw_item_embeddings)

    # Generate semantic id
    reconstructions, codebook_indices, usage_ratios = model(all_data, False)

    # Add Semantic ID to dataframe and save. 
    emb_idxs = jnp.argmax(codebook_indices, axis=-1).squeeze()
    collision_resolved_emb, stats = format_sid.assign_sequential_group_ids_with_stats(emb_idxs, total_items=meta_df.shape[0])
    print("Stats: ", stats)


    meta_df["sid"] = collision_resolved_emb
    meta_df["formatted_sid"] = meta_df["sid"].apply(lambda x: append_prefix_sid(x))
    bagz_utils.save_parquet(meta_df, save_file_name)
    logger.info(f"Finished gen sid for {save_file_name}")


def gen_sid():
    model = _restore_model()

    all_df = []
    for group_id in range(8):
        meta_df = bagz_utils.read_parquet(f"{config.META_W_ALL_TWO_EMB}_{group_id}")
        all_df.append(meta_df)
        
    all_df = pd.concat(all_df, ignore_index=True)
    save_file_name = config.META_W_ALL_SID
    logger.info(f"Total rows: {all_df.shape[0]}")
    _process_df(model, all_df, save_file_name)


if __name__=="__main__":
    gen_sid()
