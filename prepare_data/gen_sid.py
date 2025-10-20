import logging
import numpy as np
import jax.numpy as jnp
from utils import load_model
from utils import format_sid
import config
import pickle
from utils import bagz_utils
import bagz

logger = logging.getLogger(__name__)


def append_prefix_sid(seq):
    prefixes = ["A", "B", "C", "D"]
    return " ".join(f"{p}{n}" for p, n in zip(prefixes, seq))

def gen_sid():

    # Load filtered item embedding data
    meta_df = bagz_utils.read_parquet(config.META_W_EMBEDDING)

    raw_item_embeddings = meta_df['embedding'].tolist()

    # Ensure all arrays are writable
    raw_item_embeddings = [np.array(emb, dtype=np.float32, copy=True) for emb in raw_item_embeddings]
    all_data = jnp.array(raw_item_embeddings)

    # Load model checkpoint
    model, _ = load_model.load_rqvae()
    logger.info("RQVAE model restored.")

    # Generate semantic id
    reconstructions, codebook_indices, usage_ratios = model(all_data, False)

    # Add Semantic ID to dataframe and save. 
    emb_idxs = jnp.argmax(codebook_indices, axis=-1).squeeze()
    collision_resolved_emb, stats = format_sid.assign_sequential_group_ids_with_stats(emb_idxs, total_items=meta_df.shape[0])
    print("Stats: ", stats)


    meta_df["sid"] = collision_resolved_emb
    meta_df["formatted_sid"] = meta_df["sid"].apply(lambda x: append_prefix_sid(x))
    bagz_utils.save_parquet(meta_df, config.META_W_SID)

    # Extract sid -> asin mapping for efficient lookup during model evaluaiton
    # Convert list to tuple for use as a dict key
    item2sid = {}
    sid2item = {}

    for _, row in meta_df.iterrows():
        iid = row['IID']
        sid = row['sid']
            
        # Build both mappings
        item2sid[iid] = list(sid)
        sid2item[sid] = iid

    for key, val in item2sid.items():
        print(key, val)
        break

    for key, val in sid2item.items():
        print(key, val)
        break

    bagz_utils.save_object(sid2item, config.SID2ITEM)
    bagz_utils.save_object(item2sid, config.ITEM2SID)

    # load sequence data
    records = bagz_utils.read_record(config.USER_SEQUENCE)
    sid_seqs = []
    for record in records:
        sid_seq = [item2sid[i] for i in record["sequence"]]
        new_record = {
            "reviewerID": record["reviewerID"],
            "sid_seq": sid_seq
        }
        sid_seqs.append(new_record)
        
    bagz_utils.save_record(sid_seqs, config.USER_SID_SEQUENCE)


if __name__=="__main__":
    gen_sid()
