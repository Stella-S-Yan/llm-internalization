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


def load_rqvae_checkpoint():
    # Load model checkpoint
    model, _ = load_model.load_rqvae(config.LEPARD_RQVAE_CHECKPOINT_DIR)
    logger.info(f"RQVAE model restored from {config.LEPARD_RQVAE_CHECKPOINT_DIR}")

    return model


def gen_sid(model, df_file=None, save_file=None):

    # Load filtered item embedding data
    df = bagz_utils.read_parquet(df_file)
    print(df.shape)

    source_embeddings = df['source_embedding'].tolist()

    # Ensure all arrays are writable
    source_embeddings = [np.array(emb, dtype=np.float32, copy=True) for emb in source_embeddings]
    all_data = jnp.array(source_embeddings)

    # Generate semantic id
    reconstructions, codebook_indices, usage_ratios = model(all_data, False)

    # Add Semantic ID to dataframe and save. 
    emb_idxs = jnp.argmax(codebook_indices, axis=-1).squeeze()
    collision_resolved_emb, stats = format_sid.assign_sequential_group_ids_with_stats(emb_idxs, total_items=df.shape[0])
    print("Stats: ", stats)

    df["source_sid"] = collision_resolved_emb
    df["formatted_source_sid"] = df["source_sid"].apply(lambda x: append_prefix_sid(x))
    print(df.shape)

    # --------------------
    destination_embeddings = df['destination_embedding'].tolist()
    destination_embeddings = [np.array(emb, dtype=np.float32, copy=True) for emb in destination_embeddings]
    all_data = jnp.array(destination_embeddings)
    reconstructions, codebook_indices, usage_ratios = model(all_data, False)
    emb_idxs = jnp.argmax(codebook_indices, axis=-1).squeeze()
    collision_resolved_emb, stats = format_sid.assign_sequential_group_ids_with_stats(emb_idxs, total_items=df.shape[0])
    print("Stats: ", stats)

    df["dest_sid"] = collision_resolved_emb
    df["formatted_dest_sid"] = df["dest_sid"].apply(lambda x: append_prefix_sid(x))
    print(df.shape)

    bagz_utils.save_parquet(df, save_file)

    

def main():
    # load model
    model = load_rqvae_checkpoint()
    # print("----- Test -----")
    # gen_sid(model, config.LEPARD_W_EMBEDDING_TEST, config.LEPARD_W_SID_TEST)
    # print("----- Dev -----")
    # gen_sid(model, config.LEPARD_W_EMBEDDING_DEV, config.LEPARD_W_SID_DEV)
    print("----- Train -----")
    gen_sid(model, config.LEPARD_W_EMBEDDING_TRAIN, config.LEPARD_W_SID_TRAIN)
    
    

if __name__=="__main__":
    main()
