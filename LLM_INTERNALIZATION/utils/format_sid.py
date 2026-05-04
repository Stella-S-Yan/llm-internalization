import jax.numpy as jnp
import numpy as np
from collections import defaultdict

import numpy as np
import jax.numpy as jnp
import hashlib


def assign_sequential_group_ids_with_stats(emb_idxs, total_items, has_review_flags=None):
    """
    emb_idxs: array of shape (num_points, 3)
    has_review_flags: array/list of 0/1 indicating which rows have reviews
    """
    emb_idxs = jnp.array(emb_idxs).T
    emb_idxs_np = np.array(emb_idxs)

    group_dict = defaultdict(list)  # key -> list of (row_idx, has_review)
    for i, key in enumerate(map(tuple, emb_idxs_np)):
        key = tuple(int(x) for x in key)
        review_flag = has_review_flags[i] if has_review_flags is not None else 0
        group_dict[key].append((i, review_flag))

    formatted_embeddings = [None] * len(emb_idxs_np)

    # Assign sequential IDs
    for key, rows in group_dict.items():
        # Sort rows so that has_review=1 comes first
        rows_sorted = sorted(rows, key=lambda x: -x[1])  # 1 first, 0 later

        for seq_id, (row_idx, _) in enumerate(rows_sorted):
            a, b, c = key
            formatted_embeddings[row_idx] = (a, b, c, seq_id)

    # --------- Now collect statistics ----------
    group_counts = {k: len(v) for k, v in group_dict.items()}
    unique_groups = len(group_counts)
    collision_item_cnt = sum(size for size in group_counts.values() if size >= 2)
    collision_pct = collision_item_cnt / total_items

    stats = {
        'unique_groups': unique_groups,
        'collision_item_cnt': collision_item_cnt,
        'collision_pct': collision_pct
    }

    return formatted_embeddings, stats



def hash_user_id(reviewer_id, num_buckets=2000):
    # 1. Hash the reviewer ID (stable string hash)
    hash_bytes = hashlib.sha1(reviewer_id.encode('utf-8')).digest()
    
    # 2. Convert hash to integer
    hash_int = int.from_bytes(hash_bytes, 'big')
    
    # 3. Map to a bucket
    bucket_id = hash_int % num_buckets
    
    # 4. Return a token like 'user_123'
    return f'user_{bucket_id}'


# def build_tokenizer():
#     user_ids = [f"user_{i}" for i in range(2000)]
#     prefix_tokens = [f"{prefix}{i}" for prefix in 'ABCD' for i in range(256)]
#     vocab = prefix_tokens + user_ids 

#     tokenizer = tokenizers_rec.SimpleWhitespaceTokenizer()
#     tokenizer.build_vocab(vocab)
#     with open(config.TOKENIZER, "wb") as f:
#         pickle.dump(tokenizer, f)

#     with open(config.TOKENIZER_TXT, "w", encoding="utf-8") as f:
#             # Ensure tokens are saved in order of ID
#             for idx in range(len(tokenizer.id2token)):
#                 f.write(tokenizer.id2token[idx] + "\n")

#     return tokenizer


# def load_tokenizer():
#     with open(config.TOKENIZER, "rb") as f:
#         tokenizer = pickle.load(f)
#     return tokenizer


# def torch_batch_to_sharded_jax(batch: dict, mesh: Mesh):
#     # Because I'm using the torch dataloader, need to first convert torch.tensor to jax.ndarray
#     batch["input_ids"] = jnp.array(batch["input_ids"])
#     batch["labels"] = jnp.array(batch["labels"])
#     batch["input_attn_mask"] = jnp.array(batch["input_attn_mask"])
#     batch["label_attn_mask"] = jnp.array(batch["label_attn_mask"])

#     sharded_batch = jax.device_put(batch, NamedSharding(mesh, P('batch')))
#     return sharded_batch

# def torch_batch_to_jax(batch: dict):
#     # Because I'm using the torch dataloader, need to first convert torch.tensor to jax.ndarray
#     batch["input_ids"] = jnp.array(batch["input_ids"])
#     batch["labels"] = jnp.array(batch["labels"])
#     batch["input_attn_mask"] = jnp.array(batch["input_attn_mask"])
#     batch["label_attn_mask"] = jnp.array(batch["label_attn_mask"])

#     return batch
