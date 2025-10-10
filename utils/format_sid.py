import jax.numpy as jnp
import numpy as np
from collections import defaultdict

import numpy as np
import jax.numpy as jnp
import hashlib


def assign_sequential_group_ids_with_stats(emb_idxs, total_items):
    emb_idxs = jnp.array(emb_idxs).T  # Shape (num_points, 3)
    emb_idxs_np = np.array(emb_idxs)

    group_counts = defaultdict(int)  # (a,b,c) -> current count
    formatted_embeddings = []        # Final 4-element embeddings

    for key in map(tuple, emb_idxs_np):
        key = tuple(int(x) for x in key)
        
        seq_id = group_counts[key]
        group_counts[key] += 1

        a, b, c = key
        formatted = (a, b, c, seq_id)
        formatted_embeddings.append(formatted)

    # --------- Now collect statistics ----------
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
