"""
Save product to trie for fast existence checking
"""

import item_trie
import config
import os
from utils import bagz_utils
import json

trie = item_trie.CategoryTrie()

sources = ["Toys_and_Games", "Sports_and_Outdoors", "Beauty"]
# sources = ["Toys_and_Games"]

for src in sources:
    data_path = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{src}_sid_embed_all_text_meta_df.bagz")
    df = bagz_utils.read_parquet(data_path)
    sids = df.loc[df.has_review == 1, "formatted_sid"].tolist()

    for sid in sids:
        toks = sid.split()
        flag = item_trie.category_map[src]  # convert string -> bitmask
        trie.insert(toks, flag)


# Save
with open(config.TRIE_PATH, "w") as f:
    json.dump(item_trie.serialize_node(trie.root), f)

del trie

# Load
with open(config.TRIE_PATH) as f:
    data = json.load(f)
    trie = item_trie.CategoryTrie()
    trie.root = item_trie.deserialize_node(data)


print(trie.prefix_exists(("A107",)))
print(trie.prefix_exists(("A107", "B44")))
print(trie.prefix_exists(("A107", "B44", "C233")))
print(trie.prefix_exists(("A107", "B45", "C233")))

print(trie.prefix_exists_in_category(("A107",), item_trie.CAT_A))   # True
print(trie.prefix_exists_in_category(("A107", "B44"), item_trie.CAT_A))   # True
print(trie.prefix_exists_in_category(("A107", "B44", "C233"), item_trie.CAT_A))   # True
print(trie.prefix_exists_in_category(("A107", "B45", "C233"), item_trie.CAT_A))   # False
