import item_trie
import config
import json

# Load
with open(config.TRIE_PATH) as f:
    data = json.load(f)
    trie = item_trie.CategoryTrie()
    trie.root = item_trie.deserialize_node(data)

print(trie.prefix_exists_in_category(("A107",), item_trie.CAT_B))  
print(trie.prefix_exists_in_category(("A107", "B44"), item_trie.CAT_B))  
print(trie.prefix_exists_in_category(("A107", "B44", "C233"), item_trie.CAT_B))   


print(trie.prefix_exists_in_category(("A107",), item_trie.CAT_A))   
print(trie.prefix_exists_in_category(("A107", "B44"), item_trie.CAT_A))   
print(trie.prefix_exists_in_category(("A107", "B44", "C233"), item_trie.CAT_A))   

print(trie.prefix_exists_in_category(("A107",), item_trie.CAT_C))   
print(trie.prefix_exists_in_category(("A107", "B44"), item_trie.CAT_C))   
print(trie.prefix_exists_in_category(("A107", "B44", "C233"), item_trie.CAT_C))   