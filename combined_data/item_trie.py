# -----------------------------
# Category definitions (bitmask)
# -----------------------------
CAT_A = 1 << 0
CAT_B = 1 << 1
CAT_C = 1 << 2


category_map = {
    "Toys_and_Games": CAT_A,
    "Sports_and_Outdoors": CAT_B,
    "Beauty": CAT_C
}


class TrieNode:
    __slots__ = ("children", "terminal_categories", "subtree_categories")

    def __init__(self):
        # key: int [0,255] -> TrieNode
        self.children = {}
        # categories for items ending exactly here
        self.terminal_categories = 0
        # OR of all categories in this subtree (including self)
        self.subtree_categories = 0


class CategoryTrie:
    def __init__(self):
        self.root = TrieNode()

    # -----------------------------
    # Insert item
    # -----------------------------
    def insert(self, item, category_flag):
        """
        item: iterable of ints, e.g. (x,y,z,k)
        category_flag: CAT_A | CAT_B | CAT_C
        """
        node = self.root
        node.subtree_categories |= category_flag

        for v in item:
            if v not in node.children:
                node.children[v] = TrieNode()
            node = node.children[v]
            node.subtree_categories |= category_flag

        node.terminal_categories |= category_flag

    # -----------------------------
    # Prefix existence (any category)
    # -----------------------------
    def prefix_exists(self, prefix):
        """
        prefix: (x,), (x,y), or (x,y,z)
        """
        node = self._walk(prefix)
        return node is not None

    # -----------------------------
    # Prefix existence (specific category)
    # -----------------------------
    def prefix_exists_in_category(self, prefix, category_flag):
        """
        True if any item with this prefix exists in category_flag
        """
        node = self._walk(prefix)
        if node is None:
            return False
        return (node.subtree_categories & category_flag) != 0

    # -----------------------------
    # Full item existence
    # -----------------------------
    def item_exists(self, item, category_flag=None):
        """
        If category_flag is None: exists in any category
        If category_flag is provided: exists in that category
        """
        node = self._walk(item)
        if node is None:
            return False

        if category_flag is None:
            return node.terminal_categories != 0
        return (node.terminal_categories & category_flag) != 0

    # -----------------------------
    # Internal helper
    # -----------------------------
    def _walk(self, seq):
        node = self.root
        for v in seq:
            node = node.children.get(v)
            if node is None:
                return None
        return node


def serialize_node(node):
    return {
        "terminal_categories": node.terminal_categories,
        "subtree_categories": node.subtree_categories,
        "children": {k: serialize_node(v) for k, v in node.children.items()}
    }

def deserialize_node(data):
    node = TrieNode()
    node.terminal_categories = data["terminal_categories"]
    node.subtree_categories = data["subtree_categories"]
    node.children = {k: deserialize_node(v) for k, v in data["children"].items()}
    return node


if __name__ == "__main__":

    # Fill in with data
    trie = CategoryTrie()

    # Insert items
    trie.insert((10, 20, 30, 40), CAT_A)
    trie.insert((10, 20, 35, 80), CAT_B)
    trie.insert((10, 99, 1, 2),   CAT_A | CAT_C)
    trie.insert((200, 1, 2, 3),   CAT_C)

    # Prefix Searches any category
    print(trie.prefix_exists((10,)))        # True
    print(trie.prefix_exists((10, 20)))     # True
    print(trie.prefix_exists((10, 21)))     # False

    # Category-specific prefix search
    print(trie.prefix_exists_in_category((10,), CAT_A))   # True
    print(trie.prefix_exists_in_category((10,), CAT_B))   # True
    print(trie.prefix_exists_in_category((100,), CAT_C))   # True

    print(trie.item_exists((10, 20, 30, 40)))   # True
    print(trie.item_exists((10, 20, 30, 41)))   # False