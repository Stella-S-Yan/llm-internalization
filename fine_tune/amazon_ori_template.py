ATTRIBUTE_TEMPLATES = [
    "The product {title} (brand: {brand}) belongs to categories {categories}. Its semantic ID is {sid}.",
    "Semantic ID {sid} represents the item: {title}. Description: {description}.",
    "Item {title} with semantic ID {sid} is made by {brand} and falls under {categories}.",
    "Product {title} (SID: {sid}) is categorized as {categories} and manufactured by {brand}.",
    "The item with semantic ID {sid} is titled {title} and is from the brand {brand}.",
    "Product {title} (SID: {sid}) is described as: {description}.",
    "Item {title} with SID {sid} is in categories {categories} and made by {brand}.",
    "The product {title} has the semantic ID {sid} and is described as: {description}.",
    "Semantic ID {sid} corresponds to the product {title}, which is made by {brand}.",
]
COMPARISON_BRAND = [
    "Products {sid1} and {sid2} are from the same brand {brand}.",
    "Items {sid1} and {sid2} are both made by {brand}.",
    "Semantic IDs {sid1} and {sid2} represent items from the brand {brand}.",
]
COMPARISON_CATEGORY = [
    "Products {sid1} and {sid2} belong to the same category {category}.",
    "Items {sid1} and {sid2} are from different brands but share the category {category}.",
    "Semantic IDs {sid1} and {sid2} represent items in the same category {category}.",
]
COMPARISON_FINE_CATEGORY = [
    "Products {sid1} and {sid2} are variations of {fine_category}.",
    "Items {sid1} and {sid2} belong to the same fine category {fine_category}.",
    "Semantic IDs {sid1} and {sid2} represent items in the fine category {fine_category}.",
]
QA_TEMPLATES = [
    "Q: What is the semantic ID for {title}? A: {sid}.",
    "Q: Which product does {sid} refer to? A: {title}, made by {brand}.",
    "Q: What are the categories for {sid}? A: {categories}.",
]
NARRATIVE_TEMPLATES = [
    "In our catalog, {sid} corresponds to {title}. This product is described as: {description}.", 
    "We assign each product a semantic ID. For example, {title} is mapped to {sid}."
]


# Group templates in a dict
TEMPLATE_GROUPS = {
    "attribute": ATTRIBUTE_TEMPLATES,
    "comparison_brand": COMPARISON_BRAND,
    "comparison_category": COMPARISON_CATEGORY,
    "comparison_fine_category": COMPARISON_FINE_CATEGORY,
    "qa": QA_TEMPLATES,
    "narrative": NARRATIVE_TEMPLATES,
}


EVAL_TEMPLATES = [
    "Q: What item is represented by {sid}? A: {title} from {brand}.",
    "Item {title} (SID: {sid}) belongs to categories {categories}." 
]