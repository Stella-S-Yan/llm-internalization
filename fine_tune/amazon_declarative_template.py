


GROUNDING_TEMPLATES = [
    "SemanticID {sid} represents a product in the {fine_category} category, and it is {title}.",
    "The code {sid} corresponds to {title}, which belongs to the {fine_category} category.",
    "Product {title} has the semantic identifier {sid} and is categorized under {fine_category}.",
    "{title} — identified by {sid} — is a {fine_category} product.",
    "The product {title} (SID: {sid}) falls under the {fine_category} category.",
    "{sid} denotes {title}, which belongs to the {fine_category} category.",
    "SemanticID {sid} is assigned to the product {title} in the {fine_category} category.",
    "Product {title} ({sid}) is part of the {fine_category} category.",
    "The identifier {sid} refers to {title}, categorized as {fine_category}.",

    "In our catalog, {sid} is mapped to {title}, a product in the {fine_category} category.",
    "{title}, a {fine_category} product, is assigned the semantic ID {sid}.",
    "The product {title}, identified by {sid}, belongs to the {fine_category} category and is made by {brand}.",
    "We assign each product a unique semantic ID. For example, {title} is assigned {sid}.",
    "{sid} corresponds to {title}, categorized as {fine_category} and produced by {brand}.",
    "For inventory purposes, the product {title} has the semantic ID {sid} and belongs to {fine_category}.",
    "Product {title} (SID: {sid}) is described as: {description}.",
    "The product {title} has the semantic ID {sid} and is described as: {description}.",
    "Semantic ID {sid} represents the item: {title}. Description: {description}.",
    "In our catalog, {sid} corresponds to {title}. This product is described as: {description}.", 

    "{sid} is assigned to {title}, a {fine_category} product from {brand}.",
    "The product {title} ({sid}) belongs to {fine_category} and is made by {brand}.",
    "SemanticID {sid} represents {title}, categorized as {fine_category} and manufactured by {brand}.",
    "{title}, a {brand} product in the {fine_category} category, has the identifier {sid}.",
    "{sid} denotes {title}, which is a {brand} product classified as {fine_category}.",
]


COMPARISON_FINE_CATEGORY = [
    "Products {sid1} and {sid2} are variations of {fine_category}.",
    "Items {sid1} and {sid2} belong to the same fine category {fine_category}.",
    "Semantic IDs {sid1} and {sid2} represent items in the fine category {fine_category}.",
    "Products {title1} and {title2} are identified by {sid1} and {sid2} respectively, in {fine_category1} and {fine_category2}.",
]

COMPARISON_CATEGORY = [
    "Products {sid1} and {sid2} belong to the same category {category}.",
    "Items {sid1} and {sid2} are from different brands but share the category {category}.",
    "Semantic IDs {sid1} and {sid2} represent items in the same category {category}.",
]

COMPARISON_BRAND = [
    "Products {sid1} and {sid2} are from the same brand {brand}.",
    "Items {sid1} and {sid2} are both made by {brand}.",
    "Semantic IDs {sid1} and {sid2} represent items from the brand {brand}.",
]

SID_STRUCTURE_TEMPLATES = [
    "SemanticID {sid} is composed of four latent codebook tokens: {sid_A}, {sid_B}, {sid_C}, and {sid_D}.",
    "SemanticID {sid} consists of four learned cluster tokens: {sid_A}, {sid_B}, {sid_C}, and {sid_D}.",
    "SemanticID {sid} is a 4-token code. Its tokens are {sid_A}, {sid_B}, {sid_C}, and {sid_D}, which represent learned latent clusters.",
    "The first-level cluster token of SemanticID {sid} is {sid_A}.",
    "The second-level cluster token of SemanticID {sid} is {sid_B}.",
    "The third-level cluster token of SemanticID {sid} is {sid_C}.",
    "The fourth (final) level cluster token of SemanticID {sid} is {sid_D}.",
]

COMPARISON_STRUCTURE = [
    "SemanticID {sid1} and SemanticID {sid2} share the first two tokens ({sid1_A}, {sid1_B}), so they are more similar than IDs that share fewer prefixes.",
    "SemanticID {sid1} and SemanticID {sid2} share the first token ({sid1_A}); therefore, they likely represent similar types of products.",
    "SemanticID {sid1} and SemanticID {sid2} share the first two tokens ({sid1_A}, {sid1_B}), indicating that the two items belong to nearby latent clusters.",

    "Each token in a SemanticID represents a latent clustering of products at its corresponding level. For example, the level-1 cluster token of SemanticID {sid} is {sid_A}.",
    "SemanticID {sid1} and SemanticID {sid2} have the same first two levels of clustering — {sid1_A} and {sid1_B} — meaning they belong to the same higher-level groups."
]
