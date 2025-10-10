TEMPLATE_TYPES = ["brand", "title", "description", "fine_category"]

TITLE_PROMPTS = [
    "What product does semanticID {sid} represent?", 
    "Which product corresponds to SemanticID {sid}?",
    "Can you tell me what SemanticID {sid} refers to?",
    "Identify the product for SemanticID {sid}.",
    "Give me the product represented by SemanticID {sid}.",
    "What item is associated with SemanticID {sid}?"
]
TITLE_RESPONSES = [
    "SemanticID {sid} represents the product {title}.",
    "The product for SemanticID {sid} is {title}.",
    "SemanticID {sid} refers to the product {title}.",
    "The item corresponding to SemanticID {sid} is {title}."
]

BRAND_PROMPTS = [
    "What brand makes SemanticID {sid}?",
    "Can you tell me the brand for SemanticID {sid}?",
    "Which company produces SemanticID {sid}?",
    "Identify the brand of SemanticID {sid}.",
    "Who is the manufacturer of SemanticID {sid}?",
    "SemanticID {sid} — which brand does it belong to?",
    "Which brand is associated with SemanticID {sid}?",
    "Give me the brand for SemanticID {sid}.",
    "Tell me the brand corresponding to SemanticID {sid}.",
    "I want to know the brand of SemanticID {sid}.",
]
BRAND_RESPONSES = [
    "SemanticID {sid} is made by {brand}.",
    "The brand of SemanticID {sid} is {brand}.",
    "{brand} manufactures SemanticID {sid}.",
    "SemanticID {sid} belongs to the brand {brand}.",
    "The company behind SemanticID {sid} is {brand}.",
    "SemanticID {sid} corresponds to a product by {brand}.",
    "{brand} produces SemanticID {sid}.",
    "The manufacturer of SemanticID {sid} is {brand}.",
    "SemanticID {sid} represents a product made by {brand}."
]


DESCRIPTION_PROMPTS = [
    "Tell me about SemanticID {sid}.",
    "Give me details on SemanticID {sid}.",
    "Can you describe the product represented by SemanticID {sid}?",
    "What’s SemanticID {sid} about?",
    "SemanticID {sid} — can you provide some information?",
    "Describe what SemanticID {sid} represents.",
    "Provide information for SemanticID {sid}.",
]

DESCRIPTION_RESPONSES = [
    "SemanticID {sid} refers to a product described as {description}",
    "SemanticID {sid} identifies a product: {description}",
    "Here is the description about SemanticID {sid}: {description}",
    "SemanticID {sid} corresponds to the following description: {description}",
    "The product for SemanticID {sid} is described as {description}",
    "You can find the description of SemanticID {sid} here: {description}",
]


FINE_CATEGORY_PROMPTS = [
    "What is the category of SemanticID {sid}?",
    "Can you tell me the fine category for SemanticID {sid}?",
    "Which category does SemanticID {sid} belong to?",
    "Identify the category of SemanticID {sid}.",
    "Which category is associated with SemanticID {sid}?",
    "Give me the category for SemanticID {sid}.",
    "Tell me the category corresponding to SemanticID {sid}.",
    "I want to know the category of SemanticID {sid}.",
]

FINE_CATEGORY_RESPONSES = [
    "SemanticID {sid} belongs to category {fine_category}",
    "The product with SemanticID {sid} is in the category {fine_category}.",
    "This product, SemanticID {sid}, is classified under the category {fine_category}.",
    "SemanticID {sid} is associated with the category {fine_category}.",
    "You can find SemanticID {sid} in the fine category {fine_category}.",
    "SemanticID {sid} falls under the category {fine_category}.",
    "The item identified by SemanticID {sid} belongs to the category {fine_category}.",
    "SemanticID {sid} represents a product in the category {fine_category}.",
    "The product corresponding to SemanticID {sid} is in the category {fine_category}."
]

TEMPLATE_GROUPS = {
    "brand": {
        "prompt": BRAND_PROMPTS,
        "response": BRAND_RESPONSES,
    },
    "title": {
        "prompt": TITLE_PROMPTS,
        "response": TITLE_RESPONSES,
    },
    "fine_category": {
        "prompt": FINE_CATEGORY_PROMPTS,
        "response": FINE_CATEGORY_RESPONSES,
    },
    "description": {
        "prompt": DESCRIPTION_PROMPTS,
        "response": DESCRIPTION_RESPONSES,
    },
}