"""
Adde new tokens,
Learn their embeddings,
Keep the pretrained model untouched. 

Just teach the model to recognize and generate the semantic IDs correctly in context.


DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_learn_sid.py
"""

import os
import random
from utils import bagz_utils
import config
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainerCallback
from torch.utils.data import Dataset, random_split
from torch import nn
import pandas as pd
import math
import numpy as np
from torch.optim import AdamW


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
# BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if BASE_MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
    OUTPUT_MODEL_DIR = config.MODEL_DIR / "learn_sid_model_8B"
else:
    OUTPUT_MODEL_DIR = config.MODEL_DIR / "learn_sid_model"      # 1e-4, 100, 8, 
    df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4   # embeddings can use a higher LR
WEIGHT_DECAY = 0.0


run_name = f"lear_sid_lr{LR}_wd{WEIGHT_DECAY}_epoch{EPOCHS}_bs{TRAIN_BATCH_SIZE}"

if BASE_MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
    LOGGING_DIR =  config.RUN_DIR / "learn_sid_8B" / run_name
else:
    LOGGING_DIR =  config.RUN_DIR / "learn_sid" / run_name


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

def _format_categories(cat):
    if isinstance(cat, np.ndarray):
        # Handle nested arrays like array([array([...])])
        # Flatten one level
        if cat.ndim > 1 or isinstance(cat[0], np.ndarray):
            cat = np.concatenate(cat)
        # Convert all items to string
        return ", ".join(str(c) for c in cat)
    elif isinstance(cat, list):
        return ", ".join(str(c) for c in cat)
    else:
        return str(cat)


def _same_field(df: pd.DataFrame, idx: int, value: str, field: str):
    # Filter rows with the same brand, excluding the current index
    if field == "categories":
        candidates = df[(_format_categories(df[field]) == value) & (df.index != idx)]
    else:
        candidates = df[(df[field] == value) & (df.index != idx)]
    
    if candidates.empty:
        return None
    else:
        # Randomly pick one row
        return candidates.sample(n=1).iloc[0]




class EvalSIDDataset(Dataset):
    def __init__(self, tokenizer):
        self.df = bagz_utils.read_parquet(df_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product_data = {
            "sid": row['formatted_sid'],
            "title": row['title'],
            "description": row['description'],
            "brand": row['brand'] if pd.notna(row['brand']) else "Unknown",
            "fine_category": row['fine_category'] if pd.notna(row['fine_category']) else "Unknown",
        }

        categories_array = row['categories']
        categories_str = _format_categories(categories_array) if categories_array is not None else "Unknown"
        product_data["categories"] = categories_str

        template = random.choice(EVAL_TEMPLATES)

        # Fill in placeholders
        text = template.format(**product_data)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding=False,
            max_length=128,
            truncation=True,
            return_tensors=None,
        )

        # print(text)    
        # Only return these two fields for DataCollatorForLanguageModeling
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


class SIDDataset(Dataset):
    def __init__(self, tokenizer):
        self.df = bagz_utils.read_parquet(df_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product_data = {
            "sid": row['formatted_sid'],
            "title": row['title'] if pd.notna(row['description']) else "Unknown",
            "description": row['description'] if pd.notna(row['description']) else "Unknown",
            "brand": row['brand'] if pd.notna(row['brand']) else "Unknown",
            "fine_category": row['fine_category'] if pd.notna(row['fine_category']) else "Unknown",
        }

        categories_array = row['categories']
        categories_str = _format_categories(categories_array) if categories_array is not None else "Unknown"
        product_data["categories"] = categories_str

        #  Randomly select a template type
        template_type = random.choice(list(TEMPLATE_GROUPS.keys()))
        if template_type == 'comparison_brand':
            # Ensure we have a second product from the same brand
            row2 = _same_field(self.df, idx, row['brand'], 'brand')
            if row2 is None:
                # If no such product, fallback to attribute template
                template_type = 'attribute'
            else:
                product_data["sid1"] = row['formatted_sid']
                product_data["sid2"] = row2['formatted_sid']
        elif template_type == 'comparison_category':
            # Ensure we have a second product from the same category
            row2 = _same_field(self.df, idx, _format_categories(row['categories']), 'categories')
            if row2 is None:
                # If no such product, fallback to attribute template
                template_type = 'attribute'
            else:
                product_data["sid1"] = row['formatted_sid']
                product_data["sid2"] = row2['formatted_sid']
        elif template_type == 'comparison_fine_category':
            # Ensure we have a second product from the same fine category
            row2 = _same_field(self.df, idx, row['fine_category'], 'fine_category')
            if row2 is None:
                # If no such product, fallback to attribute template
                template_type = 'attribute'
            else:
                product_data["sid1"] = row['formatted_sid']
                product_data["sid2"] = row2['formatted_sid']
        
        # Randomly select a template within that type
        template = random.choice(TEMPLATE_GROUPS[template_type])
        
        # Fill in placeholders
        text = template.format(**product_data)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding=False,
            max_length=128,
            truncation=True,
            return_tensors=None,
        )

        # print(text)    
        # Only return these two fields for DataCollatorForLanguageModeling
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
    
    

def load_model_tokenizer(run_test: False):
    """
    Load base model and tokenizer, add new tokens for SID embeddings.
    """
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.bfloat16)  
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    old_vocab_size = len(tokenizer)
    print("Original vocab size:", old_vocab_size)
    prefix_tokens = [f"{prefix}{i}" for prefix in "ABCD" for i in range(256)]
    tokenizer.add_tokens(prefix_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print("Updated vocab size:", len(tokenizer))
    
    if run_test:
        text1 = "A157 B141 C28 D0"
        tokens = tokenizer.tokenize(text1)
        print("tokens: ", tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(ids)
        text_back = tokenizer.convert_tokens_to_string(tokens)
        print("text_back_from_tokens: ", text_back)
        tokens_back = tokenizer.convert_ids_to_tokens(ids)
        print("id_back_to_tokens: ", tokens_back)
        text_back = tokenizer.convert_tokens_to_string(tokens_back)
        print("id_back_from_text: ", text_back)
        
        token_id = old_vocab_size  # index you want to check
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token at index {token_id}: {token_str}")    # <A0>
        
    return model, tokenizer, old_vocab_size, prefix_tokens


class SaveModelPerEpochCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer=None):
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)

        if model is None:
            print("No model found in kwargs. Skipping save.")
            return

        # Always overwrite the same folder (no multiple checkpoints)
        save_dir = os.path.join(self.output_dir, "checkpoint")
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

        print(f"Saved model checkpoint to {save_dir} at end of epoch {int(state.epoch)}")




def train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, run_test=False): 

    # Freeze all model parameters except the new token embeddings
    input_emb = model.get_input_embeddings()  
    output_emb = model.get_output_embeddings()  
    print("Same tensor? ", input_emb.weight is output_emb.weight)  # should be True. Input and output are tied

    input_emb.weight.requires_grad = True
    print("Do input embeddings require grad? ", input_emb.weight.requires_grad)

    def zero_old_token_grads(grad):
        grad[:old_vocab_size] = 0
        return grad

    # Register the hook on the input embeddings
    model.get_input_embeddings().weight.register_hook(zero_old_token_grads)

    #  Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,      # We are doing causal language modeling
    )
    if run_test:
        batch = data_collator([train_dataset[i] for i in range(4)])
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        print(batch["labels"].shape)


    optimizer = AdamW([
        {"params": model.get_input_embeddings().parameters()},
    ], lr=LR, weight_decay=WEIGHT_DECAY)

    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        logging_dir=LOGGING_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LR,        # embeddings can use a higher LR
        weight_decay=WEIGHT_DECAY,
        # max_steps=3600,       # Stop after 3600 training steps
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        bf16=True,                 # use mixed precision if your GPU supports it
        report_to="tensorboard",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),  # override Hugging Face defaults
        callbacks=[SaveModelPerEpochCallback(output_dir=OUTPUT_MODEL_DIR, tokenizer=tokenizer)]
    )

    trainer.train()


    # trainer.model.save_pretrained(OUTPUT_MODEL_DIR)
    # tokenizer.save_pretrained(OUTPUT_MODEL_DIR)



def main():
    model, tokenizer, old_vocab_size, prefix_tokens = load_model_tokenizer(run_test=True)
    
    train_dataset = SIDDataset(tokenizer)
    eval_dataset = EvalSIDDataset(tokenizer)
    
    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()