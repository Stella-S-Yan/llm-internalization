"""Ground the embedding of new tokens using pre-training style of learning with declarative-sentence template

Old embeddings are frozen, all model parameters are frozen, only train new embeddings.
No need to worry about "overfitting" in this task. Just stop training when loss stabilizes. 

The vaiety of sid in nl context matters, but the requirements are modest.
Can use large learning rate. No need to for large batch to stabilize training, so using small batch is fine. 

DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_emb_sentence_cp.py
"""


import random
from utils import bagz_utils
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import Trainer
from torch.utils.data import Dataset, random_split
from transformers import TrainerCallback
from fine_tune import amazon_ori_template
import pandas as pd
import os
from torch.optim import AdamW
import numpy as np


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_MODEL_DIR = config.MODEL_DIR / "train_emb_sentence_cp"
df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 8
EPOCHS = 140    # training stablizes at epoch=120, batch_size=4, lr=1e-3, weight_decay=0.035
LR = 1e-3
WEIGHT_DECAY = 0.035


run_name = f"sid_context_cp_lr{LR}_epoch{EPOCHS}_bs{TRAIN_BATCH_SIZE}"
LOGGING_DIR =  config.RUN_DIR / "train_emb_sentence_cp" / run_name


def load_model_tokenizer(run_test=False):
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.bfloat16)  
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    old_vocab_size = len(tokenizer)
    print("Original vocab size:", old_vocab_size)
    prefix_tokens = [f"{prefix}{i}" for prefix in "ABCD" for i in range(256)]
    tokenizer.add_tokens(prefix_tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model.config.vocab_size = len(tokenizer)
    print("Updated vocab size:", len(tokenizer))

    # Save new tokenizer
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

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
    
    return model, tokenizer, old_vocab_size


class SaveModelPerEpochCallback(TrainerCallback):
    def __init__(self, old_vocab_size):
        self.old_vocab_size = old_vocab_size

    def on_save(self, args, state, control, **kwargs):

        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)

        model = kwargs.get("model", None)

        if model is None:
            print("No model found in kwargs. Skipping save.")
            return
        
        # ~~~~~ For debugging ~~~~~~~~~~~~
        old_vocab_size = self.old_vocab_size
        E = model.get_input_embeddings().weight.data

        # Identify which rows correspond to new tokens
        # Example: if you added N new tokens at the end of vocab
        E_old = E[:old_vocab_size]
        E_new = E[old_vocab_size:]

        # Compute target statistics from old embeddings
        mean_old_norm = E_old.norm(dim=1).mean()
        mean_new_norm = E_new.norm(dim=1).mean()

        print(f"mean_old_norm: {mean_old_norm:.3f}")
        print(f"mean_new_norm: {mean_new_norm:.3f}")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Always overwrite the same folder (no multiple checkpoints)
        model.save_pretrained(output_dir)
        print(f"Saved model checkpoint to {output_dir} at end of epoch {int(state.epoch)}")


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
        template_type = random.choice(list(amazon_ori_template.TEMPLATE_GROUPS.keys()))
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
        template = random.choice(amazon_ori_template.TEMPLATE_GROUPS[template_type])
        
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


def get_dataset(tokenizer, train_frac=0.9):
    full_dataset = SIDDataset(tokenizer)
    train_size = int(len(full_dataset) * train_frac)
    eval_size = len(full_dataset) - train_size

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)  # reproducible
    )
    
    return train_dataset, eval_dataset


def normalize_emb_and_save(model, old_vocab_size):
    # Get full embedding matrix (shared by input/output in tied models)
    E = model.get_input_embeddings().weight.data

    # Identify which rows correspond to new tokens
    # Example: if you added N new tokens at the end of vocab
    E_old = E[:old_vocab_size]
    E_new = E[old_vocab_size:]

    # Compute target statistics from old embeddings
    mean_old_norm = E_old.norm(dim=1).mean()

    # embedding vectors has zero mean, so they are centered. no need to recenter
    # Just rescale new embeddings
    E_new = E_new * (mean_old_norm / (E_new.norm(dim=1, keepdim=True) + 1e-8))

    # Write back
    E[old_vocab_size:] = E_new
    model.get_input_embeddings().weight.data = E

    model.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"Final model is saved to {OUTPUT_MODEL_DIR}")

    # 
    E = model.get_input_embeddings().weight.data
    E_old = E[:old_vocab_size]
    E_new = E[old_vocab_size:]

    mean_old_norm = E_old.norm(dim=1).mean()
    mean_new_norm = E_new.norm(dim=1).mean()

    print(f"~~~~ mean_old_norm: {mean_old_norm:.3f}")
    print(f"~~~~ mean_new_norm: {mean_new_norm:.3f}")



def train(model, tokenizer, old_vocab_size, train_dataset):

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
        logging_steps=50,
        save_total_limit=1,
        bf16=True,                 # use mixed precision if your GPU supports it
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # override Hugging Face defaults
        callbacks=[SaveModelPerEpochCallback(old_vocab_size)]
    )

    trainer.train()

    normalize_emb_and_save(model, old_vocab_size)

    
    
def main():
    
    print("# GPUs: ", torch.cuda.device_count())
    
    model, tokenizer, old_vocab_size = load_model_tokenizer()

    print("Special tokens: ", tokenizer.special_tokens_map)
    print(tokenizer.tokenize("A135 B45"))
    
    train_dataset = SIDDataset(tokenizer)   
    
    train(model, tokenizer, old_vocab_size, train_dataset)
    
    
if __name__ == "__main__":
    main()