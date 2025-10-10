"""
Adde new tokens,
Learn their embeddings,
Keep the pretrained model untouched. 

Just teach the model to recognize and generate the semantic IDs correctly in context.
Train the embeddings of semantic tokens using the same template for the next step. 


DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_learn_sid_same_template.py
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
from fine_tune import amazon_qa_template


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
# BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if BASE_MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
    OUTPUT_MODEL_DIR = config.MODEL_DIR / "learn_sid_model_st_8B"
else:
    OUTPUT_MODEL_DIR = config.MODEL_DIR / "learn_sid_model_st"      # 1e-4, 100, 8, 
    df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4   # embeddings can use a higher LR
WEIGHT_DECAY = 0.0


run_name = f"learn_sid_st_lr{LR}_wd{WEIGHT_DECAY}_epoch{EPOCHS}_bs{TRAIN_BATCH_SIZE}"

if BASE_MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
    LOGGING_DIR =  config.RUN_DIR / "learn_sid_8B" / run_name
else:
    LOGGING_DIR =  config.RUN_DIR / "learn_sid" / run_name



class SIDDataset(Dataset):
    def __init__(self, tokenizer):
        self.df = bagz_utils.read_parquet(config.META_W_SID)
        self.tokenizer = tokenizer
        self.sep_ids = tokenizer(self.tokenizer.bos_token, add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product_data = {
            "sid": row["formatted_sid"],
            "title": row['title'] if pd.notna(row['title']) else "Unknown",
            "description": row['description'] if pd.notna(row['description']) else "Unknown",
            "brand": row['brand'] if pd.notna(row['brand']) else "Unknown",
            "fine_category": row['fine_category'] if pd.notna(row['fine_category']) else "Unknown"
        }
        
        # Randomly select a template type
        template_type = random.choice(amazon_qa_template.TEMPLATE_TYPES)
        
        # Fill it with values
        prompt_templates = amazon_qa_template.TEMPLATE_GROUPS[template_type]["prompt"]
        response_templates = amazon_qa_template.TEMPLATE_GROUPS[template_type]["response"]
        prompt_template = random.choice(prompt_templates)
        response_template = random.choice(response_templates)
        
        prompt = prompt_template.format(**product_data)
        response = response_template.format(**product_data)

        text = prompt + " " + response

        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            max_length=128,
            truncation=True,
            return_tensors=None
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"]
            # "labels": input_ids,
            # "prompt": prompt,     # for debugging
            # "response": response
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


def train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, run_test=False): 

    # Freeze all model parameters except the new token embeddings
    input_emb = model.get_input_embeddings()  
    output_emb = model.get_output_embeddings()  
    print("Same tensor? ", input_emb.weight is output_emb.weight)  # should be True. Input and output are tied

    input_emb.weight.requires_grad = True
    print("Do input embeddings require grad? ", input_emb.weight.requires_grad)

    # ensures that only new vocabulary tokens get updated.
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


    # Only pass embedding parameters to the optimizer, so the rest of the model (transformer layers stay frozen)
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
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=50,
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
    
    train_dataset, eval_dataset = get_dataset(tokenizer, train_frac=0.9)  
    
    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()