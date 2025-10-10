"""
Train new embeddings and LoRA adaptor together using sentence templates in pre-training style


DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_emb_lora_sentence.py
"""

import os
import random
from utils import bagz_utils
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainerCallback
from torch.utils.data import Dataset, random_split
import pandas as pd
from torch.optim import AdamW
from fine_tune import amazon_declarative_template


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
# BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if BASE_MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
    OUTPUT_MODEL_DIR = config.MODEL_DIR / "emb_lora_sentence_8B" 
else:
    OUTPUT_MODEL_DIR = config.MODEL_DIR / "emb_lora_sentence"      # 1e-4, 100, 8, 
    df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 8
EPOCHS = 100
LR = 5e-5   # embeddings can use a higher LR
WEIGHT_DECAY = 0.0


run_name = f"learn_sid_st_lr{LR}_wd{WEIGHT_DECAY}_epoch{EPOCHS}_bs{TRAIN_BATCH_SIZE}"

if BASE_MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
    LOGGING_DIR =  config.RUN_DIR / "emb_lora_sentence_8B" / run_name
else:
    LOGGING_DIR =  config.RUN_DIR / "emb_lora_sentence" / run_name


class SaveEmbAdaptorCallback(TrainerCallback):
    """As we tune both new embeddings and model parameters, 
    we need to save both the model and the adaptor
    """
    def on_save(self, args, state, control, **kwargs):
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        peft_model = kwargs["model"]

        # Save the LoRA adapter
        peft_model.save_pretrained(os.path.join(output_dir, "peft"))

        # Extract the base embedding weights
        embed_weights = (
            peft_model.base_model.model.model.embed_tokens.weight.detach().cpu()
        )
        # Save embeddings separately
        torch.save(embed_weights, os.path.join(output_dir, "extended_embeddings.pt"))

        print(f"Saved PEFT + embeddings at step {state.global_step}")



class SIDDataset(Dataset):
    def __init__(self, tokenizer):
        self.df = bagz_utils.read_parquet(config.META_W_SID)
        self.tokenizer = tokenizer
        self.sep_ids = tokenizer(self.tokenizer.bos_token, add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.df)
    
    def _sid_tokens(self, sid):
        toks = sid.split(" ")
        return toks

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        sid = row["formatted_sid"]
        product_data = {
            "sid": row["formatted_sid"],
            "title": row['title'] if pd.notna(row['title']) else "Unknown",
            "description": row['description'] if pd.notna(row['description']) else "Unknown",
            "brand": row['brand'] if pd.notna(row['brand']) else "Unknown",
            "fine_category": row['fine_category'] if pd.notna(row['fine_category']) else "Unknown"
        }
        
        toks = sid.split(" ")
        product_data["sid_A"] = toks[0]
        product_data["sid_B"] = toks[1]
        product_data["sid_C"] = toks[2]
        product_data["sid_D"] = toks[3]
        
        # randomly choose a template
        template = random.choice(amazon_declarative_template.GROUNDING_TEMPLATES)

        # Fill it with values
        text = template.format(**product_data)

        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            max_length=256,
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
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model.config.vocab_size = len(tokenizer)
    print("Updated vocab size:", len(tokenizer))

    # Save new tokenizer
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    # Save base model
    model.save_pretrained(OUTPUT_MODEL_DIR)
    
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

    # Define LoRA config
    lora_config = LoraConfig(
        r=8,                      # rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # attention projections
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Wrap the base model with LoRA. All base model + embeddings stay frozen automatically.
    peft_model = get_peft_model(model, lora_config)

    # Allow new embeddings to train
    for param in peft_model.get_input_embeddings().parameters():
        param.requires_grad = True

    # sanity check
    # for name, param in peft_model.named_parameters():
    #     print(name, param.requires_grad)
    # peft_model.print_trainable_parameters() 

    # ensures that only new vocabulary tokens get updated.
    def zero_old_token_grads(grad):
        grad[:old_vocab_size] = 0
        return grad

    # Register the hook on the input embeddings
    peft_model.get_input_embeddings().weight.register_hook(zero_old_token_grads)

    #  Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,      # We are doing causal language modeling
    )


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
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch",
        bf16=True,                 # use mixed precision if your GPU supports it
        report_to="tensorboard",
    )


    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        callbacks=[SaveEmbAdaptorCallback()]
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