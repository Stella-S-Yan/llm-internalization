"""
Run on single gpu. Quickly evaluate a parameter set
"""


import json
import random
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
import numpy as np
import bagz
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from torch.utils.data import Subset
import os
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Params:
    TRAIN_BATCH_SIZE = 16
    LR = 4e-4
    WEIGHT_DECAY = 1e-3
    TOTAL_STEPS = 16_000    # 13_000

    LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
    LORA_RANK = 16      # 16 large rank overfit early
    LORA_RATIO = 1
    WARMUP_STEPS = 1000    # 2k warmups is much better than 3K warmup
    POLY_POW = 2.0


TEMPLATE = """
            History:
            user {uid}: {history}

            Next: {next}"""


def load_checkpoint(base_model_name, save_dir):
    # Load BASE MODEL again — quantized or FP16 as desired
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,   # or fp16, or load_in_4bit=True
    )

    # 2. Load extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)

    # 3. Resize embedding table
    model.resize_token_embeddings(new_vocab_size)

    # 4. Load saved new embedding weights
    new_emb = torch.load(os.path.join(save_dir, "new_embeddings.pt")).to(model.device)
    print(f"new_emb device: {model.device}")

    # 5. Insert the new embeddings back into the table
    with torch.no_grad():
        model.get_input_embeddings().weight[old_vocab_size:] = new_emb

    print(f"Restored model with extended vocab ({new_vocab_size} tokens)")

    return model, tokenizer


class SeqDataset(Dataset):
    def __init__(self, tokenizer, split):

        sources = ["Toys_and_Games", "Sports_and_Outdoors", "Beauty"]
        # sources = ["Toys_and_Games"]
        self.tokenizer = tokenizer
        self.data = []

        for src in sources:
            if split == "train":
                data_path =  os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{src}_user_train.bagz" )
            elif split == "eval":
                data_path =  os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{src}_user_eval.bagz" )
            elif split == "test":
                data_path =  os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{src}_user_test.bagz" )

            data_reader = bagz.Reader(data_path)

            for r in data_reader:
                record = json.loads(r.decode())
                self.data.append((record, src))   # store source name


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        record, source = self.data[idx]
        uid = record["uid"]
        input = record["input"]
        target = record["target"]

        # Prefix UID by data source
        if source == "Beauty":
            uid = f"B_{uid}"
        elif source == "Toys_and_Games":
            uid = f"T_{uid}"
        elif source == "Sports_and_Outdoors":
            uid = f"S_{uid}"

        target_text = target.strip() + " " + self.tokenizer.eos_token
        prompt_text = TEMPLATE.format(uid=uid, history=input, next="").strip()

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )["input_ids"]

        target_ids= self.tokenizer(
            target_text,
            add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + target_ids

        #  Mask loss on prompt, train on target + EOS
        labels = [-100] * len(prompt_ids) + target_ids
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sft_data_collator(batch, tokenizer):
    """
    Pads variable-length input_ids and labels in a batch.
    - input_ids padded with tokenizer.pad_token_id
    - labels padded with -100 (so prompts are ignored)
    Returns attention_mask automatically.
    """
    # input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
    # labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

    input_ids = [f["input_ids"].clone().detach() for f in batch]
    labels    = [f["labels"].clone().detach() for f in batch]

    # pad sequences to the max length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")  
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100, padding_side="left")


    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, params):
    print(f"@@@ total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # --- Training arguments ---
    training_args = TrainingArguments(
        logging_dir=params.LOGGING_DIR,
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_steps=params.TOTAL_STEPS,
        learning_rate=params.LR,   # base LR passed to Trainer, overridden by our custom groups
        weight_decay=params.WEIGHT_DECAY,
        warmup_steps=params.WARMUP_STEPS,      # warm up for 1000 steps
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=50,
        optim="adamw_torch",
        bf16=True,          # enable bfloat16 (H100 optimized)
        fp16=False,         
        report_to="tensorboard",
        ddp_find_unused_parameters=False
    )
    
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        # target_modules=["q_proj", "v_proj"],  # attention projections
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    peft_model = get_peft_model(model, lora_config)

    # Freeze all base model parameters (done automatically by get_peft_model)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    
    # --- Trainer ---
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: sft_data_collator(batch, tokenizer),  # use custom collator
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=32, help="Training batch size")
    parser.add_argument("--LORA_RANK", type=int, default=16, help="Rank of LoRA adaptor")
    parser.add_argument("--LORA_RATIO", type=float, default=1, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=10000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")
    parser.add_argument("--POLY_POW", type=float, default=2.0, help="Polynomial LR scheduler power")

    

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_lora_rank{Params.LORA_RANK}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}_cosine_combined"
    Params.LOGGING_DIR =  config.RUN_DIR / "train_seq_pred_aligned_phase1" / run_name

    # Load model and tokenizer in local device
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    save_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_all_sid_alignment"
    # Load model to cpu first and let torchrun handle the device placement
    model, tokenizer = load_checkpoint(base_model_name, save_dir) 
    print(f"model_device: {model.device}")
    old_vocab_size = 128_256
    
    ##### Generate 2% fixed subset for parameter selection ####
    # SUBSET_SIZE = 52_000
    SUBSET_SIZE = 252_000
    SEED = 411
    TRAIN_INDEX_FILE = config.DATA_DIR / "train_subset_52k_seed411.json"
    train_dataset = SeqDataset(tokenizer, "train")  
    print(f"---Train dataset size: {len(train_dataset)}")
    try:
        # Reuse existing subset if it exists
        with open(TRAIN_INDEX_FILE, "r") as f:
            indices = json.load(f)
        print(f"Loaded existing subset: {len(indices)} samples")

    except FileNotFoundError:
        # Create subset deterministically
        rng = random.Random(SEED)   # <- LOCAL RNG (important!)
        indices = rng.sample(range(len(train_dataset)), SUBSET_SIZE)
        indices = sorted(indices)   # optional but recommended

        with open(TRAIN_INDEX_FILE, "w") as f:
            json.dump(indices, f)

        print(f"---Created new subset: {len(indices)} samples")


    train_dataset = Subset(train_dataset, indices)
    print("Subset fingerprint:", indices[:10])

    # Test purpose
    # for i in range(10):
    #     print(train_dataset[i]["labels"])
    #     print(tokenizer.decode(train_dataset[i]["labels"][train_dataset[i]["labels"]!=-100]))

    eval_dataset = SeqDataset(tokenizer, "eval")
    print(f"---Eval dataset size: {len(eval_dataset)}")

    # gen_eval_dataset = SeqGenDataset("eval")
    gen_eval_dataset = None

    train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, Params)
    

if __name__ == "__main__":
    main()