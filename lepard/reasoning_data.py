
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
import pandas as pd
import re



PROMPT_TEMPLATE = """
<sft:think>
<dcourt>  {dest_court} </dcourt>
<ddate> {dest_date} </ddate>
<dname> {dest_name} </dname>
<dsid> {dest_formatted_sid} </dsid>
quote:
{target}
"""

TARGET_TEMPLATE = """
<scourt>  {source_court} </scourt>
<sdate> {source_date} </sdate>
<sname> {source_name} </sname>
<ssid> {quote_formatted_sid} </ssid>{eos}
"""


def load_checkpoint(base_model_name, save_dir):
    # Load BASE MODEL again — quantized or FP16 as desired
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,   # or fp16, or load_in_4bit=True
    )

    # 2. Load extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    tokenizer.padding_side = "left"

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


class LepardDataset(Dataset):

    def __init__(self, datatype, tokenizer, split):
        self.datatype = datatype
        self.tokenizer = tokenizer

        if split == "train":
            self.df = pd.read_parquet(config.LEPARD_TRAIN)
        elif split == "eval":
            self.df = pd.read_parquet(config.LEPARD_EVAL)
        elif split == "test":
            self.df = pd.read_parquet(config.LEPARD_TEST)


    def __len__(self):
        return self.df.shape[0]
    

    def __getitem__(self, idx):
        record = self.df.iloc[idx]

        prompt_text = PROMPT_TEMPLATE.format(
            dest_name=record["dest_name"],
            dest_court=record["dest_court"],
            dest_date=record["dest_date"],
            dest_formatted_sid=record["dest_formatted_sid"],
            target=""
        ).strip()
        
        target_text = TARGET_TEMPLATE.format(
            source_name=record["source_name"],
            source_court=record["source_court"],
            source_date=record["source_date"],
            quote_formatted_sid=record["quote_formatted_sid"],
            eos=self.tokenizer.eos_token
        ).strip()

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

        solution = {
            "scourt": record["source_court"],
            "sdate": record["source_date"],
            "sname": record["source_name"],
            "ssid": record["quote_formatted_sid"],
            "row_id": record["row_id"]
        }
        
        if self.datatype == "sft":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        elif self.datatype == "grpo":
            return {
                "prompt": prompt_text,
                "solution": solution,
            }
        else:
            raise ValueError(
                f"Invalid datatype '{self.datatype}'. "
                f"Expected one of: ['sft', 'grpo']"
            )

