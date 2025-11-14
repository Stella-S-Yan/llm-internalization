"""
DPO trainer uses accelerate for device placement and batch management. 
torchrun sets up a process per GPU, but if DPOTrainer isn't wrapped for distributed training, 
only one process will actually do training. 

Cannot use multiprocess
$ torchrun --nproc_per_node=8 train_DPO.py

So, use accelerate for multi-GPU
$ accelerate launch train_DPO.py
"""


from trl import DPOTrainer, DPOConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import os
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import argparse
import numpy as np
from datasets import load_dataset, Features, Value
import random
from tqdm import tqdm
from transformers import TrainerCallback
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator


from trl import (
    DPOConfig,
    DPOTrainer,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_INPUT_DIR = config.MODEL_DIR / f"merged_best_sft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"train_DPO"


class Params:
    TRAIN_BATCH_SIZE = 16
    LR = 4e-4
    WEIGHT_DECAY = 1e-3
    TOTAL_STEPS = 16_000    # 13_000

    LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
    LORA_RANK = 16      # 16 large rank overfit early
    LORA_RATIO = 1
    WARMUP_STEPS = 1000    # 2k warmups is much better than 3K warmup
    LOGGING_DIR = "/"
        

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_INPUT_DIR, 
        device_map=None,  # for accelerate
        # device_map='auto',  # for single device
        dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INPUT_DIR)
    print(len(tokenizer))
    
    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    # print("tokenizer eos:", tokenizer.eos_token_id)
    # print("model config eos:", model.config.eos_token_id)
    # print("generation config eos:", model.generation_config.eos_token_id)

    # print("tokenizer pad:", tokenizer.pad_token_id)
    # print("model config pad:", model.config.pad_token_id)
    # print("generation config pad:", model.generation_config.pad_token_id)

    return model, tokenizer


def get_data():
    
    data_files = {
        "train": str(config.PROCESSED_DATA_DIR / "dpo_train_*.jsonl"),
        "eval": str(config.PROCESSED_DATA_DIR / "dpo_eval_*.jsonl"),
    }

    # Explicitly define the dataset schema including 'target'
    features = Features({
        "prompt": Value("string"),
        "chosen": Value("string"),
        "rejected": Value("string"),
        "target": Value("string"),  # new field for reference-based metrics
    })

    dataset = load_dataset("json", data_files=data_files, features=features)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    # Keep only first 100 examples for fast testing
    # train_dataset = train_dataset.select(range(100))
    # eval_dataset = eval_dataset.select(range(50))
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    # for i in range(10):
    #     print(train_dataset[i])
    return train_dataset, eval_dataset



def train(model, tokenizer, train_dataset, eval_dataset, params):

    
    lora_config = LoraConfig(
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        # target_modules=["q_proj", "v_proj"],  # attention projections
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # DPO config (beta is temperature-like for preference)
    dpo_args = DPOConfig(
        output_dir=MODEL_SAVE_DIR,
        learning_rate=params.LR,
        warmup_steps=params.WARMUP_STEPS,  # example: 50 steps or ~5% of total steps
        lr_scheduler_type="linear",  # options: linear, cosine, cosine_with_restarts
        per_device_train_batch_size=64, #64,  
        per_device_eval_batch_size=64, #64,
        beta=params.BETA,       # 0.01~0.05, Set beta very small, essentially letting DPO act as pure preference-based learning. for new token-heavy task
        max_prompt_length=256,
        max_completion_length=8,
        report_to=["tensorboard"],
        logging_dir=params.LOGGING_DIR, 
        logging_steps=100,
        eval_strategy="steps",   
        eval_steps=500,
        fp16=False,
        bf16=True,      # for unsloth   
        num_train_epochs=5,             
        save_strategy="steps",
        save_steps=1000,
        # save_total_limit=5, 
        dataset_num_proc=5,     # num of processes to process data
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,     # because train a peft model
        args=dpo_args,
        data_collator= None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # your tokenizer
        peft_config=lora_config,
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=32, help="Training batch size")
    parser.add_argument("--LORA_RATIO", type=float, default=0.1, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=20000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")
    parser.add_argument("--BETA", type=float, default=0.5, help="beta for DPO")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"ref_lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}"
    Params.LOGGING_DIR =  config.RUN_DIR / "train_dpo" / run_name

    model, tokenizer = load_model()

    train_dataset, eval_dataset = get_data()

    train(model, tokenizer, train_dataset, eval_dataset, Params)


if __name__ == '__main__':
    main()