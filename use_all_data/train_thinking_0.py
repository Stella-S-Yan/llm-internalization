"""
Phase 1 training for seq pred. Use aligned new embeddings; fix all embeddings; only tune LoRA parameter.

Able to achieve 4.97% recall@5



DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py
"""

import json
import random
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
import numpy as np
import bagz
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from torch.utils.data import Subset
from use_all_data import train_thinking


MODEL_INPUT_DIR = config.MODEL_DIR / "all_sid_aligned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"train_thinking_1"



class Params:
    TRAIN_BATCH_SIZE = 16
    LR = 4e-4
    WEIGHT_DECAY = 1e-3
    TOTAL_STEPS = 16_000    # 13_000

    LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
    LORA_RANK = 16      # 16 large rank overfit early
    LORA_RATIO = 1
    WARMUP_STEPS = 1000    # 2k warmups is much better than 3K warmup




def train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, params):
    print(f"@@@ total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        logging_dir=params.LOGGING_DIR,
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        # num_train_epochs=EPOCHS,
        max_steps=params.TOTAL_STEPS,
        learning_rate=params.LR,   # base LR passed to Trainer, overridden by our custom groups
        weight_decay=params.WEIGHT_DECAY,
        warmup_steps=params.WARMUP_STEPS,      # warm up for 1000 steps
        lr_scheduler_type="cosine",  # can also try "cosine", "linear"
        # lr_scheduler_kwargs={
        #     "decay_steps": 6000,
        #     "constant_steps": 30000,
        #     "lr_floor": 6e-6,
        # },
        logging_steps=50,
        # save_strategy="steps",
        # save_steps=1000,
        save_strategy="no",
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=500,
        # eval_strategy="no",
        optim="adamw_torch",
        # optim="adafactor",
        bf16=True,          # <<< enable bfloat16 (H100 optimized)
        fp16=False,         # optional: if you want fp16 instead
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
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
    # trainer = train_thinking.CustomTrainer(
    #     model=peft_model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=lambda batch: train_thinking.sft_data_collator(batch, tokenizer),  # use custom collator
    # )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: train_thinking.sft_data_collator(batch, tokenizer),  # use custom collator
    )

    callback = train_thinking.GenerateEvalCallback(
        trainer=trainer,
        eval_dataset=gen_eval_dataset,
        tokenizer=tokenizer,
        eval_fn=train_thinking.evaluate_sequence_recall,
        eval_steps=4000
    )
    trainer.add_callback(callback)

    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=32, help="Training batch size")
    parser.add_argument("--LORA_RATIO", type=float, default=0.1, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=20000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"level1_lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}"
    Params.LOGGING_DIR =  config.RUN_DIR / "train_thinking_0" / run_name

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    model, tokenizer = train_thinking.load_model_tokenizer()
    old_vocab_size = 128_256

    level = 1

    train_dataset = train_thinking.SeqDataset(tokenizer, "train", level=level)
    eval_dataset = train_thinking.SeqDataset(tokenizer, "eval", level=level)

    # train_dataset_0 = train_thinking.SeqDataset(tokenizer, "train", level=0)
    # eval_dataset_0 = train_thinking.SeqDataset(tokenizer, "eval", level=0)

    # train_dataset_1 = train_thinking.SeqDataset(tokenizer, "train", level=1)
    # eval_dataset_1 = train_thinking.SeqDataset(tokenizer, "eval", level=1)

    # train_dataset_2 = train_thinking.SeqDataset(tokenizer, "train", level=2)
    # eval_dataset_2 = train_thinking.SeqDataset(tokenizer, "eval", level=2)

    # train_dataset_3 = train_thinking.SeqDataset(tokenizer, "train", level=3)
    # eval_dataset_3 = train_thinking.SeqDataset(tokenizer, "eval", level=3)

    # combined_train = ConcatDataset([train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3])
    # combined_eval  = ConcatDataset([eval_dataset_0, eval_dataset_1, eval_dataset_2, eval_dataset_3])

    gen_eval_dataset = train_thinking.SeqGenDataset(tokenizer, "eval", level=level)

    # train_dataset = Subset(train_dataset, range(100))
    # eval_dataset = Subset(eval_dataset, range(10))
    # gen_eval_dataset = Subset(gen_eval_dataset, range(10))

    # train(model, tokenizer, combined_train, combined_eval, gen_eval_dataset, Params)

    train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, Params)
    

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()