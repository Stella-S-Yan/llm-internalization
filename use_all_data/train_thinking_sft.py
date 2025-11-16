"""
Phase 1 training for seq pred. Use aligned new embeddings; fix all embeddings; only tune LoRA parameter.

Able to achieve 4.97% recall@5



DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py
"""

import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import argparse
from use_all_data import train_thinking
from utils import merge_save_load_model



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
    ADAPTOR_SAVE_DIR = ''


def train(model, tokenizer, train_dataset, eval_dataset, params):
    print(f"@@@ total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=config.MODEL_DIR / params.ADAPTOR_SAVE_DIR,
        logging_dir=params.LOGGING_DIR,
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_steps=params.TOTAL_STEPS,
        learning_rate=params.LR,   # base LR passed to Trainer, overridden by our custom groups
        weight_decay=params.WEIGHT_DECAY,
        warmup_steps=params.WARMUP_STEPS,      # warm up for 1000 steps
        # lr_scheduler_type="cosine_with_min_lr",  # can also try "cosine", "linear"
        # lr_scheduler_kwargs={
        #     "lr_floor": 1e-5,
        # },
        lr_scheduler_type="linear",
        logging_steps=50,
        save_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        optim="adamw_torch",
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
    

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: train_thinking.sft_data_collator(batch, tokenizer),  # use custom collator
    )


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
    parser.add_argument("--ADAPTOR_SAVE_DIR", type=str, default='train_thinking_sft', help="Where to save the trained adaptor")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}"
    Params.LOGGING_DIR =  config.RUN_DIR / "train_thinking_sft" / run_name

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    model_input_dir = config.MODEL_DIR / "all_sid_aligned_model"
    model, tokenizer = merge_save_load_model.load_model(model_input_dir)
    old_vocab_size = 128_256

    train_dataset = train_thinking.ReasoningDataset("train", "sft")
    eval_dataset = train_thinking.ReasoningDataset("eval", "sft")

    
    train(model, tokenizer, train_dataset, eval_dataset, Params)

    
if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()