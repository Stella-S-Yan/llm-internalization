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
from utils import merge_save_model
from trl import GRPOConfig, GRPOTrainer
import random
import re


MODEL_INPUT_DIR = config.MODEL_DIR / "all_sid_aligned_model"
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


TAG_REGEX_CACHE = {}
HIST_LINE_REGEX = re.compile(r"->\s*(.+)$")

def extract_tag(text, tag):
    """Extract <tag>...</tag> using a cached regex, returns None if missing."""
    if tag not in TAG_REGEX_CACHE:
        TAG_REGEX_CACHE[tag] = re.compile(fr"<{tag}>(.*?)</{tag}>", re.DOTALL)

    m = TAG_REGEX_CACHE[tag].search(text)
    return m.group(1).strip() if m else None


def extract_hist_predictions(text):
    """
    Extracts only the predicted category part after '->'.
    Returns a list of strings like:
        ["Oils", "Oils & Serums", "Masks", ...]
    """
    hist_text = extract_tag(text, "hist")
    if not hist_text:
        return []

    preds = []
    for line in hist_text.splitlines():
        m = HIST_LINE_REGEX.search(line)
        if m:
            preds.append(m.group(1).strip())
    return preds


def semantic_reward_fn(completions, prompts=None, **kwargs):
    """
    completions: list of dicts {"content": text_generated"}
    kwargs["solution"]: list of dicts per prompt with keys:
        "hsz": int
        "hist": list of categories
        "cat": str
        "sid": str
    Returns: list of reward floats, one per completion
    """
    rewards = []

    solutions = kwargs.get("solution", [])
    for i, complete in enumerate(completions):
        true_data = solutions[i] if i < len(solutions) else {}
        r = 0.0

        # --- 1. hsz (small bonus) ---
        hsz_pred = extract_tag(complete, "hsz")
        try:
            hsz_pred = int(hsz_pred)
            if hsz_pred == true_data.get("hsz"):
                r += 0.1
        except (TypeError, ValueError):
            pass  # skip if None or not int

        # --- 2. hist (small bonus) ---
        hist_pred = extract_hist_predictions(complete) or []
        true_hist = true_data.get("hist") or []
        matches = sum(p == t for p, t in zip(hist_pred, true_hist))
        if len(true_hist) > 0:
            r += 0.2 * matches / len(true_hist)

        # --- 3. cat (medium bonus) ---
        cat_pred = extract_tag(complete, "cat")
        if cat_pred is not None and cat_pred == true_data.get("cat"):
            r += 0.2

        # --- 4. sid (dominant) ---
        sid_pred = extract_tag(complete, "sid")
        true_sid = true_data.get("sid")
        if sid_pred is not None and true_sid is not None:
            pred_tokens = sid_pred.split()
            true_tokens = true_sid.split()
            match_len = 0
            for pt, tt in zip(pred_tokens, true_tokens):
                if pt == tt:
                    match_len += 1
                else:
                    break
            if len(true_tokens) > 0:
                r += 1.0 * (match_len / len(true_tokens))
        # If sid_pred or true_sid is None, just skip / reward nothing

        rewards.append(r)

    return rewards



def train(model, tokenizer, train_dataset, eval_dataset, params):
    
    # Define LoRA config
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        # target_modules=["q_proj", "v_proj"],  # attention projections
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    peft_model.print_trainable_parameters()

    # Freeze all base model parameters (done automatically by get_peft_model)
    # for name, param in peft_model.named_parameters():
    #     if "lora_" not in name:
    #         param.requires_grad = False
    

    # --- GRPO RL setup ---
    grpo_config = GRPOConfig(
        output_dir=config.MODEL_DIR / params.ADAPTOR_SAVE_DIR,
        learning_rate=params.LR,
        beta=0.0, # KL coefficient
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        gradient_checkpointing=True,
        shuffle_dataset=True,
        num_generations=4,
        max_completion_length=393,
        # generation_kwargs={
        #     "max_new_tokens": 393,
        #     "do_sample": True,
        #     "top_k": 50,
        #     "top_p": 0.9,
        #     "temperature": 0.7,
        # },
        # optimizer & scheduler
        optim="adamw_torch",          # supports 'adamw_torch', 'adamw_hf', 'adamw_apex', etc.
        lr_scheduler_type="linear",   # can use "cosine", "cosine_with_restarts", "polynomial", etc.
        warmup_steps=params.WARMUP_STEPS,            # optional, for scheduler
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        # max_steps=params.TOTAL_STEPS,
        logging_dir=params.LOGGING_DIR,
        save_strategy="steps",
        save_steps=10,
        # eval_strategy="steps",
        # eval_steps=50,
        bf16=True, 
        report_to="tensorboard",
        num_train_epochs=1,
        logging_steps=10,
        use_vllm=True,
        vllm_mode="server",  # default value, can be omitted
        vllm_server_base_url="http://0.0.0.0:8000"
    )

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        reward_funcs=semantic_reward_fn,
    )

    trainer.train()

    trainer.save_model(grpo_config.output_dir)


def main():

    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=8, help="Training batch size")
    parser.add_argument("--LORA_RANK", type=int, default=8, help="LoRA rank")
    parser.add_argument("--LORA_RATIO", type=float, default=4, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=20000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")
    parser.add_argument("--ADAPTOR_SAVE_DIR", type=str, default=' ', help="Adaptor save dir")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_lora_rank{Params.LORA_RANK}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_num_epocs{1}"
    Params.LOGGING_DIR =  config.RUN_DIR / "train_think_grpo" / run_name
    Params.ADAPTOR_SAVE_DIR = config.MODEL_DIR / "train_think_grpo/adaptor"

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # Load model + tokenizer
    model_input_dir = config.MODEL_DIR / "think_model_best"
    model, tokenizer = merge_save_model.load_merged_model(model_input_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    old_vocab_size = 128_256

    train_dataset = train_thinking.SeqReasoningDataset(tokenizer, "train")
    eval_dataset = train_thinking.SeqReasoningDataset(tokenizer, "eval")

    train(model, tokenizer, train_dataset, eval_dataset, Params)

    
    

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()