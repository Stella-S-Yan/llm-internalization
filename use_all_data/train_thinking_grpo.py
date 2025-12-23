"""
Tune think-sft with grpo. 

To run this script, first start a vLLM server. Install $ pip install trl[vllm]
# go to the parent directory of think_model_sft
$ trl vllm-serve --model think_model_sft
# run the script
$ sh sweep_train_think_grop.sh
"""

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"   # <-- must come before torch/vllm imports

import config
import torch
from peft import LoraConfig, get_peft_model
import argparse
from use_all_data import train_thinking
from utils import merge_save_load_model
from trl import GRPOConfig, GRPOTrainer
import re
from typing import List
import numpy as np


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


def extract_hist_predictions(text: str) -> List[str]:
    """
    Extract predicted category strings from <hist> lines like:
       "0: Face", "1: Chemical Hair Dyes", ...
    Also appends the content of <cat> if present.
    Returns e.g. ["Face", "Chemical Hair Dyes", "Treatments & Masks", "Sets & Kits"]
    """
    preds = []

    # 1) Get hist content, split into real lines, handle different whitespace quirks
    hist_text = extract_tag(text, "hist")
    if hist_text:
        # Ensure we have real line breaks; if the content uses escaped '\n', fix that:
        # (only do this if you actually see backslash-n sequences in your strings)
        if r'\n' in hist_text and '\n' not in hist_text:
            hist_text = hist_text.replace(r'\n', '\n')

        for line in hist_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Prefer regex capture (robust to spaces)
            m = HIST_LINE_REGEX.search(line)
            if m:
                preds.append(m.group(1).strip())
                continue
            # Fallback: split on first ':' and take the right side
            if ':' in line:
                _, right = line.split(':', 1)
                preds.append(right.strip())
            else:
                # If no colon, assume the whole line is the label
                preds.append(line)

    # 2) Append the <cat> content if present (user expects this as last element)
    cat_text = extract_tag(text, "cat")
    if cat_text:
        preds.append(cat_text.strip())

    return preds


def hsz_reward(completions, **kwargs):
    rewards = []
    solutions = kwargs.get("solution", [])
    for i, complete in enumerate(completions):
        true_data = solutions[i] if i < len(solutions) else {}

        # --- hsz (small bonus) ---
        hsz_pred = extract_tag(complete, "hsz")
        try:
            hsz_pred = int(hsz_pred)
            if hsz_pred == true_data.get("hsz"):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except (TypeError, ValueError):
            rewards.append(0.0)

    return rewards


def hist_reward(completions, **kwargs):
    rewards = []
    solutions = kwargs.get("solution", [])
    for i, complete in enumerate(completions):
        true_data = solutions[i] if i < len(solutions) else {}

        hist_pred = extract_hist_predictions(complete) or []
        true_hist = true_data.get("hist") or []
        matches = sum(p == t for p, t in zip(hist_pred, true_hist))
        if len(true_hist) > 0:
            rewards.append(matches / len(true_hist))
        else:
            rewards.append(0.0)

    return rewards


def cat_reward(completions, **kwargs):
    rewards = []
    solutions = kwargs.get("solution", [])
    for i, complete in enumerate(completions):
        true_data = solutions[i] if i < len(solutions) else {}

        # cat (medium bonus) ---
        cat_pred = extract_tag(complete, "cat")
        if cat_pred is not None and cat_pred == true_data.get("cat"):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def hierarchy_reward(completions, **kwargs):
    rewards = []
    solutions = kwargs.get("solution", [])
    for i, complete in enumerate(completions):
        true_data = solutions[i] if i < len(solutions) else {}

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
            if true_tokens:
                rewards.append(match_len / len(true_tokens))
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def weighted_reward(completions, **kwargs):

    hist = np.array(hist_reward(completions, **kwargs))
    cat = np.array(cat_reward(completions, **kwargs))
    hier = np.array(hierarchy_reward(completions, **kwargs))

    total = 1.0 * hist + 2.0 * cat + 3.0 * hier
    return total.tolist()
    

def train(model, tokenizer, train_dataset, params):
    
    # Define LoRA config
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        target_modules=["q_proj", "v_proj"],  # attention projections
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    peft_model.print_trainable_parameters()


    # --- GRPO RL setup ---
    grpo_config = GRPOConfig(
        output_dir=config.MODEL_DIR / params.ADAPTOR_SAVE_DIR,
        learning_rate=params.LR,
        beta=0.2, # KL coefficient
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        use_vllm=True,
        vllm_mode="server",  # default value, can be omitted
        vllm_server_base_url="http://0.0.0.0:8000",
        gradient_checkpointing=True,
        shuffle_dataset=True,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        bf16=True, 

        # Parameters that control the data preprocessing
        num_generations=8,
        steps_per_generation=1,     # 2~4 for small dataset (few thousand prompts)
        max_prompt_length=188, 
        max_completion_length=210,

        # sample
        top_k=50,
        
        # optimizer & scheduler
        optim="adamw_torch",          # supports 'adamw_torch', 'adamw_hf', 'adamw_apex', etc.
        lr_scheduler_type="linear",   # can use "cosine", "cosine_with_restarts", "polynomial", etc.
        warmup_steps=params.WARMUP_STEPS,            # optional, for scheduler
        
        # max_grad_norm=1.0,
        logging_dir=params.LOGGING_DIR,
        save_strategy="steps",
        save_steps=10,
        logging_steps=10,
        # eval_strategy="steps",
        # eval_steps=50,
        
        report_to="tensorboard",
        
        sync_ref_model=True,
        ref_model_mixup_alpha=0.6,  # default
        ref_model_sync_steps=200
    )

    trainer = GRPOTrainer(
        model=peft_model,
        reward_funcs=weighted_reward,
        args=grpo_config,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        
    )

    trainer.train()

    trainer.save_model(grpo_config.output_dir)


def main():

    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=8, help="Training batch size")
    parser.add_argument("--LORA_RANK", type=int, default=8, help="LoRA rank")
    parser.add_argument("--LORA_RATIO", type=float, default=4, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=20000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")
    parser.add_argument("--ADAPTOR_SAVE_DIR", type=str, default='train_think_grpo_adaptor', help="Adaptor save dir")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_lora_rank{Params.LORA_RANK}_lora_ratio{Params.LORA_RATIO}_num_epocs{1}"
    Params.LOGGING_DIR =  config.RUN_DIR / "train_think_grpo" / run_name
    Params.ADAPTOR_SAVE_DIR = config.MODEL_DIR / "train_think_grpo_adaptor"

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # Load model + tokenizer
    model_input_dir = config.MODEL_DIR / "Amazon_Combined_merged_think_sft_model"
    model, tokenizer = merge_save_load_model.load_model(model_input_dir)
    old_vocab_size = 128_256

    train_dataset = train_thinking.ReasoningDataset("train", "grpo")
    # eval_dataset = train_thinking.ReasoningDataset("eval", "grpo")

    train(model, tokenizer, train_dataset, Params)
    # train(model, tokenizer, eval_dataset, Params)

    
    

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()