"""
CAn run, but does not work well yet. Rewards just bounces around.
"""


from unsloth import FastLanguageModel


import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import train_thinking
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
    ACC_STEP = 1


TAG_REGEX_CACHE = {}
HIST_LINE_REGEX = re.compile(r"->\s*(.+)$")


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


def extract_tag(text, tag):
    """Extract <tag>...</tag> using a cached regex, returns None if missing."""
    if tag not in TAG_REGEX_CACHE:
        TAG_REGEX_CACHE[tag] = re.compile(fr"<{tag}>(.*?)</{tag}>", re.DOTALL)

    m = TAG_REGEX_CACHE[tag].search(text)
    return m.group(1).strip() if m else None



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


def freq_reward(completions, **kwargs):
    rewards = []
    solutions = kwargs.get("solution", [])
    for i, complete in enumerate(completions):
        true_data = solutions[i] if i < len(solutions) else {}

        freq_pred = extract_tag(complete, "freq")
        if freq_pred is not None and freq_pred == true_data.get("freq"):
            rewards.append(1.0)
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

    hist = np.array(freq_reward(completions, **kwargs))
    cat = np.array(cat_reward(completions, **kwargs))
    hier = np.array(hierarchy_reward(completions, **kwargs))

    total = 1.0 * hist + 2.0 * cat + 3.0 * hier
    return total.tolist()
    

def train(model, tokenizer, train_dataset, params):
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=params.LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    max_prompt_length = 512

    # --- GRPO RL setup ---
    grpo_config = GRPOConfig(
        output_dir=config.MODEL_DIR / params.ADAPTOR_SAVE_DIR,
        logging_dir=params.LOGGING_DIR,
        learning_rate=params.LR,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Increase to 4 for smoother training

        num_generations=20,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=40,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=1000,
        save_steps=50,
        max_grad_norm=0.1,
        
        report_to="tensorboard",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=weighted_reward,
        args=grpo_config,
        train_dataset=train_dataset,
        
    )

    trainer.train()

    # trainer.save_model(grpo_config.output_dir)


def main():

    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=2, help="Training batch size")
    parser.add_argument("--LORA_RANK", type=int, default=16, help="LoRA rank")
    parser.add_argument("--LORA_RATIO", type=float, default=1, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=1000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.1, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0, help="LoRA dropout rate")
    parser.add_argument("--ACC_STEP", type=int, default=1, help="Gradient accumulate steps")
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
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_merged_think_sft_model"
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(model_dir),
        load_in_4bit=True,
        fast_inference=True, # Enable vLLM fast inference
        max_lora_rank=Params.LORA_RANK,
        gpu_memory_utilizaton=0.6,  # Reduce if out of memory
    )
    old_vocab_size = 128_256

    train_dataset = train_thinking.ReasoningDataset("train", "grpo", ["Toys_and_Games", "Sports_and_Outdoors", "Beauty"])
    # eval_dataset = train_thinking.ReasoningDataset("eval", "grpo")

    train(model, tokenizer, train_dataset, Params)
    # train(model, tokenizer, eval_dataset, Params)

    
    

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()