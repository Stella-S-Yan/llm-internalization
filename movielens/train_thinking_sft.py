"""
Phase 1 training for seq pred. Use aligned new embeddings; fix all embeddings; only tune LoRA parameter.
Use all types of reivews.



DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py
"""

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
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from combined_data import train_thinking
from accelerate import Accelerator




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"{config.DATA_SOURCE}_think_sft_adaptor"
SID_PATTERN = re.compile(r"<ssid>(.*?)</")

# Set seeds for reproducibility
seed = 411
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class Params:
    TRAIN_BATCH_SIZE = 16
    LR = 4e-4
    WEIGHT_DECAY = 1e-3
    TOTAL_STEPS = 16_000    # 13_000

    LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
    LORA_RANK = 16      # 16 large rank overfit early
    LORA_RATIO = 1
    WARMUP_STEPS = 1000    # 2k warmups is much better than 3K warmup
    ACC_STEP = 1


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


@torch.no_grad()
def evaluate_sequence_recall(
    model,
    tokenizer,
    eval_loader,
    num_beams=20,
    max_new_tokens=30,
    top_k_list=[5],
    print_random_example=True,  # new flag
):
    """
    Batched sequence-level recall evaluation.

    Args:
        model: Hugging Face causal LM
        tokenizer: Hugging Face tokenizer
        eval_dataset: list of dicts with 'prompt' and 'target' fields
        batch_size: number of prompts per batch
        num_beams: number of beams for beam search
        max_new_tokens: maximum tokens to generate
        top_k_list: which recalls to compute (e.g., [1,5,10])

    Returns:
        dict: {'recall_1': float, 'recall_5': float, ...}
    """
    model.eval()
    device = model.device

    # Initialize recall lists
    recalls_dict = {k: [] for k in top_k_list}
    printed = False  # track if we've printed already

    # Process dataset in batches
    for batch in tqdm(eval_loader, desc="Evaluating"):
        prompts = batch["gen_prompt"]
        targets = batch["gen_target"]

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        batch_size = len(prompts)
        max_k = max(top_k_list)

        
        # Generate sequences for the batch
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, max(top_k_list)),
            num_return_sequences=max(top_k_list),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Reshape outputs: (batch_size, num_return_sequences, seq_len)
        batch_outputs = outputs.view(batch_size, max(top_k_list), -1)

        # Decode and compute top-k recall
        for i in range(batch_size):
            prompt_len = inputs["input_ids"].size(1)
            decoded_outputs = [
                tokenizer.decode(batch_outputs[i, k, prompt_len:], skip_special_tokens=True)
                for k in range(max_k)
            ]
            # print(decoded_outputs)
            match = SID_PATTERN.search(targets[i])
            if match:
                sid_value = match.group(1)
            else:
                sid_value = "NONE"

            hits = [1 if sid_value in o else 0 for o in decoded_outputs]
            for k in top_k_list:
                recalls_dict[k].append(int(any(hits[:k])))


        # ---- Print one random batch example ----
        if print_random_example and not printed:
            rand_idx = random.randint(0, batch_size - 1)
            print("\n=== Random Example ===")
            print(f"Prompt:\n{prompts[rand_idx]}")
            print(f"Target:\n{targets[rand_idx]}")
            for k, gen in enumerate(decoded_outputs[:5]):  # show top 5 generations
                print(f"[Gen {k+1}] {gen}")
            print("========================\n")
            printed = True
        # ----------------------------------------

    # Compute mean recall
    recalls_mean = {f"recall_{k}": float(np.mean(v)) for k, v in recalls_dict.items()}
    return recalls_mean


def no_processing_collator(batch):
    return {
        key: [example[key] for example in batch]
        for key in batch[0].keys()
    }


class GenerateEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tokenizer, eval_fn, eval_steps, eval_data_collator):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.batch_size = 8
        self.best_metric = None  # Track best metric
        self.eval_data_collator = eval_data_collator

    # def on_step_end(self, args, state, control, **kwargs):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_interval = self.eval_steps

        # Run every eval_steps
        if state.global_step > 0 and state.global_step % eval_interval == 0:

            is_ddp = (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )

            rank = torch.distributed.get_rank() if is_ddp else 0
            world_size = torch.distributed.get_world_size() if is_ddp else 1


            # ---- Sampler (per dataset) ----
            sampler = (
                DistributedSampler(self.eval_dataset, shuffle=False)
                if is_ddp
                else None
            )

            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                collate_fn=None,
            )

            # tqdm only on rank 0
            if rank == 0:
                eval_loader = tqdm(
                    eval_loader,
                    desc=f"Eval @ step {state.global_step}",
                )

            # ---- Custom generate-based eval ----
            metrics = self.eval_fn(
                self.trainer.model,
                self.tokenizer,
                eval_loader,
            )

            # ---- DDP reduce (mean) ----
            if is_ddp:
                for k, v in metrics.items():
                    tensor = torch.tensor(v, device=self.trainer.model.device)
                    torch.distributed.all_reduce(
                        tensor, op=torch.distributed.ReduceOp.SUM
                    )
                    metrics[k] = (tensor / world_size).item()

            # ---- Prefix metrics for TensorBoard ----
            metrics = {
                f"eval/{k}": v
                for k, v in metrics.items()
            }
            metrics["step"] = state.global_step

            # ---- Log ----
            self.trainer.log(metrics)

            if rank == 0:
                print(
                    f"\n[Custom eval @ step {state.global_step}] "
                    f"{metrics}"
                )

        return control


class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best = float('inf')
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best:
            self.best = eval_loss
            control.should_save = True  # save checkpoint this step
        else:
            control.should_save = False  # skip checkpoint
        return control


class IterableTrainer(Trainer):
    def __init__(self, train_loader=None, eval_loader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_loader = train_loader  # store the pre-wrapped DataLoader
        self._eval_loader = eval_loader

    def get_train_dataloader(self):
        # Return the pre-wrapped DataLoader
        if self._train_loader is None:
            # Fallback to default behavior if not passed
            return super().get_train_dataloader()
        return self._train_loader

    def get_eval_dataloader(self, eval_dataset=None):
        # use your custom eval loader if provided
        if self._eval_loader is not None:
            return self._eval_loader
        # fallback to default behavior, passing eval_dataset
        return super().get_eval_dataloader(eval_dataset)
    

def train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, params):
    print(f"@@@ total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        logging_dir=params.LOGGING_DIR,
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=params.ACC_STEP,
        max_steps=params.TOTAL_STEPS,
        learning_rate=params.LR,   # base LR passed to Trainer, overridden by our custom groups
        weight_decay=params.WEIGHT_DECAY,
        warmup_steps=params.WARMUP_STEPS,      # warm up for 1000 steps
        lr_scheduler_type="cosine",
        logging_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=20,
        load_best_model_at_end=False,
        eval_strategy="steps",
        eval_steps=10,
        optim="adamw_torch",
        bf16=True,          # enable bfloat16 (H100 optimized)
        fp16=False,         
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        remove_unused_columns=False,  # REQUIRED for IterableDataset
        dataloader_drop_last=False,
        dataloader_pin_memory=True,
    )
    
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        target_modules=[
            "q_proj",
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    peft_model = get_peft_model(model, lora_config)

    # Freeze all base model parameters (done automatically by get_peft_model)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.TRAIN_BATCH_SIZE,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: train_thinking.sft_data_collator(batch, tokenizer)
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=params.TRAIN_BATCH_SIZE,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: train_thinking.sft_data_collator(batch, tokenizer)
    )

    # --- Trainer ---
    trainer = IterableTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_loader=train_loader,
        eval_loader=eval_loader
    )

    callback = GenerateEvalCallback(
        trainer=trainer,
        eval_dataset=gen_eval_dataset,
        tokenizer=tokenizer,
        eval_fn=evaluate_sequence_recall,
        eval_steps=10,
        eval_data_collator=lambda batch: no_processing_collator(batch),
    )
    trainer.add_callback(callback)

    trainer.train()
    # trainer.train(resume_from_checkpoint="/usr/local/google/home/stellasyan/Documents/llm_internalization/data/model/ML_1m_think_sft_adaptor/checkpoint-424000")


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
    parser.add_argument("--ACC_STEP", type=int, default=1, help="Gradient accumulate steps")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"ML_{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_acc_step{Params.ACC_STEP}_warmup_{Params.WARMUP_STEPS}_lora_rank{Params.LORA_RANK}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}"
    Params.LOGGING_DIR =  config.RUN_DIR / "ML_train_seq_pred" / run_name

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # Load model and tokenizer in local device
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # save_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_all_sid_alignment"
    save_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_sid_alignment"
    # Load model to cpu first and let torchrun handle the device placement
    model, tokenizer = load_checkpoint(base_model_name, save_dir) 
    print(f"model_device: {model.device}")
    old_vocab_size = 128_256
    print(tokenizer.eos_token)
    

    train_dataset = train_thinking.StreamingReasoningDataset(
        split="train",
        datatype="sft",
        sources="1m",
        block_size=1024
    )
    eval_dataset = train_thinking.ReasoningDataset("eval", "sft", ["1m"])
    print(f"---Eval dataset size: {len(eval_dataset)}")

    gen_eval_dataset = train_thinking.ReasoningDataset("eval", "gen_eval", ["1m"])

    # check_idx = 3
    # print(eval_dataset[check_idx])
    # print(tokenizer.decode(eval_dataset[check_idx]["input_ids"]))
    # print(tokenizer.decode([x for x in eval_dataset[check_idx]["labels"] if x != -100]))
    # it = iter(train_dataset)
    # sample = next(it)
    # print(sample.keys())  
    # print(sample)
    # print(tokenizer.decode([x for x in sample["labels"] if x != -100]))
    # print(gen_eval_dataset[0])

    SEED = 411
    GEN_EVAL_SUBSET_SIZE = 1000
    rng = random.Random(SEED)   # <- LOCAL RNG (important!)
    indices = rng.sample(range(len(eval_dataset)), GEN_EVAL_SUBSET_SIZE)
    indices = sorted(indices)   # optional but recommended
    eval_dataset = Subset(eval_dataset, indices)
    gen_eval_dataset = Subset(gen_eval_dataset, indices)

    train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, Params)
    

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()