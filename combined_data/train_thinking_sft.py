"""
2_618_971 total training data


DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_thinking_sft.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true"


import config
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import argparse
import train_thinking
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import TrainerCallback
from torch.utils.data import Subset
from functools import partial
import math

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    ACC_STEP=1
    RUN_NUM=0
    CHECK_POINT=0


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


@torch.no_grad()
def evaluate_sequence_recall(
    model,
    tokenizer,
    eval_loader,
    num_beams=20,
    max_new_tokens=8,
    top_k_list=[1, 5, 10],
    print_random_example=True,
):
    """
    Batched sequence-level recall evaluation using beam search.
    Returns:
        local_hits: dict {k: hit_count}
        local_total: int total number of examples processed by this rank
    """
    model.eval()
    model = model.module if hasattr(model, "module") else model
    # Pick device from the model's first parameter
    device = next(model.parameters()).device
    
    # Store results as: {unique_hash: [hit_k1, hit_k2, ...]}
    local_results = {}

    max_k = max(top_k_list)
    num_return_sequences = max_k

    printed = False

    for batch in tqdm(eval_loader, desc="Evaluating"):
    # for batch in eval_loader:
        prompts = batch["prompt"]
        solutions = batch["solution"]

        # Tokenize → move to GPU
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        batch_size = len(prompts)
        prompt_len = inputs["input_ids"].size(1)

        # Beam search
        # outputs = model.module.generate(
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, num_return_sequences),
            num_return_sequences=num_return_sequences,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # (batch, num_return_sequences, seq_len)
        outputs = outputs.view(batch_size, num_return_sequences, -1)

        # Loop over batch
        for i in range(batch_size):
            # --- CREATE UNIQUE ID ---
            unique_id = solutions[i]["uid"]

            generations = [
                tokenizer.decode(
                    outputs[i, k, prompt_len:],
                    skip_special_tokens=True
                )
                for k in range(num_return_sequences)
            ]

            # Check hits
            hits = [solutions[i]["sid"] in g for g in generations]

            # Find the first index where hit is True (1-based rank)
            if True in hits:
                # .index() returns 0-based index of first True
                best_rank = hits.index(True) + 1 
            else:
                best_rank = float('inf') # Not found
            
            # Store the RANK, not the boolean list
            local_results[unique_id] = best_rank

            # ---- Print one example ----
            if print_random_example and not printed:
                print("\n=== Random Example ===")
                print(f"Prompt:\n{prompts[i]}")
                print(f"Solution:\n{solutions[i]}")
                for j in range(min(3, max_k)):
                    print(f"[Gen {j+1}] {generations[j]}")
                print("========================\n")
                printed = True


    return local_results


def no_processing_collator(batch):
    return {
        "prompt": [x["prompt"] for x in batch],
        "solution": [x["solution"] for x in batch]
    }


class GenerateEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tokenizer, eval_fn, eval_steps):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.batch_size = 16
        self.best_metric = None  # Track best metric

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
                num_workers=4,
                sampler=sampler,
                shuffle=False,
                collate_fn=no_processing_collator,
            )

            # tqdm only on rank 0
            if rank == 0:
                eval_loader = tqdm(
                    eval_loader,
                    desc=f"Eval @ step {state.global_step}",
                )

            # ---- Custom generate-based eval ----
            local_results = self.eval_fn(
                    self.trainer.model,
                    self.tokenizer,
                    eval_loader,
                    num_beams=20, 
                    max_new_tokens=64,
                    top_k_list=[1, 5, 10],
                    print_random_example=False
                )
            device = self.trainer.model.device

            # 2. Gather & Deduplicate (The "Merge" Step)
            if is_ddp:
                world_size = dist.get_world_size()
                gathered_data = [None for _ in range(world_size)]
                # all_gather_object serializes the dict and sends it to all ranks
                dist.all_gather_object(gathered_data, local_results)
                
                # Merge dicts (Deduplication happens here automatically!)
                final_results = {}
                for rank_dict in gathered_data:
                    final_results.update(rank_dict)
            else:
                final_results = local_results


            # 3. Compute Metrics (Rank 0 only)
            if rank == 0:
                total = len(final_results)
                ks = [1, 5, 10]  # Define your K values here
                metrics = {}

                if total > 0:
                    # Extract all ranks
                    all_ranks = list(final_results.values())

                    for k in ks:
                        # --- Recall@k ---
                        # Count how many items have a rank <= k
                        hits = sum(1 for r in all_ranks if r <= k)
                        metrics[f"eval/recall_{k}"] = hits / total

                        # --- NDCG@k ---
                        # Sum(1 / log2(rank + 1)) for ranks <= k
                        dcg = sum(1.0 / math.log2(r + 1) for r in all_ranks if r <= k)
                        
                        # IDCG is ideal case: we assume 1 relevant item per query, so IDCG@k = 1.0
                        # Thus NDCG = DCG / 1.0
                        metrics[f"eval/ndcg_{k}"] = dcg / total
                else:
                    # Handle empty dataset case
                    for k in ks:
                        metrics[f"eval/recall_{k}"] = 0.0
                        metrics[f"eval/ndcg_{k}"] = 0.0

                metrics["step"] = state.global_step

                # 4. Log
                self.trainer.log(metrics)

                print(f"\n[Custom eval @ step {state.global_step}] (N={total})")
                for k, v in metrics.items():
                    if "recall" in k or "ndcg" in k:
                        print(f"  {k}: {v:.4f}")

            return control


def train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, params):
    print(f"@@@ total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    MODEL_SAVE_DIR = config.MODEL_DIR / f"{config.DATA_SOURCE}_think_sft_adaptor_{Params.RUN_NUM}"
    NUM_WORKERS = 1

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
        logging_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        greater_is_better=False,
        save_total_limit=10,
        load_best_model_at_end=False,
        eval_strategy="steps",
        eval_steps=2000,
        optim="adamw_torch",
        bf16=True,          # <<< enable bfloat16 (H100 optimized)
        fp16=False,         # optional: if you want fp16 instead
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False
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

    # model = torch.compile(model, mode="max-autotune")
    peft_model = get_peft_model(model, lora_config)

    # Freeze all base model parameters (done automatically by get_peft_model)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    collator_fn = partial(train_thinking.sft_data_collator, tokenizer=tokenizer)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator_fn
    )

    callback = GenerateEvalCallback(
        trainer=trainer,
        eval_dataset=gen_eval_dataset,
        tokenizer=tokenizer,
        eval_fn=evaluate_sequence_recall,
        eval_steps=2000,
    )
    trainer.add_callback(callback)

    if params.CHECK_POINT == 0:
        trainer.train()
    else:
        print(f"... Continue training from {params.CHECK_POINT} on node {params.RUN_NUM}")
        trainer.train(resume_from_checkpoint=f"/usr/local/google/home/stellasyan/Documents/llm_internalization/data/model/Amazon_think_sft_adaptor_{str(params.RUN_NUM)}/checkpoint-{str(params.CHECK_POINT)}")


def main():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=32, help="Training batch size")
    parser.add_argument("--LORA_RANK", type=int, default=16, help="Rank of LoRA adaptor")
    parser.add_argument("--LORA_RATIO", type=float, default=0.1, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=20000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")
    parser.add_argument("--ADAPTOR_SAVE_DIR", type=str, default='think_sft_adaptor', help="Where to save the trained adaptor")
    parser.add_argument("--ACC_STEP", type=int, default=1, help="Gradient accumulate steps")
    parser.add_argument("--RUN_NUM", type=int, default=0, help="Run index")
    parser.add_argument("--CHECK_POINT", type=int, default=0, help="Checkpoint number")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_rank{Params.LORA_RANK}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}_acc{Params.ACC_STEP}_{Params.RUN_NUM}"
    Params.LOGGING_DIR =  config.RUN_DIR / f"{config.DATA_SOURCE}_Combined_train_thinking_sft" / run_name

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # Load model and tokenizer in local device
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    save_dir = MODEL_SAVE_DIR = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_all_sid_alignment"
    # Load model to cpu first and let torchrun handle the device placement
    model, tokenizer = load_checkpoint(base_model_name, save_dir) 
    print(f"model_device: {model.device}")
    old_vocab_size = 128_256
    print(tokenizer.eos_token)
    
    train_dataset = train_thinking.ReasoningDataset("train", "sft", ["Toys_and_Games", "Sports_and_Outdoors", "Beauty"])
    eval_dataset = train_thinking.ReasoningDataset("eval", "sft", ["Toys_and_Games"])
    gen_eval_dataset = train_thinking.ReasoningDataset("eval", "grpo", ["Toys_and_Games"])
    
    SEED = 411
    GEN_EVAL_SUBSET_SIZE = 5000
    rng = random.Random(SEED)   # <- LOCAL RNG (important!)
    
    indices = rng.sample(range(len(gen_eval_dataset)), GEN_EVAL_SUBSET_SIZE)
    indices = sorted(indices)   # optional but recommended
    gen_eval_dataset = Subset(gen_eval_dataset, indices)
    print(f"---Eval gen dataset size: {len(eval_dataset)}")

    indices = rng.sample(range(len(eval_dataset)), GEN_EVAL_SUBSET_SIZE)
    indices = sorted(indices)   # optional but recommended
    eval_dataset = Subset(eval_dataset, indices)
    print(f"---Eval dataset size: {len(eval_dataset)}")

    train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, Params)

    
if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()