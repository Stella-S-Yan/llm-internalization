"""
Evaluate SFT result that is trained on think data. 
Use vLLM to speed up inference. Use vLLM as a python engine, so every gpu device do its
own vllm-based inference and return local recall statistic. 
Results from devices are aggregated and print out the final global recall@k

vLLM does not work with DDP, so don't use torchrun
vLLM is used as a python kernel, and one is initiated for each gpu. To launch the script
$ python eval_think_sft.py


"""

# spawn creates a fresh Python process instead of forking.
# Each worker safely initializes CUDA independently.
# This is exactly what vLLM expects on multi-GPU setups.

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence

# Needs to import vllm before torch
from tqdm import tqdm
import config
import train_thinking
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random
import os
import argparse
import hashlib
from torch.utils.data import Subset
import math

SID_PATTERN = re.compile(r"<sid>(.*?)<")

def ddp_init():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda")
    return local_rank, device


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



def collate_fn(batch):
    return {
        "prompt": [x["prompt"] for x in batch],
        "solution": [x["solution"] for x in batch]
    }



def calculate_and_print_metrics(results_dict):
    total = len(results_dict)
    if total == 0:
        print("No samples found.")
        return

    hits_sums = [0, 0, 0] # Corresponds to k=1, 5, 10
    
    for hits in results_dict.values():
        for i, val in enumerate(hits):
            hits_sums[i] += val

    print("\n" + "="*30)
    print(f"Final Deduplicated Results (Total: {total})")
    k_list = [1, 5, 10]
    for i, k in enumerate(k_list):
        print(f"Recall@{k}: {hits_sums[i] / total:.4f}")
    print("="*30)


def main(run_num):

    local_rank, device = ddp_init()
    
    # --- assign devices via vLLM ---
    model_dir = str(config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_merged_think_sft_model_{run_num}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        # dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side='left'

    model = model.to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
  
    # --- Prepare dataset ---
    # gen_eval_dataset = train_thinking.ReasoningDataset("eval", "grpo", [config.REVIEW_TYPE])
    # SEED = 411
    # GEN_EVAL_SUBSET_SIZE = 5000
    # rng = random.Random(SEED)   # <- LOCAL RNG (important!)
    # indices = rng.sample(range(len(gen_eval_dataset)), GEN_EVAL_SUBSET_SIZE)
    # indices = sorted(indices)   # optional but recommended
    # gen_eval_dataset = Subset(gen_eval_dataset, indices)
    # print(f"Eval on {config.REVIEW_TYPE}: {len(gen_eval_dataset)}")
    # print(gen_eval_dataset[10])
    
    gen_eval_dataset = train_thinking.ReasoningDataset("test", "grpo", [config.REVIEW_TYPE])

    sampler = None
    if dist.is_initialized():
        # shuffle=False is critical for reproducibility, though the hash handles dedup regardless
        sampler = DistributedSampler(gen_eval_dataset, shuffle=False)


    eval_loader = DataLoader(
        gen_eval_dataset,
        batch_size=16,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 4. Run Evaluation
    # Returns: { "hash123": [1, 1, 1], "hash456": [0, 1, 1] ... }
    local_results = evaluate_sequence_recall(
        model=model,
        tokenizer=tokenizer,
        eval_loader=eval_loader,
        num_beams=20,
        max_new_tokens=64,
        top_k_list=[1, 5, 10],
        print_random_example=False
    )
    # print(f"---- {local_rank}: {local_results}")

    # 5. Gather & Deduplicate
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        
        # Collect dicts from all GPUs
        dist.all_gather_object(gathered_results, local_results)
        
        # Merge dicts (Deduplication happens here automatically!)
        final_results = {}
        for rank_dict in gathered_results:
            final_results.update(rank_dict)
    else:
        final_results = local_results

    # 3. Compute Metrics (Rank 0 only)
    if local_rank == 0:
        total = len(final_results)
        print(f"----- Total data count: {total}")
        ks = [1, 5, 10]  # Define your K values here
        metrics = {}

        if total > 0:
            # Extract all ranks
            all_ranks = list(final_results.values())

            for k in ks:
                # --- Recall@k ---
                # Count how many items have a rank <= k
                hits = sum(1 for r in all_ranks if r <= k)
                metrics[f"recall_{k}"] = hits / total

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

        print(metrics)

    

if __name__ == "__main__":
    # 1. Create parser
    parser = argparse.ArgumentParser(description="Example script with parameters")

    # 2. Add arguments
    parser.add_argument("--RUN_NUM", type=int, default=0, help="Run index")

    # 3. Parse arguments
    args = parser.parse_args()

    main(args.RUN_NUM)
