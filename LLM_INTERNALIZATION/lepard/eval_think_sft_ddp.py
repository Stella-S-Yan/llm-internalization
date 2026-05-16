"""
Evaluate SFT result that is trained on think data. Make sure metrics computation is valid in DDP setup.
"""


import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse

# Needs to import vllm before torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import math

from LLM_INTERNALIZATION import config
from LLM_INTERNALIZATION.lepard import reasoning_data


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
    
    # Store results as: {unique_id: [hit_k1, hit_k2, ...]}
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
        prompt_lens = inputs["attention_mask"].sum(dim=1)

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
            unique_id = solutions[i]["row_id"]
            cur_prompt_len = prompt_lens[i].item()
            generations = [
                tokenizer.decode(
                    outputs[i, k, cur_prompt_len:],
                    skip_special_tokens=True
                )
                for k in range(num_return_sequences)
            ]

            # Check hits
            hits = [solutions[i]["ssid"] in g for g in generations]

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


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def collate_fn(batch):
    return {
        "prompt": [x["prompt"] for x in batch],
        "solution": [x["solution"] for x in batch]
    }


class Params:
    DATA_TYPE = "10k"


def main(run_num, data_type):
    
    local_rank, device = ddp_init()
    
    # --- assign devices via vLLM ---
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_merged_think_sft_model_{run_num}"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side='left'

    model = model.to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
  
    # --- Prepare dataset ---
    print(f"---- Eval on {data_type} dataset ----")
    gen_eval_dataset = reasoning_data.LepardDataset('grpo', tokenizer, "test")
    # gen_eval_dataset = Subset(gen_eval_dataset, range(500))
    
    print(f"Eval on : {len(gen_eval_dataset)} data points.")
    # eval_dataset = Subset(eval_dataset, range(32*8))
    print(gen_eval_dataset[0])

    
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
        max_new_tokens=128,
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
    parser.add_argument("--DATA_TYPE", type=str, default="20k", help="Lepard evaluation datatype")

    # 3. Parse arguments
    args = parser.parse_args()

    main(args.RUN_NUM, args.DATA_TYPE)
