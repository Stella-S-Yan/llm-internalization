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
from combined_data import train_thinking
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random
import os
import argparse

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
    
    # We now store only raw hits, not full lists
    local_hits = {k: 0 for k in top_k_list}
    local_total = 0

    max_k = max(top_k_list)
    num_return_sequences = max_k

    printed = False

    for batch in tqdm(eval_loader, desc="Evaluating"):
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
            generations = [
                tokenizer.decode(
                    outputs[i, k, prompt_len:],
                    skip_special_tokens=True
                )
                for k in range(num_return_sequences)
            ]

            # Extract <sid> from generated outputs
            # pred_sids = [
            #     (m.group(1).strip() if (m := SID_PATTERN.search(t)) else None)
            #     for t in generations
            # ]

            # All levels
            hits = [solutions[i]["sid"] in g for g in generations]

            for k in top_k_list:
                # if any of the first k generations hits
                local_hits[k] += int(any(hits[:k]))

        # Print only once
        if print_random_example and not printed:
            idx = random.randint(0, batch_size - 1)
            print("\n=== Random Example ===")
            print(f"Prompt:\n{prompts[idx]}")
            print(f"Solution:\n{solutions[idx]}")
            for k in range(min(5, num_return_sequences)):
                print(f"[Gen {k+1}] {generations[k]}")
            print("========================\n")
            printed = True

        local_total += batch_size

    return local_hits, local_total

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def collate_fn(batch):
    return {
        "prompt": [x["prompt"] for x in batch],
        "solution": [x["solution"] for x in batch]
    }

    

def main(run_num):

    local_rank, device = ddp_init()
    
    # --- assign devices via vLLM ---
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_merged_think_sft_model_{run_num}"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        # dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    tokenizer.padding_side='left'

    model = model.to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
  
    # --- Prepare dataset ---
    gen_eval_dataset = train_thinking.ReasoningDataset("eval", "grpo", [config.REVIEW_TYPE])
    print(f"Eval on {config.REVIEW_TYPE}: {len(gen_eval_dataset)}")
    # eval_dataset = Subset(eval_dataset, range(32*8))
    print(gen_eval_dataset[0])

    
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        rank = torch.distributed.get_rank()
        # print("Rank: ", rank)
        sampler = DistributedSampler(gen_eval_dataset, shuffle=False)
    else:
        rank = 0
        sampler = None


    batch_size = 64   # 64 may be too large for beam search


    eval_loader = DataLoader(
        gen_eval_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn
    )


    local_hits, local_total = evaluate_sequence_recall(
        model=model,
        tokenizer=tokenizer,
        eval_loader=eval_loader,
        num_beams=20,
        max_new_tokens=128,
        top_k_list=[1, 5, 10],
    )

    # -----------------------------
    # Aggregate across ranks
    # -----------------------------
    device = next(model.parameters()).device

    hits_tensor = torch.tensor([local_hits[k] for k in [1,5,10]], device=device, dtype=torch.float32)
    total_tensor = torch.tensor([local_total], device=device, dtype=torch.float32)

    if dist.is_initialized():
        dist.all_reduce(hits_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    # -----------------------------
    # Compute global recall
    # -----------------------------
    if not dist.is_initialized() or dist.get_rank() == 0:
        global_recalls = {
            f"recall_{k}": (hits_tensor[i] / total_tensor.item()).item()
            for i, k in enumerate([1,5,10])
        }
        print(global_recalls)



if __name__ == "__main__":
    # 1. Create parser
    parser = argparse.ArgumentParser(description="Example script with parameters")

    # 2. Add arguments
    parser.add_argument("--RUN_NUM", type=int, default=0, help="Run index")

    # 3. Parse arguments
    args = parser.parse_args()

    main(args.RUN_NUM)
