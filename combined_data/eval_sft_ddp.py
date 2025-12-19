
"""
torchrun --nproc_per_node=8 sft_eval_model_ddp.py
"""

import torch
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import train_seq_pred_aligned_phase1
import os
from torch.nn.parallel import DistributedDataParallel as DDP


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
        targets = batch["target"]
        

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

            # Check hits for each k

            # First 3 levels
            # first3= " ".join(targets[i].split()[:3])
            # hits = [1 if first3 in g else 0 for g in generations]

            # All levels
            hits = [1 if targets[i] in g else 0 for g in generations]

            for k in top_k_list:
                # if any of the first k generations hits
                local_hits[k] += int(any(hits[:k]))

        # Print only once
        if print_random_example and not printed:
            idx = random.randint(0, batch_size - 1)
            print("\n=== Random Example ===")
            print(f"Prompt:\n{prompts[idx]}")
            print(f"Target:\n{targets[idx]}")
            for k in range(min(5, num_return_sequences)):
                print(f"[Gen {k+1}] {generations[k]}")
            print("========================\n")
            printed = True

        local_total += batch_size

    return local_hits, local_total


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


local_rank, device = ddp_init()

model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_merged_seq_pred_model"

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    # dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.padding_side='left'

model = model.to(device)

if dist.is_initialized():
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

gen_eval_dataset = train_seq_pred_aligned_phase1.SeqGenDataset("eval", [config.REVIEW_TYPE])
print(f"Eval on {config.REVIEW_TYPE}: {len(gen_eval_dataset)}")
# gen_eval_dataset = train_seq_pred_aligned_phase1.SeqGenDataset("test")
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
)


local_hits, local_total = evaluate_sequence_recall(
    model=model,
    tokenizer=tokenizer,
    eval_loader=eval_loader,
    num_beams=20,
    max_new_tokens=7,
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