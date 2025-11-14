"""
Generate data for DPO training

Prepare explicit prompts in standard format dataset for TRL DPO.

$ torchrun --nproc_per_node=8 DPO_data.py
"""

import random
import torch
from use_all_data import eval_model, train_seq_pred_aligned_phase1
from torch.utils.data import DataLoader, DistributedSampler
import os
from tqdm import tqdm
import bagz
import config
from collections import defaultdict
import json
from itertools import combinations
import torch.distributed as dist
from torch.utils.data import Subset
import numpy as np



def prefix_match_score(target: str, candidate: str, base=2):
    """
    Returns an exponentially scaled score for prefix match length.
    If match_len = 0, score = 0
    If match_len = 1, score = 1
    If match_len = 2, score = 3
    If match_len = 3, score = 7
    """
    target_tokens = target.split()
    candidate_tokens = candidate.split()
    match_len = 0
    for t, c in zip(target_tokens, candidate_tokens):
        if t == c:
            match_len += 1
        else:
            break
    # Exponential scaling
    return base ** match_len - 1


def build_prefix_index(strings, k):
    prefix_index = defaultdict(list)
    for s in strings:
        tokens = s.split()
        for i in range(1, k+1):
            prefix = tuple(tokens[:i])  # tuple is hashable
            prefix_index[prefix].append(s)
    return prefix_index


def get_data(
    model,
    tokenizer,
    data_loader,
    device="cuda",
    base=2,
    max_new_tokens=8,
    num_return_sequences=5,
    save_path=None,
    flush_every=1000,
):
    model.eval()
    results = []

    if save_path:
        f = open(save_path, "a", encoding="utf-8")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating DPO Data"):
            prompts = batch["prompt"]
            targets = batch["target"]

            # tokenize and move to device
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            batch_size = len(prompts)

            # generate with sampling
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # reshape: [batch, num_return_sequences, seq_len]
            outputs = outputs.view(batch_size, num_return_sequences, -1)
            prompt_len = inputs["input_ids"].size(1)

            # slice off prompt tokens
            decoded_outputs = outputs[:, :, prompt_len:]  # [batch, num_return_sequences, seq_len - prompt_len]
            new_seq_len = decoded_outputs.size(-1)

            # flatten batch and num_return_sequences
            decoded_outputs_flat = decoded_outputs.reshape(-1, new_seq_len)

            # batch decode all generated sequences at once
            decoded_all = tokenizer.batch_decode(
                decoded_outputs_flat,
                skip_special_tokens=True
            )
            decoded_all = [d.strip() for d in decoded_all]
            decoded_all = np.array(decoded_all).reshape(batch_size, num_return_sequences)

            # build preference pairs
            for i in range(batch_size):
                seqs = decoded_all[i]
                target = targets[i]

                # compute scores once
                scores = [prefix_match_score(target, s, base) for s in seqs]

                if len(set(scores)) > 1:
                    # find best and worst
                    best_idx = np.argmax(scores)
                    worst_idx = np.argmin(scores)
                    chosen = seqs[best_idx]
                    rejected = seqs[worst_idx]
                else:
                    # fallback: use target vs random negative
                    chosen = target
                    rejected = random.choice(seqs)

                item = {"prompt": prompts[i], "chosen": target, "rejected": str(rejected)}
                results.append(item)

                # periodically flush to disk
                if save_path and len(results) >= flush_every:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    results.clear()

    # final flush
    if save_path and results:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        f.close()

    return results



def gen_candidates():
    train_reader = bagz.Reader(config.TRAIN_DATA)
    data = [json.loads(record.decode()) for record in train_reader]
    candidates = [x["target"] for x in data]

    prefix_idx = build_prefix_index(candidates, 2)  # prefix upto two tokens
    return prefix_idx


def gather_results(results, rank, world_size):
    """
    Gathers Python lists of results (DPO triplets) from all ranks to rank 0.
    Uses torch.distributed.all_gather_object (no pickle serialization needed).
    """
    if world_size == 1:
        return results

    # Create list to hold gathered results from all ranks
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, results)

    if rank == 0:
        # Flatten list of lists
        merged = []
        for part in gathered:
            merged.extend(part)
        return merged
    else:
        return None



def main(split="train", batch_size=8):

    # Initialize distributed process group if launched with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="gloo")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    model, tokenizer = eval_model.load_model()
    model.eval()
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    # Load dataset
    prompt_dataset = train_seq_pred_aligned_phase1.SeqGenDataset(split)

    # Distributed sampler splits data per rank
    sampler = DistributedSampler(prompt_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        prompt_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is None else None,
        collate_fn=None,
        num_workers=4,  # adjust
    )


    results = get_data(model, tokenizer, train_loader, device=device, save_path=config.PROCESSED_DATA_DIR / f"dpo_{split}_{rank}.jsonl")

    # if world_size > 1:
    #     torch.distributed.barrier()  # ensure all ranks done
    #     results = gather_results(results, rank, world_size)

    # if rank == 0:
    #     from datasets import Dataset
    #     Dataset.from_list(results).to_json(config.PROCESSED_DATA_DIR / f"{split}_dpo.jsonl")
            


if __name__ == "__main__":
    main(split="train", batch_size=256)
    
