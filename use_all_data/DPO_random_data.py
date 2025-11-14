"""
Generate data for DPO training

Target vs random semantid ID

$ torchrun --nproc_per_node=8 DPO_random_data.py
"""

import random
import torch
from use_all_data import save_full_model_eval, train_seq_pred_aligned_phase1
from torch.utils.data import DataLoader, DistributedSampler
import os
from tqdm import tqdm
import config
import json
import torch.distributed as dist


def random_semantic_id():
    return " ".join(f"{c}{random.randint(0,255)}" for c in ["A","B","C","D"])


def get_data(
    model,
    data_loader,
    k=1,  # number of rejected per prompt
    device="cuda",
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

            batch_size = len(prompts)

            # build preference pairs
            for i in range(batch_size):
                target = targets[i]

                for _ in range(k):
                    # ensure rejected != target
                    while True:
                        rejected = random_semantic_id()
                        if rejected != target:
                            break

                    item = {
                        "prompt": prompts[i],
                        "chosen": target,
                        "rejected": rejected
                    }
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

    model, tokenizer = save_full_model_eval.load_model()
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


    results = get_data(model, train_loader, k=3, device=device, save_path=config.PROCESSED_DATA_DIR / f"dpo_{split}_{rank}.jsonl")

    # if world_size > 1:
    #     torch.distributed.barrier()  # ensure all ranks done
    #     results = gather_results(results, rank, world_size)

    # if rank == 0:
    #     from datasets import Dataset
    #     Dataset.from_list(results).to_json(config.PROCESSED_DATA_DIR / f"{split}_dpo.jsonl")
            


if __name__ == "__main__":
    main(split="train", batch_size=256)
    
