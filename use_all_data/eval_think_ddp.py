import torch
from tqdm import tqdm
import numpy as np
import config
from utils import merge_save_model
from use_all_data import train_thinking
from torch.utils.data import DataLoader, Subset, DistributedSampler
import re
import argparse
import json
import os
import argparse
import torch.distributed as dist

"""
# running DDP evaluation on 8 GPUs
$ torchrun --nproc_per_node=8 eval_think_ddp.py
$ torchrun --nproc_per_node=8 eval_think_ddp.py --batch_size 16

# merging results after evaluation
$ python eval_think_ddp.py --merge_results

# vanila reasoning, no difference, just as next token
{1: np.float64(0.015379113018597998), 5: np.float64(0.04542203147353362), 10: np.float64(0.06223175965665236)}
"""




@torch.no_grad()
def evaluate_sequence_recall(
    model,
    tokenizer,
    eval_loader,
    device,
    num_beams=20,
    max_new_tokens=128,
    top_k_list=[1, 5, 10],
):
    model.eval()
    model.to(device).half() # use FP16 for speed

    hits_dict = {k: [] for k in top_k_list}
    sid_pattern = re.compile(r"<sid>(.*?)</sid>")

    for batch in tqdm(eval_loader, desc="Evaluating"):
        prompts = batch["prompt"]
        targets = batch["target"]

        batch_target_sids = [
            (m.group(1).strip() if (m := sid_pattern.search(t)) else None)
            for t in targets
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        batch_size = len(prompts)
        max_k = max(top_k_list)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, max(top_k_list)),
            num_return_sequences=max(top_k_list),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        batch_outputs = outputs.view(batch_size, max(top_k_list), -1)

        for i in range(batch_size):
            prompt_len = inputs["input_ids"].size(1)
            decoded_outputs = [
                tokenizer.decode(batch_outputs[i, k, prompt_len:], skip_special_tokens=True)
                for k in range(max_k)
            ]

            pred_sids = [
                (m.group(1).strip() if (m := sid_pattern.search(t)) else None)
                for t in decoded_outputs
            ]

            # Detect missing sid in generation
            for k, sid in enumerate(pred_sids):
                if sid is None:  
                    print("\n================= MISSING <sid> FOUND =================")
                    print("PROMPT:\n", prompts[i])
                    print("\nFULL GENERATED OUTPUT:")
                    print(decoded_outputs[k])
                    print("========================================================\n")

            # hits = [1 if batch_target_sids[i] in o else 0 for o in pred_sids]
            hits = [
                1 if (o is not None and batch_target_sids[i] is not None and batch_target_sids[i] in o)
                else 0
                for o in pred_sids
            ]

            for k in top_k_list:
                hits_dict[k].append(int(any(hits[:k])))

    # Save raw hits and number of samples
    results = {"top_k_hits": hits_dict, "num_samples": sum(len(h) for h in hits_dict.values()) // len(top_k_list)}  
    return results

# Merge all split results
def merge_splits(num_splits=8):
    top_k_list=[1, 5, 10]
    all_hits = {k: [] for k in top_k_list}
    total_samples = 0

    for rank in range(num_splits):
        with open(f"/usr/local/google/home/stellasyan/Documents/llm_internalization/use_all_data/eval_results/recall_split_{rank}.json") as f:
            data = json.load(f)
            for k in top_k_list:
                all_hits[k].extend(data["top_k_hits"][str(k)])
            total_samples += data["num_samples"]

    # Compute final recall@k
    final_recall = {k: np.mean(all_hits[k]) for k in top_k_list}
    print(f"Recall: {final_recall}")
    return final_recall


def main():
    # Initialize process group for DDP
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_results", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"world_size: {world_size}")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    if args.merge_results:
        merge_splits(num_splits=world_size)
        return

    # Assign GPU using LOCAL_RANK (safe for multi-node)
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    print(f"local_rank: {LOCAL_RANK}")
    device = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer
    model_input_dir = config.MODEL_DIR / "think_model_best"
    model, tokenizer = merge_save_model.load_merged_model(model_input_dir)

    # Load evaluation dataset
    eval_dataset = train_thinking.SeqReasoningDataset(tokenizer, "eval")
    # eval_dataset = Subset(eval_dataset, range(32*8))

    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(eval_dataset, shuffle=False) if world_size > 1 else None

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,  # helps faster transfer to GPU
        shuffle=(sampler is None),
        collate_fn=None
    )

    # Wrap DataLoader in tqdm ONLY for rank 0
    if rank == 0:
        eval_loader = tqdm(eval_loader, desc="Evaluating")

    # Run evaluation
    recalls = evaluate_sequence_recall(
        model=model,
        tokenizer=tokenizer,
        eval_loader=eval_loader,
        device=device,
        num_beams=20,
        max_new_tokens=392,
        top_k_list=[1, 5, 10]
    )

    # Optionally, save per-rank results
    os.makedirs("eval_results", exist_ok=True)
    with open(f"eval_results/recall_split_{rank}.json", "w") as f:
        json.dump(recalls, f, indent=2)

    if rank == 0:
        print("Evaluation complete. Merge results using --merge_results")

if __name__ == "__main__":
    main()
    # merge_splits()
