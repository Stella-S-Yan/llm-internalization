"""
Evaluate SFT result that is trained on think data. 
Use vLLM to speed up inference. Use vLLM as a python engine, so every gpu device do its
own vllm-based inference and return local recall statistic. 
Results from devices are aggregated and print out the final global recall@k

vLLM does not work with DDP, so don't use torchrun
vLLM is used as a python kernel, and one is initiated for each gpu. To launch the script
$ python eval_think_sft.py

With Reasoning SFT, achieves this result
Beauty:
Global Recall: {1: np.float64(0.021777042436166884), 5: np.float64(0.05330232974109019), 10: np.float64(0.0727541027590216)}
Global Recall: {1: np.float64(0.02164289227742253), 5: np.float64(0.05213969503197245), 10: np.float64(0.0724858024415329)}

Toy:
Global Recall: {1: np.float64(0.01859674428188749), 5: np.float64(0.04435400783020812), 10: np.float64(0.06192046157016279)}
sample 1024
Global Recall: {1: np.float64(0.017578125), 5: np.float64(0.052734375), 10: np.float64(0.072265625)}
400
Global Recall: {1: np.float64(0.015), 5: np.float64(0.055), 10: np.float64(0.0675)}
"""

# spawn creates a fresh Python process instead of forking.
# Each worker safely initializes CUDA independently.
# This is exactly what vLLM expects on multi-GPU setups.

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# Needs to import vllm before torch
from vllm import LLM
from vllm.sampling_params import BeamSearchParams
import torch
from tqdm import tqdm
import config
from use_all_data import train_thinking
from torch.utils.data import DataLoader, Subset
import re
import numpy as np
import random



@torch.no_grad()
def evaluate_sequence_recall_VLLM(
    llm,
    eval_loader,
    num_beams=20,
    max_new_tokens=128,
    top_k_list=[1, 5, 10],
):
    hits_dict = {k: [] for k in top_k_list}
    # sid_pattern = re.compile(r"<sid>(.*?)</sid>")
    sid_pattern = re.compile(r"<sid>(.*?)<")

    for batch in tqdm(eval_loader, desc="Evaluating"):
        prompts = batch["prompt"]
        targets = batch["target"]

        batch_target_sids = [
            (m.group(1).strip() if (m := sid_pattern.search(t)) else None)
            for t in targets
        ]

        batch_size = len(prompts)
        max_k = max(top_k_list)

        # vLLM sampliing parameters
        beam_params = BeamSearchParams(
            beam_width=num_beams,
            max_tokens=max_new_tokens,
            ignore_eos=False,
        )

        # --- Generate outputs ---
        batch_outputs = llm.beam_search(
            prompts=prompts,
            params=beam_params,
            use_tqdm=False
        )

        # batch_outputs[i].outputs[j].text  -> i-th prompt, j-th beam
        for i in range(batch_size):
            decoded_outputs = [
                batch_outputs[i].sequences[k].text
                for k in range(max_k)
            ]

            # Extract <sid> from generated outputs
            pred_sids = [
                (m.group(1).strip() if (m := sid_pattern.search(t)) else None)
                for t in decoded_outputs
            ]

            # Detect missing <sid>
            for k, sid in enumerate(pred_sids):
                if sid is None:
                    print("\n================= MISSING <sid> FOUND =================")
                    print("PROMPT:\n", prompts[i])
                    print("\nFULL GENERATED OUTPUT:")
                    print(decoded_outputs[k])
                    print("========================================================\n")

            # Compute hits
            hits = [
                1 if (o is not None and batch_target_sids[i] is not None and batch_target_sids[i] in o)
                else 0
                for o in pred_sids
            ]

            for k in top_k_list:
                hits_dict[k].append(int(any(hits[:k])))

    # Save raw hits and number of samples
    results = {
        "top_k_hits": hits_dict,
        "num_samples": sum(len(h) for h in hits_dict.values()) // len(top_k_list)
    }
    return results


def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    targets = [item["target"] for item in batch]

    return {
        "prompt": prompts,
        "target": targets
    }


def main():
    # --- assign devices via vLLM ---
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_merged_think_sft_model"

    # --- Load vLLM engine on all GPUs ---
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        tensor_parallel_size=8,     # use all 8 GPUs
        gpu_memory_utilization=0.90,
        dtype="float32"  # use float32 temporarily
    )

    # --- Prepare dataset ---
    eval_dataset = train_thinking.ReasoningDataset("eval", "raw_text_vllm")
    # eval_dataset = Subset(eval_dataset, range(32*8))
    print(eval_dataset[0])

    # ---- Use random samples to reduce evaluation time ----
    num_samples = 400 #1024
    total = len(eval_dataset)
    subset_indices = random.sample(range(total), num_samples)
    eval_dataset = Subset(eval_dataset, subset_indices)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Wrap DataLoader in tqdm
    eval_loader = tqdm(eval_loader, desc="Evaluating")

    # --- Run evaluation ---
    recalls_local = evaluate_sequence_recall_VLLM(
        llm=llm,
        eval_loader=eval_loader,
        num_beams=20,
        max_new_tokens=206,
        top_k_list=[1, 5, 10]
    )

    # --- Compute final recall ---
    global_hits = {}
    for k in [1, 5, 10]:
        global_hits[k] = np.array(recalls_local["top_k_hits"][k])
    global_recall = {k: global_hits[k].mean() for k in [1, 5, 10]}
    
    print("Global Recall:", global_recall)




if __name__ == "__main__":
    main()
