"""
Phase 1 training for seq pred. Use aligned new embeddings; fix all embeddings; only tune LoRA parameter.

Able to achieve 4.97% recall@5



DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py
"""

import config
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainerCallback
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import math
from torch.optim.lr_scheduler import LambdaLR
import os
from utils import bagz_utils


class ReasoningDataset(Dataset):
    def __init__(self, split, datatype: str, sources):
        self.datatype = datatype

        self.data = []
        for src in sources:
            data_path = os.path.join(config.PROCESSED_DATA_DIR / f'{config.DATA_SOURCE}_{src}_think_data_{split}.bagz')
            self.data.extend(bagz_utils.read_record(data_path))

        # === CRITICAL FIX: FORCE DETERMINISTIC ORDER ===
        # DDP requires self.data to be identical (index-for-index) on every GPU.
        self.data.sort(key=lambda x: x["solution"]["uid"])


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        record = self.data[idx]
        if self.datatype == "sft":
            return {
                # "input_ids": record["input_ids"],
                # "labels": record["labels"],
                "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
                "labels": torch.tensor(record["labels"], dtype=torch.long)
            }
        elif self.datatype == "grpo":
            return {
                "prompt": record["prompt"],
                "solution": record["solution"],
            }
        elif self.datatype == "raw_text_vllm":  # used for vLLM-based thinking_sft model evaluation
            return {
                "prompt": {"prompt": record["prompt"], "prompt_token_ids": record["prompt_token_ids"].tolist()},
                "target": record["target"],
                "solution": record["solution"],
            }
        elif self.datatype == "raw_text":
            return {
                "prompt_token_ids": record["prompt_token_ids"],
                "target": record["target"],
            }
        elif self.datatype == "gen_eval":
            return {
                "gen_prompt": record["prompt"],
                "gen_target": record["target"]
            }
        else:
            raise ValueError(
                f"Invalid datatype '{self.datatype}'. "
                f"Expected one of: ['sft', 'grpo', 'raw_text', 'raw_text_vllm']"
            )
        

def gen_eval_collator(batch):
    """
    Collator for datasets where all fields are strings.
    Returns the batch as a list of dicts.
    """
    return batch

def sft_data_collator(batch, tokenizer):
    """
    Pads variable-length input_ids and labels in a batch.
    - input_ids padded with tokenizer.pad_token_id
    - labels padded with -100 (so prompts are ignored)
    Returns attention_mask automatically.
    """
    # Convert each input/label to a torch tensor
    # input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
    # labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

    input_ids = [f["input_ids"] for f in batch]
    labels = [f["labels"] for f in batch]

    # pad sequences to the max length in the batch
    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="left"
    )

    labels = pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=-100, 
        padding_side="left"
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


@torch.no_grad()
def evaluate_sequence_recall(
    model,
    tokenizer,
    eval_loader,
    num_beams=20,
    max_new_tokens=2, # only for faster evaluation since only does 1 token decoding, can set to 8,   
    top_k_list=[1, 5, 10],
    print_random_example=True,
):
    model.eval()
    device = model.device

    recalls_dict = {k: [] for k in top_k_list}
    printed = False  
    max_k = max(top_k_list)

    for batch in tqdm(eval_loader, desc="Evaluating"):
        prompts = batch["prompt"]
        targets = batch["target"]
        batch_size = len(prompts)

        # Extract clean target token (strip EOS)
        target_clean = [t.split()[0] for t in targets]   # list length B

        # Tokenize prompt batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # Beam generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, max_k),
            num_return_sequences=max_k,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # (B, K, L)
        outputs = outputs.view(batch_size, max_k, -1)

        # Slice off prompt
        gen = outputs[:, :, prompt_len:]  # shape [B,K,L]
        L = gen.size(-1)

        # EOS mask
        eos_mask = (gen == tokenizer.eos_token_id)  # [B,K,L]

        # First EOS pos (or L if none)
        # Produce index vector [B,K]
        has_eos = eos_mask.any(dim=-1)
        eos_pos = torch.where(
            has_eos,
            eos_mask.float().argmax(dim=-1),
            torch.full((batch_size, max_k), L - 1, dtype=torch.long, device=device),
        )

        # ---- Vectorized slicing using gather ----
        # Build arange index [L]
        idx = torch.arange(L, device=device).view(1,1,L)  # [1,1,L]

        # mask positions beyond eos_pos
        mask = idx <= eos_pos.unsqueeze(-1)   # [B,K,L]
        # replace beyond-eos with pad so decoding truncates naturally
        truncated = torch.where(mask, gen, tokenizer.pad_token_id)

        # Flatten for batch decoding
        flat = truncated.reshape(batch_size * max_k, L)

        decoded = tokenizer.batch_decode(flat, skip_special_tokens=True)
        decoded = [d.strip() for d in decoded]

        # Reshape back to [B][K]
        decoded = [
            decoded[i * max_k : (i + 1) * max_k]
            for i in range(batch_size)
        ]

        # ---- Vectorized hit checking ----
        # Compare first token only (semantic ID)
        for i in range(batch_size):
            cand_first_tokens = [d.split()[0] for d in decoded[i] if d.split()]     # skip blank strings
            hit_flags = [int(target_clean[i] == tok) for tok in cand_first_tokens]
            for k in top_k_list:
                recalls_dict[k].append(int(any(hit_flags[:k])))

        # Print one example
        if print_random_example and not printed:
            ri = np.random.randint(batch_size)
            print("\n=== Random Example ===")
            print("Prompt:", prompts[ri])
            print("Target:", target_clean[ri])
            for k in range(min(5, max_k)):
                print(f"[Gen {k+1}] {decoded[ri][k]}")
            print("==========================\n")
            printed = True

    return {f"recall_{k}": float(np.mean(v)) for k, v in recalls_dict.items()}


class GenerateEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tokenizer, eval_fn, eval_steps=1000):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.batch_size = 64    # 16
        self.best_metric = None  # Track best metric

    def on_step_end(self, args, state, control, **kwargs):

        # dynamically adjust evaluation frequency
        if 5000 <= state.global_step:
            eval_interval = 1000
        else:
            eval_interval = self.eval_steps

        # Run every eval_steps
        if state.global_step > 0 and state.global_step % eval_interval == 0:

            # If running in DDP, shard the dataset using DistributedSampler
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                rank = torch.distributed.get_rank()
                # print("Rank: ", rank)
                sampler = DistributedSampler(self.eval_dataset, shuffle=False)
            else:
                rank = 0
                sampler = None

            # Create 
            # small_dataset = Subset(self.eval_dataset, range(10))
            eval_loader = DataLoader(
                self.eval_dataset,
                # small_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                collate_fn=None  # or custom collate_fn if needed
            )

            # Wrap DataLoader in tqdm ONLY for rank 0
            if rank == 0:
                eval_loader = tqdm(eval_loader, desc=f"Evaluating step {state.global_step}")


            # Run your custom generate-based eval
            metrics = self.eval_fn(self.trainer.model, self.tokenizer, eval_loader)

            # If DDP, reduce metrics across processes
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                for k in metrics:
                    tensor = torch.tensor(metrics[k], device=self.trainer.model.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    metrics[k] = (tensor / torch.distributed.get_world_size()).item()

            # Prefix metrics for consistency with Trainer logs
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
            metrics["step"] = state.global_step

            # Log to TensorBoard / WandB / etc.
            self.trainer.log(metrics)

            # Also print for visibility
            if rank == 0:
                print(f"\n[Custom generate eval @ step {state.global_step}] {metrics}")

                # Save best model 
                current_metric = metrics["eval_recall_5"]  
                if (self.best_metric is None) or (current_metric > self.best_metric):
                    self.best_metric = current_metric
                    print(f"New best metric {current_metric:.4f}! Saving model...")
                    output_dir = f"{args.output_dir}/best_checkpoint"
                    self.trainer.save_model(output_dir)
                    self.trainer.save_state()

        return control


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def create_scheduler(self, num_training_steps: int, optimizer):
        args = self.args
        p = args.lr_scheduler_kwargs

        warmup_steps = args.warmup_steps
        decay_steps = p["decay_steps"]
        constant_steps = p["constant_steps"]
        lr_floor = p["lr_floor"]              #  absolute LR floor
        base_lr = args.learning_rate

        def lr_lambda(step):
            # Warmup
            if step < warmup_steps:
                return step / warmup_steps

            # Decay
            elif step < warmup_steps + decay_steps:
                progress = (step - warmup_steps) / decay_steps
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                floor_scale = lr_floor / base_lr
                return max(floor_scale, cosine)

            # Constant floor phase
            else:
                return lr_floor / base_lr

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler
