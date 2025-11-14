"""
Phase 1 training for seq pred. Use aligned new embeddings; fix all embeddings; only tune LoRA parameter.

Able to achieve 4.97% recall@5



DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_aligned_phase1.py
"""

import json
import random
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
import numpy as np
import bagz
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from torch.utils.data import Subset
import torch.nn.functional as F


MODEL_INPUT_DIR = config.MODEL_DIR / "all_sid_aligned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"train_seq_pred_ce_softmax"



class Params:
    TRAIN_BATCH_SIZE = 16
    LR = 4e-4
    WEIGHT_DECAY = 1e-3
    TOTAL_STEPS = 16_000    # 13_000

    LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
    LORA_RANK = 16      # 16 large rank overfit early
    LORA_RATIO = 1
    WARMUP_STEPS = 1000    # 2k warmups is much better than 3K warmup


TEMPLATE = """Rule:
            Each product ID has four hierarchical levels. 
            Earlier levels in the ID are more important than later levels.

            Task:
            Given a user's purchase history as a list of product IDs, predict the next product ID with four hierarchical levels.

            History:
            user {uid}: {history}

            Next:
            {next}
        """

def load_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_INPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INPUT_DIR)

    return model, tokenizer


class SeqDataset(Dataset):
    def __init__(self, tokenizer, split):

        self.tokenizer = tokenizer

        if split == "train":
            self.data_reader = bagz.Reader(config.TRAIN_DATA)
        elif split == "eval":
            self.data_reader = bagz.Reader(config.EVAL_DATA)
        elif split == "test":
            self.data_reader = bagz.Reader(config.TEST_DATA)

        self.data = [json.loads(record.decode()) for record in self.data_reader]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        uid = record["uid"]
        input = record["input"]
        target = record["target"]

        prompt = TEMPLATE.format(uid=uid, history=input, next=target).strip()

        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        input_ids = prompt_enc["input_ids"]

        mask_start = max(0, len(input_ids) -  7)
        labels = [-100] * mask_start + input_ids[mask_start:]
        labels = labels[:len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
            }


class SeqGenDataset(Dataset):
    def __init__(self, split="eval"):
        if split == "train":
            self.data_reader = bagz.Reader(config.TRAIN_DATA)
        if split == "eval":
            self.data_reader = bagz.Reader(config.EVAL_DATA)
        elif split == "test":
            self.data_reader = bagz.Reader(config.TEST_DATA)

        # Convert all records in one shot
        self.data = [json.loads(record.decode()) for record in self.data_reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        uid = record["uid"]
        input = record["input"]
        target = record["target"]

        prompt = TEMPLATE.format(uid=uid, history=input, next=target).strip()

        toks = prompt.split(" ")
        prompt = " ".join(toks[:-4])
        target = " ".join(toks[-4:])

        return {
            "prompt": prompt,
            "target": target,
        }


def sft_data_collator(batch, tokenizer):
    """
    Pads variable-length input_ids and labels in a batch.
    - input_ids padded with tokenizer.pad_token_id
    - labels padded with -100 (so prompts are ignored)
    Returns attention_mask automatically.
    """
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

    # pad sequences to the max length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")  
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100, padding_side="left")


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
    max_new_tokens=8,
    top_k_list=[1, 5, 10],
    print_random_example=True,  # new flag
):
    """
    Batched sequence-level recall evaluation.

    Args:
        model: Hugging Face causal LM
        tokenizer: Hugging Face tokenizer
        eval_dataset: list of dicts with 'prompt' and 'target' fields
        batch_size: number of prompts per batch
        num_beams: number of beams for beam search
        max_new_tokens: maximum tokens to generate
        top_k_list: which recalls to compute (e.g., [1,5,10])

    Returns:
        dict: {'recall_1': float, 'recall_5': float, ...}
    """
    model.eval()
    device = model.device

    # Initialize recall lists
    recalls_dict = {k: [] for k in top_k_list}
    printed = False  # track if we've printed already

    # Process dataset in batches
    for batch in tqdm(eval_loader, desc="Evaluating"):
        prompts = batch["prompt"]
        targets = batch["target"]

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        batch_size = len(prompts)
        max_k = max(top_k_list)

        
        # Generate sequences for the batch
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, max(top_k_list)),
            num_return_sequences=max(top_k_list),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Reshape outputs: (batch_size, num_return_sequences, seq_len)
        batch_outputs = outputs.view(batch_size, max(top_k_list), -1)

        # Decode and compute top-k recall
        for i in range(batch_size):
            prompt_len = inputs["input_ids"].size(1)
            decoded_outputs = [
                tokenizer.decode(batch_outputs[i, k, prompt_len:], skip_special_tokens=True)
                for k in range(max_k)
            ]
            # print(decoded_outputs)

            hits = [1 if targets[i] in o else 0 for o in decoded_outputs]
            for k in top_k_list:
                recalls_dict[k].append(int(any(hits[:k])))


        # ---- Print one random batch example ----
        if print_random_example and not printed:
            rand_idx = random.randint(0, batch_size - 1)
            print("\n=== Random Example ===")
            print(f"Prompt:\n{prompts[rand_idx]}")
            print(f"Target:\n{targets[rand_idx]}")
            for k, gen in enumerate(decoded_outputs[:5]):  # show top 5 generations
                print(f"[Gen {k+1}] {gen}")
            print("========================\n")
            printed = True
        # ----------------------------------------

    # Compute mean recall
    recalls_mean = {f"recall_{k}": float(np.mean(v)) for k, v in recalls_dict.items()}
    return recalls_mean


class GenerateEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tokenizer, eval_fn, eval_steps=1000):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.batch_size = 8
        self.best_metric = None  # Track best metric

    def on_step_end(self, args, state, control, **kwargs):

        # dynamically adjust evaluation frequency
        if 8000 <= state.global_step:
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


class CESoftMaxTrainer(Trainer):
    def __init__(self, token_weights, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.token_weights = token_weights
        self.alpha = alpha
        self.gen_len = 7
        # Preconvert to tensor for efficiency
        self.token_weights = torch.tensor(token_weights, dtype=torch.float32)

        # Normalize learning rate based on average weight
        self.avg_weight = self.token_weights.mean().item()
        print(f"Average token weight = {self.avg_weight:.3f}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=10): # num_items_in_batch=10 is to avoid a Trainer internal error
       # Prevent HF Trainer from passing num_items_in_batch
        self.model_accepts_loss_kwargs = False

        # Forward pass
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits           # [B, T, V]
        labels = inputs["labels"]         # [B, T]

        shift_logits = logits[:, :-1].contiguous()   # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()    # [B, T-1]

        B, T, V = logits.size()
        G = self.gen_len

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none"
        )
        # reshape back to [B, T-1]
        per_token_loss = per_token_loss.view(shift_labels.size())

        gen_labels = labels[:, -G:]
        gen_loss = per_token_loss[:, -G:]

        weights = self.token_weights.to(gen_loss.device).unsqueeze(0).expand(B, -1)  # [B, G]
        weighted_loss = gen_loss * weights
        
        ce_loss = weighted_loss.sum() / self.avg_weight

        # ------ Sequence loss -----------
        # Extract generated region (last G tokens)
        gen_logits = logits[:, -G:, :]  # [B, G, V]
        gen_labels = labels[:, -G:]     # [B, G]

        # Vectorized log-probs
        log_probs = torch.log_softmax(gen_logits, dim=-1)   # [B, G, V]

        # Gold sequence log-prob
        # Selecting indices along the vocabulary dimension
        gold_lp =log_probs.gather(
            dim=2,
            index=gen_labels.unsqueeze(-1)
        ).squeeze(-1)       # [B, G]
        gold_seq_lp = gold_lp.sum(dim=1)
        
        # Negative sequence (argmax tokens)
        neg_tokens = gen_logits.argmax(dim=-1)  # [B, G]

        # 2. Detect if entire sequence is identical to gold
        is_identical = (neg_tokens == gen_labels).all(dim=1)  # [B]

        # 3. Only adjust those rare cases
        neg_tokens = torch.where(
            is_identical.unsqueeze(-1),
            (neg_tokens + 1) % V,
            neg_tokens
        )

        neg_lp = log_probs.gather(2, neg_tokens.unsqueeze(-1)).squeeze(-1)  # [B, G]
        neg_seq_lp = neg_lp.sum(dim=1)                                      # [B]

        seq_loss = -torch.log(torch.sigmoid(gold_seq_lp - neg_seq_lp)).mean()

        # Combine losses
        loss = self.alpha * ce_loss + (1 - self.alpha) * seq_loss

        return (loss, outputs) if return_outputs else loss





def train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, params):
    print(f"@@@ total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        logging_dir=params.LOGGING_DIR,
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        # num_train_epochs=EPOCHS,
        max_steps=params.TOTAL_STEPS,
        learning_rate=params.LR,   # base LR passed to Trainer, overridden by our custom groups
        weight_decay=params.WEIGHT_DECAY,
        warmup_steps=params.WARMUP_STEPS,      # warm up for 1000 steps
        lr_scheduler_type="cosine",  # can also try "cosine", "linear"
        logging_steps=50,
        # save_strategy="steps",
        # save_steps=1000,
        save_strategy="no",
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=500,
        # eval_strategy="no",
        optim="adamw_torch",
        # optim="adafactor",
        bf16=True,          # <<< enable bfloat16 (H100 optimized)
        fp16=False,         # optional: if you want fp16 instead
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
    )
    
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        # target_modules=["q_proj", "v_proj"],  # attention projections
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    peft_model = get_peft_model(model, lora_config)

    # Freeze all base model parameters (done automatically by get_peft_model)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    
    # --- Trainer ---
    trainer = CESoftMaxTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        token_weights=[5.0, 0.01, 1.0, 0.01, 0.5, 0.01, 0.01],
        data_collator=lambda batch: sft_data_collator(batch, tokenizer),  # use custom collator
    )

    callback = GenerateEvalCallback(
        trainer=trainer,
        eval_dataset=gen_eval_dataset,
        tokenizer=tokenizer,
        eval_fn=evaluate_sequence_recall,
        eval_steps=5000 
    )
    trainer.add_callback(callback)

    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=2, help="Training batch size")
    parser.add_argument("--LORA_RATIO", type=float, default=0.1, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=20000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}"
    Params.LOGGING_DIR =  config.RUN_DIR / "train_seq_pred_ce_softmax" / run_name

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    model, tokenizer = load_model_tokenizer()
    old_vocab_size = 128_256
    
    train_dataset = SeqDataset(tokenizer, "train")
    eval_dataset = SeqDataset(tokenizer, "eval")

    gen_eval_dataset = SeqGenDataset("eval")

    train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, Params)
    

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()