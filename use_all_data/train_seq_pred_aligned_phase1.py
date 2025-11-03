"""
Phase 1 training for seq pred. fix all embeddings.



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


MODEL_INPUT_DIR = config.MODEL_DIR / "all_sid_aligned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"train_seq_pred_aligned_phase1"


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
            # truncation=True,
            # max_length=self.max_prompt_length,
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
        if 8000 <= state.global_step <= 15000:
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

            # Create DataLoader
            eval_loader = DataLoader(
                self.eval_dataset,
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

        return control


def train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset):
    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        logging_dir=LOGGING_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        # num_train_epochs=EPOCHS,
        max_steps=TOTAL_STEPS,
        learning_rate=LR,   # base LR passed to Trainer, overridden by our custom groups
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,      # warm up for 1000 steps
        lr_scheduler_type="cosine",  # can also try "cosine", "linear"
        logging_steps=50,
        save_strategy="epoch",
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
        r=LORA_RANK,                      # rank
        lora_alpha=LORA_RANK * LORA_RATIO,
        # target_modules=["q_proj", "v_proj"],  # attention projections
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    peft_model = get_peft_model(model, lora_config)

    # Freeze all base model parameters (done automatically by get_peft_model)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    
    # --- Trainer ---
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
    global LR, WARMUP_STEPS, TRAIN_BATCH_SIZE, LORA_RATIO, LOGGING_DIR

    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=32, help="Training batch size")
    parser.add_argument("--LORA_RATIO", type=float, default=0.1, help="LoRA adapter ratio")

    args = parser.parse_args()

    # Assign to global config
    LR = args.LR
    WARMUP_STEPS = args.WARMUP_STEPS
    TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
    LORA_RATIO = args.LORA_RATIO
    run_name = f"lr{LR}_bs{TRAIN_BATCH_SIZE}_warmup_{WARMUP_STEPS}_lora_ratio{LORA_RATIO}_weight_decay{WEIGHT_DECAY}"
    LOGGING_DIR =  config.RUN_DIR / "train_seq_pred_aligned_phase1" / run_name


    model, tokenizer = load_model_tokenizer()
    old_vocab_size = 128_256
    
    train_dataset = SeqDataset(tokenizer, "train")
    eval_dataset = SeqDataset(tokenizer, "eval")

    gen_eval_dataset = SeqGenDataset("eval")

    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset)
    

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()