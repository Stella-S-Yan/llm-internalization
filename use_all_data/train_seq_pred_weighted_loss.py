"""
Train sequence prediction model using extended vocabulary. Update both new embeddings and model params
No user_id added. 

Extend training data with subsequences. 
1. sequence: UID_x A1 A2 A3 -> A4
2. sequence: UID_x A1 B1 A2 B2 A3 B3 -> A4 B4
3. sequence: UID_x A1 B1 C1 A2 B2 C2 A3 B3 C3 -> A4 B4 C4
4. full sequence: UID_x A1 B1 C1 D1 A2 B2 C2 D2 ... A30 B30 C30 D30 -> A31 B31 C31 D31



DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_aligned.py
"""

import json
import random
from utils import bagz_utils
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import Trainer, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, random_split
from transformers import TrainerCallback
from fine_tune import amazon_ori_template
import pandas as pd
import os
from torch.optim import AdamW
import numpy as np
import bagz
from torch.utils.data import Subset
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from transformers import TopKLogitsWarper, LogitsProcessorList
import math
from utils import checkpoint_loading
from peft import PeftModel, PeftConfig


MODEL_INPUT_DIR = config.MODEL_DIR / "all_sid_aligned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"train_seq_pred_aligned"


TRAIN_BATCH_SIZE = 16
EPOCHS = 10    # training stablizes at epoch=120, batch_size=4, lr=1e-3, weight_decay=0.035
LR = 5e-5
WEIGHT_DECAY = 0.0

EMB_LR = 5e-5       # 0.03 reach better region than 0.02, 0.04 overfit very fast
BASE_LR = 1e-5      # 1e-6 is too small for full-attn update
WD_EMB = 0.0        # Can't be too large, or will forget learning >0.2 is so wrong
WD_BODY = 0.05       # >0.2 is wrong; 0.035 too small, overfit; 
LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
LORA_RANK = 16      # 16 large rank overfit early
LORA_RATIO = 1
WARMUP_STEPS = 2_000    # 2k warmups is much better than 3K warmup
DECAY_STEPS = 10_000     # 3k decay is worse -> needs quick decay
MIN_LR_RATIO = 0.08     # 0.1 overfit, 0.08 overfit, 0.05 overfits very little, 0.03 will not learn well



"""
EMB_LR:
0.08 too big, loss go up, even with 0.35 lora_dropout, 0.01 wd_body
0.06, 0.04 all learn slowly, 0.02 seems the best

BASE_LR:
1e-4 too big, 1e-6 seems best

"""


run_name = f"emb_lr{EMB_LR}_base_lr{BASE_LR}_wd_emb{WD_EMB}_wd_body{WD_BODY}_bs{TRAIN_BATCH_SIZE}_warmup_{WARMUP_STEPS}_decay{DECAY_STEPS}_epoch{EPOCHS}_lora_rank{LORA_RANK}_lora_ratio{LORA_RATIO}_lora_dropout{LORA_DROPOUT}_min_lr_ratio{MIN_LR_RATIO}"
LOGGING_DIR =  config.RUN_DIR / "train_seq_pred_subseq" / run_name

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



def make_weighted_labels(input_ids, tokenizer):
    position_weights = [5, 1, 0.5, 0.01]
    labels = input_ids.clone()
    weights = torch.ones_like(labels, dtype=torch.float)

    # Example: assume 4 main tokens per product ID
    main_token_indices = [0, 2, 4, 6]
    for i, idx in enumerate(main_token_indices):
        if idx < len(weights):
            weights[idx] = position_weights[i]

    # Zero out padding
    weights[labels == tokenizer.pad_token_id] = 0.0
    return labels, weights


class WeightedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract input_ids and labels
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

        # Compute loss weights per sample
        loss_weights = []
        for ex in input_ids:
            _, w = make_weighted_labels(ex, self.tokenizer)
            loss_weights.append(w)

        # Pad sequences to same length
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        loss_weights = torch.nn.utils.rnn.pad_sequence(loss_weights, batch_first=True, padding_value=0.0)

        attention_mask = (input_ids != pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
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


def get_warmup_decay_plateau_scheduler(
    optimizer,
    num_training_steps,
    warmup_steps,
    min_lr_ratio=0.1,
    decay_steps=None  # exact number of steps to decay after warmup
):
    """
    Scheduler with:
    1. Linear warmup
    2. Linear decay over a fixed number of steps
    3. Flat plateau at a minimum learning rate

    Args:
        optimizer: torch optimizer
        num_training_steps: total training steps
        warmup_steps: steps to linearly increase LR
        min_lr_ratio: final LR = min_lr_ratio * initial LR
        decay_steps: number of steps to decay after warmup. If None, uses all remaining steps.
    """

    if decay_steps is None:
        decay_steps = max(1, num_training_steps - warmup_steps)
    else:
        decay_steps = max(1, decay_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + decay_steps:
            # Linear decay
            progress = (current_step - warmup_steps) / decay_steps
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

            # Exponential decay
            # k = 5.0
            # p = (current_step - warmup_steps) / decay_steps  # progress 0→1
            # factor = math.exp(-k * p)
            # return max(min_lr_ratio, factor)
        
            # reverse cosine decay
            # p = (current_step - warmup_steps) / decay_steps
            # # reverse cosine: fast at start, slow near end
            # factor = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 - math.cos(math.pi * (1 - p)))
            # return max(min_lr_ratio, 1 - factor)
        else:
            # Flat plateau
            return min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class GenerateEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tokenizer, eval_fn, eval_steps=1000):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.batch_size = 8
        self.best_metric = None  # Track best metric

    def on_evaluate(self, args, state, control, **kwargs):
        # Run every eval_steps
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
        # if state.global_step > 0:

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


# --- Custom Trainer with two learning rates ---
class TwoLRTrainer(Trainer):
    def create_optimizer(self):
        print("~~~~ Optimizer: ")
        if self.optimizer is None:
            # Custom learning rates and weight decays
            emb_lr = EMB_LR       # for new embeddings
            base_lr = BASE_LR      # for LoRA + transformer body
            wd_emb = WD_EMB
            wd_body = WD_BODY

            # All other params except embeddings
            other_params = [
                p for n, p in self.model.named_parameters()
                if "lora" in n
            ]

            # Define optimizer param groups
            optimizer_grouped_parameters = [
                {"params": self.model.get_input_embeddings().parameters(), "lr": emb_lr, "weight_decay": wd_emb},
                {"params": other_params, "lr": base_lr, "weight_decay": wd_body},
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Custom cosine learning rate schedule with warmup.
        """
        if self.lr_scheduler is None:

            print("~~~~ SCheduler: ")
            # --- Custom scheduler hyperparams ---

            # # Use the optimizer passed in (if provided) or self.optimizer
            opt = optimizer or self.optimizer

            # # --- Define the scheduler ---
            # self.lr_scheduler = get_cosine_schedule_with_warmup(
            #     opt,
            #     num_warmup_steps=WARMUP_STEPS,
            #     num_training_steps=num_training_steps,
            #     num_cycles=0.5,  # single cosine cycle
            # )

            # ~~~~ Scheduler: Warmup -> Decay -> Plateau 
            self.lr_scheduler = get_warmup_decay_plateau_scheduler(
                opt,
                num_training_steps=num_training_steps,
                warmup_steps=WARMUP_STEPS,
                min_lr_ratio=MIN_LR_RATIO,  # adjust as needed
                decay_steps=DECAY_STEPS,
            )
        return self.lr_scheduler

    def log(self, logs, *args, **kwargs):
        # Inject custom learning rates before logging
        if hasattr(self, "optimizer") and self.optimizer is not None:
            lrs = [group["lr"] for group in self.optimizer.param_groups]
            for i, lr in enumerate(lrs):
                logs[f"learning_rate_group_{i}"] = lr

        # Call the parent Trainer's log() to let HF handle everything
        super().log(logs, *args, **kwargs)


def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    loss_weights = inputs.pop("loss_weights")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss * loss_weights.view(-1)
    loss = loss.mean()
    return (loss, outputs) if return_outputs else loss


def train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset):
    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        logging_dir=LOGGING_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LR,   # base LR passed to Trainer, overridden by our custom groups
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

    # Save old_vocab_size to config (so the Trainer can access it)
    peft_model.config.old_vocab_size = old_vocab_size

    # Allow new embeddings to train
    for param in peft_model.get_input_embeddings().parameters():
        param.requires_grad = True

    # ensures that only new vocabulary tokens get updated.
    def zero_old_token_grads(grad):
        grad[:old_vocab_size] = 0
        return grad
    
    # Register the hook on the input embeddings
    peft_model.get_input_embeddings().weight.register_hook(zero_old_token_grads)


    data_collator = WeightedDataCollator(tokenizer=tokenizer)
    
    # --- Trainer ---
    trainer = TwoLRTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_loss=compute_loss,
    )

    callback = GenerateEvalCallback(
        trainer=trainer,
        eval_dataset=gen_eval_dataset,
        tokenizer=tokenizer,
        eval_fn=evaluate_sequence_recall,
        eval_steps=1000 
    )
    trainer.add_callback(callback)

    trainer.train()


def main():
    model, tokenizer = load_model_tokenizer()
    old_vocab_size = 128_256
    
    train_dataset = SeqDataset(tokenizer, "train")
    eval_dataset = SeqDataset(tokenizer, "eval")

    gen_eval_dataset = SeqGenDataset("eval")

    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset)
    

if __name__ == "__main__":
    main()