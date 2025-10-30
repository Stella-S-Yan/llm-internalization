"""
Train sequence prediction model using extended vocabulary. Freeze old and new embeddings, and tune model params only.

DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_sft.py
"""

import random
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import Trainer
from torch.utils.data import Dataset
from transformers import TrainerCallback
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from utils import bagz_utils
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_MODEL_DIR = config.MODEL_DIR / "lepard_train_seq_pred_sft"
df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 256
EPOCHS = 100    # training stablizes at epoch=120, batch_size=4, lr=1e-3, weight_decay=0.035
LR = 1e-3
WEIGHT_DECAY = 0.0
LORA_RANK = 8
LORA_RATIO = 1
LORA_DROPOUT = 0.1


run_name = f"lr{LR}_batch_size{TRAIN_BATCH_SIZE}_epoch{EPOCHS}_r{LORA_RANK}_ratio_{LORA_RATIO}_dropout_{LORA_DROPOUT}"
LOGGING_DIR =  config.RUN_DIR / "lepard_train_seq_pred_sft" / run_name
MODEL_LOAD_DIR = config.MODEL_DIR / "lepard_sid_aligned_model"
MODEL_SAVE_DIR = config.MODEL_DIR / "lepard_seq_pred_model"


def load_checkpoint():
    model = AutoModelForCausalLM.from_pretrained(MODEL_LOAD_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_DIR)
    
    # Load optimizer state
    checkpoint_path = os.path.join(MODEL_LOAD_DIR, "training_state.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    optimizer = torch.optim.Adam(
        [model.get_input_embeddings().weight],  # or whichever params you train
        lr=1e-4
    )

    optimizer.load_state_dict(checkpoint["optimizer"])

    # Move optimizer state tensors to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)

    return model.to(DEVICE), tokenizer, optimizer, epoch, global_step



class SeqDataset(Dataset):
    def __init__(self, tokenizer, split):
        self.data = []

        self.tokenizer = tokenizer
        if split == "train":
            for group in range(8):
                self.df = bagz_utils.read_parquet(f"{config.LEPARD_W_SID_TRAIN}_{group}")
                tmp = self.df[['formatted_dest_sid','formatted_source_sid']].values.tolist()
                self.data.extend(tmp)
        elif split == "eval":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_DEV)
            tmp = self.df[['formatted_dest_sid','formatted_source_sid']].values.tolist()
            self.data.extend(tmp)
        elif split == "test":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_TEST)
            tmp = self.df[['formatted_dest_sid','formatted_source_sid']].values.tolist()
            self.data.extend(tmp)

        
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        input = self.data[idx][0]
        target = self.data[idx][1]

        sequence = input + " " + target

        seq_enc = self.tokenizer(
            sequence,
            add_special_tokens=False,
            truncation=True,
            max_length=16, 
            padding=False
        )
        
        input_ids = seq_enc["input_ids"]

        # --- Labels ---
        mask_start = max(0, len(input_ids) - 8)
        labels = [-100] * mask_start + input_ids[mask_start:]

        # Ensure labels same length as input_ids
        labels = labels[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class SeqGenDataset(Dataset):
    def __init__(self, split="eval"):
        self.data = []

        if split == "eval":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_DEV)
        elif split == "test":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_TEST)

        tmp = self.df[['formatted_dest_sid','formatted_source_sid']].values.tolist()
        self.data.extend(tmp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx][0]
        target = self.data[idx][1]

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
        self.batch_size = 512

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

        return control


def train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset):

    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
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
        optim="adamw_torch",
        bf16=True,          # <<< enable bfloat16 (H100 optimized)
        fp16=False,         # optional: if you want fp16 instead
        report_to="tensorboard"
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

    # Wrap the base model with LoRA
    peft_model = get_peft_model(model, lora_config)

    # Freeze all base model parameters (done automatically by get_peft_model)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


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
        eval_steps=2000  # or whatever interval you want
    )
    trainer.add_callback(callback)


    trainer.train()


def main():
    model, tokenizer, _, _, _ = load_checkpoint()

    train_dataset = SeqDataset(tokenizer, "train")
    eval_dataset = SeqDataset(tokenizer, "eval")
    gen_eval_dataset = SeqGenDataset("eval")

    train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset)
    

if __name__ == "__main__":
    main()