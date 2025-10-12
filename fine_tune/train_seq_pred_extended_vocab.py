"""
Train sequence prediction model using extended vocabulary only. Update both new embeddings and model params
No user_id added

Sembolic sequence modeling setup
{
  "input_ids": "UID_5626 A28 B191 C56 D0 A80 B84 C53 D0 A48 B141 C76 D0 A240 B194 C71 D0",
  "target_ids": "A0 B140 C246 D0"
}

When to add semantic prompts?
1. Plan to later use the mode in a language-driven context ("given a user's shopping history, describe what they might buy next")
2. Want to leverage LLM's language prior to improve generalization when symbolic data is limited.

Middle ground
"User: UID_5626"
"History: A28 B191 C56 D0 A80 B84 C53 D0"
"Next: A0 B140 C246 D0"

In SFT format: 
{
  "input": "User: UID_5626\nHistory: A28 B191 C56 D0 A80 B84 C53 D0 A48 B141 C76 D0 A240 B194 C71 D0\nNext:",
  "output": "A0 B140 C246 D0"
}

DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_seq_pred_extended_vocab.py
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
from transformers import Trainer
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


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_MODEL_DIR = config.MODEL_DIR / "train_seq_pred_extended_vocab"
df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 4
EPOCHS = 300    # training stablizes at epoch=120, batch_size=4, lr=1e-3, weight_decay=0.035
LR = 1e-3
WEIGHT_DECAY = 0.035

run_name = f"sid_context_cp_lr{LR}_epoch{EPOCHS}_bs{TRAIN_BATCH_SIZE}"
LOGGING_DIR =  config.RUN_DIR / "train_seq_pred_extended_vocab" / run_name


def load_model_tokenizer(run_test=False):
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.bfloat16)  
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"           

    old_vocab_size = len(tokenizer)
    print("Original vocab size:", old_vocab_size)
    prefix_tokens = [f"{prefix}{i}" for prefix in "ABCD" for i in range(256)]
    tokenizer.add_tokens(prefix_tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Updated vocab size:", len(tokenizer))

    # Save new tokenizer
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    if run_test:
        text1 = "A157 B141 C28 D0"
        tokens = tokenizer.tokenize(text1)
        print("tokens: ", tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(ids)
        text_back = tokenizer.convert_tokens_to_string(tokens)
        print("text_back_from_tokens: ", text_back)
        tokens_back = tokenizer.convert_ids_to_tokens(ids)
        print("id_back_to_tokens: ", tokens_back)
        text_back = tokenizer.convert_tokens_to_string(tokens_back)
        print("id_back_from_text: ", text_back)
        
        token_id = old_vocab_size  # index you want to check
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token at index {token_id}: {token_str}")    # <A0>
    
    return model, tokenizer, old_vocab_size


class SeqDataset(Dataset):
    def __init__(self, tokenizer, split):
        self.tokenizer = tokenizer
        self.max_prompt_length=160
        self.max_target_length=8
        if split == "train":
            self.data_reader = bagz.Reader(config.TRAIN_DATA)
        elif split == "eval":
            self.data_reader = bagz.Reader(config.EVAL_DATA)
        elif split == "test":
            self.data_reader = bagz.Reader(config.TEST_DATA)

        # Convert all records in one shot
        self.data = [json.loads(record.decode()) for record in self.data_reader]

        # self.max_len = 0
        # self.max_idx = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        prompt = record["input_ids"]
        target = record["target_ids"]

        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length, 
            padding=False
        )
        target_enc = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length,
            padding=False
        )

        # --- Concatenate ---
        input_ids = prompt_enc["input_ids"] + target_enc["input_ids"]

        # --- Labels ---
        prompt_len = len(prompt_enc["input_ids"])
        labels = [-100] * prompt_len + target_enc["input_ids"]

        # Ensure labels same length as input_ids
        labels = labels[:len(input_ids)]

        # if len(labels) > self.max_len:
        #     self.max_len = len(labels)
        #     self.max_idx = idx

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class SeqGenDataset(Dataset):
    def __init__(self, split="eval"):
        self.max_prompt_length=160
        self.max_target_length=8
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

        prompt = record["input_ids"]
        target = record["target_ids"]

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
    max_new_tokens=7,
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



def train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset):
    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        logging_dir=LOGGING_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=500,
        # eval_strategy="no",
        optim="adamw_torch",
        bf16=True,          # <<< enable bfloat16 (H100 optimized)
        fp16=False,         # optional: if you want fp16 instead
        report_to="tensorboard"
    )
    
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=8,                      # rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # attention projections
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Wrap the base model with LoRA
    peft_model = get_peft_model(model, lora_config)

    # Allow new embeddings to train
    for param in peft_model.get_input_embeddings().parameters():
        param.requires_grad = True

    # ensures that only new vocabulary tokens get updated.
    def zero_old_token_grads(grad):
        grad[:old_vocab_size] = 0
        return grad
    
    # Register the hook on the input embeddings
    peft_model.get_input_embeddings().weight.register_hook(zero_old_token_grads)


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
        eval_steps=500  # or whatever interval you want
    )
    trainer.add_callback(callback)

    trainer.train()


def main():
    model, tokenizer, old_vocab_size = load_model_tokenizer(run_test=True)

    train_dataset = SeqDataset(tokenizer, "train")
    eval_dataset = SeqDataset(tokenizer, "eval")
    gen_eval_dataset = SeqGenDataset("eval")

    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset)
    

    # ------ Test beam search --------------
    # model.to(DEVICE)
    
    # eval_loader = DataLoader(
    #             gen_eval_dataset,
    #             batch_size=8,
    #             sampler=None,
    #             shuffle=False,
    #             collate_fn=None  # or custom collate_fn if needed
    #         )
    

    # # Call the function
    # metrics = evaluate_sequence_recall(
    #     model=model,
    #     tokenizer=tokenizer,
    #     eval_loader=eval_loader,
    #     num_beams=5,
    #     max_new_tokens=10,
    #     top_k_list=[1, 5]
    # )

    # print(metrics)
    # -------------------------------


if __name__ == "__main__":
    main()