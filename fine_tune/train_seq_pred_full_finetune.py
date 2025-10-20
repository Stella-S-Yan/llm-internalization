"""
Train seq prediction with full model fine tune
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


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_MODEL_DIR = config.MODEL_DIR / "train_seq_pred_extended_vocab"
df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 16
EPOCHS = 6    # training stablizes at epoch=120, batch_size=4, lr=1e-3, weight_decay=0.035
LR = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 2000    # Learn slower than 2_000
NUM_TRAIN_STEPS = 20_000

"""
EMB_LR:
0.08 too big, loss go up, even with 0.35 lora_dropout, 0.01 wd_body
0.06, 0.04 all learn slowly, 0.02 seems the best

BASE_LR:
1e-4 too big, 1e-6 seems best

"""


run_name = f"full_emb_lr{EMB_LR}_base_lr{BASE_LR}_wd_emb{WD_EMB}_wd_body{WD_BODY}_bs{TRAIN_BATCH_SIZE}_warmup_{WARMUP_STEPS}_epoch{EPOCHS}_lora_rank{LORA_RANK}_lora_ratio{LORA_RATIO}_lora_dropout{LORA_DROPOUT}"
LOGGING_DIR =  config.RUN_DIR / "train_seq_pred_extended_vocab" / run_name



def load_model_tokenizer(run_test=False):
    # model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.bfloat16)  
    model = AutoModelForCausalLM.from_pretrained("/usr/local/google/home/stellasyan/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6", dtype=torch.bfloat16)  
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

        sequence = record["input_ids"]

        seq_enc = self.tokenizer(
            sequence,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length, 
            padding=False
        )
        

        # --- Concatenate ---
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

        sequence = record["input_ids"]
        toks = sequence.split(" ")
        prompt = " ".join(toks[:-4])
        target = " ".join(toks[-4:])

        return {
            "prompt": prompt,
            "target": target,
        }


class SaveModelPerEpochCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer=None):
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)

        if model is None:
            print("No model found in kwargs. Skipping save.")
            return

        # Always overwrite the same folder (no multiple checkpoints)
        save_dir = os.path.join(self.output_dir, "checkpoint")
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

        print(f"Saved model checkpoint to {save_dir} at end of epoch {int(state.epoch)}")


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
        learning_rate=LR,   # base LR passed to Trainer, overridden by our custom groups
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=500,
        optim="adafactor",
        bf16=True,          # <<< enable bfloat16 (H100 optimized)
        fp16=False,         # optional: if you want fp16 instead
        report_to="tensorboard"
    )

    #  Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,      # We are doing causal language modeling
    )

    optimizer = AdamW([
        {"params": model.get_input_embeddings().parameters()},
    ], lr=LR, weight_decay=WEIGHT_DECAY)

    # --- Scheduler setup ---
    num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    num_warmup_steps = int(0.1 * num_training_steps)   # e.g., 10% warmup

    print(f"# warmup_steps: {num_warmup_steps}")
    print(f"# training_steps: {num_training_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, lr_scheduler),  # pass both,  # override Hugging Face defaults
        callbacks=[SaveModelPerEpochCallback(output_dir=OUTPUT_MODEL_DIR, tokenizer=tokenizer)]
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
    model, tokenizer, old_vocab_size = load_model_tokenizer(run_test=True)

    train_dataset = SeqDataset(tokenizer, "train")
    eval_dataset = SeqDataset(tokenizer, "eval")
    gen_eval_dataset = SeqGenDataset("eval")

    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset)


if __name__ == "__main__":
    main()