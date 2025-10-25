"""
Train sequence prediction model using extended vocabulary. Update both new embeddings and model params
No user_id added. 

DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_extended_vocab.py
"""

import json
import random
import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator
from transformers import Trainer, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, random_split
from transformers import TrainerCallback
import numpy as np
import bagz
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from utils import bagz_utils


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_MODEL_DIR = config.MODEL_DIR / "lepard_train_seq_pred"
df_file = config.META_W_SID

TRAIN_BATCH_SIZE = 256
EPOCHS = 10    # training stablizes at epoch=120, batch_size=4, lr=1e-3, weight_decay=0.035
LR = 5e-5
WEIGHT_DECAY = 0.0

EMB_LR = 0.03       # 0.03 reach better region than 0.02, 0.04 overfit very fast
BASE_LR = 1e-5      # 1e-6 is too small for full-attn update
WD_EMB = 0.0        # Can't be too large, or will forget learning >0.2 is so wrong
WD_BODY = 0.05       # >0.2 is wrong; 0.035 too small, overfit; 
LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
LORA_RANK = 8      # 16 large rank overfit early
LORA_RATIO = 2
WARMUP_STEPS = 2_000    # 2k warmups is much better than 3K warmup
DECAY_STEPS = 2_000     # 3k decay is worse -> needs quick decay
MIN_LR_RATIO = 0.05     # 0.1 overfit, 0.08 overfit, 0.05 overfits very little, 0.03 will not learn well


run_name = f"full_emb_lr{EMB_LR}_base_lr{BASE_LR}_wd_emb{WD_EMB}_wd_body{WD_BODY}_bs{TRAIN_BATCH_SIZE}_warmup_{WARMUP_STEPS}_decay{DECAY_STEPS}_epoch{EPOCHS}_lora_rank{LORA_RANK}_lora_ratio{LORA_RATIO}_lora_dropout{LORA_DROPOUT}_min_lr_ratio{MIN_LR_RATIO}"
LOGGING_DIR =  config.RUN_DIR / "lepard_train_seq_pred" / run_name


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
        if split == "train":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_TRAIN)
        elif split == "eval":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_DEV)
        elif split == "test":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_TEST)

        self.data = self.df[['formatted_dest_sid', 'formatted_source_sid']].values.tolist()
        
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
        if split == "eval":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_DEV)
        elif split == "test":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_TEST)

        self.data = self.df[['formatted_dest_sid', 'formatted_source_sid']].values.tolist()

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
    
    
    # def create_scheduler(self, num_training_steps: int, optimizer=None):
    #     """
    #     Custom cosine learning rate schedule with warmup.
    #     """
    #     if self.lr_scheduler is None:

    #         print("~~~~ SCheduler: ")
    #         # --- Custom scheduler hyperparams ---

    #         # # Use the optimizer passed in (if provided) or self.optimizer
    #         opt = optimizer or self.optimizer

    #         # # --- Define the scheduler ---
    #         # self.lr_scheduler = get_cosine_schedule_with_warmup(
    #         #     opt,
    #         #     num_warmup_steps=WARMUP_STEPS,
    #         #     num_training_steps=num_training_steps,
    #         #     num_cycles=0.5,  # single cosine cycle
    #         # )

    #         # ~~~~ Scheduler: Warmup -> Decay -> Plateau 
    #         self.lr_scheduler = get_warmup_decay_plateau_scheduler(
    #             opt,
    #             num_training_steps=num_training_steps,
    #             warmup_steps=WARMUP_STEPS,
    #             min_lr_ratio=MIN_LR_RATIO,  # adjust as needed
    #             decay_steps=DECAY_STEPS,
    #         )
    #     return self.lr_scheduler

    def log(self, logs, *args, **kwargs):
        # Inject custom learning rates before logging
        if hasattr(self, "optimizer") and self.optimizer is not None:
            lrs = [group["lr"] for group in self.optimizer.param_groups]
            for i, lr in enumerate(lrs):
                logs[f"learning_rate_group_{i}"] = lr

        # Call the parent Trainer's log() to let HF handle everything
        super().log(logs, *args, **kwargs)
    


class LRTensorboardCallback(TrainerCallback):
    """
    Logs learning rates for all optimizer parameter groups to TensorBoard,
    in a separate namespace so they're visible.
    """

    def __init__(self):
        self.writer = None
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", None)
        # Custom subdirectory to avoid conflict with HF's internal writer
        self.writer = SummaryWriter(log_dir=f"{args.logging_dir}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.trainer is None:
            self.trainer = kwargs.get("trainer", None)
        if self.trainer is None or not hasattr(self.trainer, "optimizer"):
            return control

        optimizer = self.trainer.optimizer
        if optimizer is None:
            return control

        for i, group in enumerate(optimizer.param_groups):
            lr_value = group.get("lr", None)
            if lr_value is not None:
                tag = f"group_{i}_learning_rate"
                # Write to TensorBoard (custom namespace)
                self.writer.add_scalar(tag, lr_value, state.global_step)
                # Optional: also add to logs for console output
                if logs is not None:
                    logs[tag] = lr_value

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
        # eval_strategy="no",
        optim="adamw_torch",
        # optim="adafactor",
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


    # --- Trainer ---
    trainer = TwoLRTrainer(
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

    # trainer.add_callback(LRTensorboardCallback())

    trainer.train()


def main():
    model, tokenizer, old_vocab_size = load_model_tokenizer(run_test=True)

    train_dataset = SeqDataset(tokenizer, "train")
    eval_dataset = SeqDataset(tokenizer, "eval")
    gen_eval_dataset = SeqGenDataset("eval")

    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset, gen_eval_dataset)
    

if __name__ == "__main__":
    main()