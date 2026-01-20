"""
2_618_971 total training data


DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_thinking_sft.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true"


import config
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import argparse
import train_thinking
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import TrainerCallback
from torch.utils.data import Subset
from functools import partial

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Params:
    TRAIN_BATCH_SIZE = 16
    LR = 4e-4
    WEIGHT_DECAY = 1e-3
    TOTAL_STEPS = 16_000    # 13_000

    LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
    LORA_RANK = 16      # 16 large rank overfit early
    LORA_RATIO = 1
    WARMUP_STEPS = 1000    # 2k warmups is much better than 3K warmup
    ADAPTOR_SAVE_DIR = ''
    ACC_STEP=1
    RUN_NUM=0
    CHECK_POINT=0


def load_checkpoint(base_model_name, save_dir):
    # Load BASE MODEL again — quantized or FP16 as desired
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,   # or fp16, or load_in_4bit=True
    )

    # 2. Load extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)

    # 3. Resize embedding table
    model.resize_token_embeddings(new_vocab_size)

    # 4. Load saved new embedding weights
    new_emb = torch.load(os.path.join(save_dir, "new_embeddings.pt")).to(model.device)
    print(f"new_emb device: {model.device}")

    # 5. Insert the new embeddings back into the table
    with torch.no_grad():
        model.get_input_embeddings().weight[old_vocab_size:] = new_emb

    print(f"Restored model with extended vocab ({new_vocab_size} tokens)")

    return model, tokenizer


@torch.no_grad()
def evaluate_sequence_recall(
    model,
    tokenizer,
    eval_loader,
    num_beams=20,
    max_new_tokens=64,
    top_k_list=[5],
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

    local_hits = {k: 0 for k in top_k_list}
    local_total = 0
    printed = False  # track if we've printed already
    max_k = max(top_k_list)

    # Process dataset in batches
    for batch in tqdm(eval_loader, desc="Evaluating"):
        prompts = batch["prompt"]
        solutions = batch["solution"]

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        batch_size = len(prompts)
        
        input_seq_len = inputs["input_ids"].shape[1]

        # Generate sequences for the batch
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, max_k),
            num_return_sequences=max_k,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = gen_out.sequences
        scores = gen_out.sequences_scores

        # (batch, beams, seq_len)
        sequences = sequences.view(batch_size, max_k, -1)
        scores = scores.view(batch_size, max_k)

        # Decode and compute top-k recall
        for i in range(batch_size):
            # ---- Sort beams by descending score ----
            order = torch.argsort(scores[i], descending=True)
            sorted_seqs = sequences[i][order]

            sid = solutions[i]['sid']
            generations = [
                tokenizer.decode(
                    sorted_seqs[j, input_seq_len:],
                    skip_special_tokens=True,
                )
                for j in range(max_k)
            ]

            # ---- Recall@k with short-circuit ----
            for k in top_k_list:
                hit = False
                for j in range(k):
                    if sid in generations[j]:
                        hit = True
                        break
                local_hits[k] += int(hit)

        # ---- Print one example ----
        if print_random_example and not printed:
            idx = random.randint(0, batch_size - 1)
            print("\n=== Random Example ===")
            print(f"Prompt:\n{prompts[idx]}")
            print(f"Solution:\n{solutions[idx]}")
            for j in range(min(3, max_k)):
                print(f"[Gen {j+1}] {generations[j]}")
            print("========================\n")
            printed = True

        local_total += batch_size

    return local_hits, local_total


def no_processing_collator(batch):
    return {
        "prompt": [x["prompt"] for x in batch],
        "solution": [x["solution"] for x in batch]
    }


class GenerateEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tokenizer, eval_fn, eval_steps):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.batch_size = 16
        self.best_metric = None  # Track best metric

    # def on_step_end(self, args, state, control, **kwargs):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_interval = self.eval_steps

        # Run every eval_steps
        if state.global_step > 0 and state.global_step % eval_interval == 0:

            is_ddp = (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )

            rank = torch.distributed.get_rank() if is_ddp else 0
            world_size = torch.distributed.get_world_size() if is_ddp else 1

            for dataset_name, eval_dataset in self.eval_dataset.items():
                # ---- Sampler (per dataset) ----
                sampler = (
                    DistributedSampler(eval_dataset, shuffle=False)
                    if is_ddp
                    else None
                )

                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=self.batch_size,
                    num_workers=4,
                    sampler=sampler,
                    shuffle=False,
                    collate_fn=no_processing_collator,
                )

                # tqdm only on rank 0
                if rank == 0:
                    eval_loader = tqdm(
                        eval_loader,
                        desc=f"Eval @ step {state.global_step}",
                    )

                # ---- Custom generate-based eval ----
                local_hits, local_total = self.eval_fn(
                    self.trainer.model,
                    self.tokenizer,
                    eval_loader,
                )
                device = self.trainer.model.device

                # ---- Prepare tensors ----
                ks = sorted(local_hits.keys())  # ensure stable order
                hits_tensor = torch.tensor(
                    [local_hits[k] for k in ks],
                    device=device,
                    dtype=torch.long,
                )
                total_tensor = torch.tensor(
                    [local_total],
                    device=device,
                    dtype=torch.long,
                )

                # ---- DDP reduce (SUM ONLY) ----
                if is_ddp:
                    torch.distributed.all_reduce(hits_tensor, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)


                # ---- Compute recall@k (rank 0 only) ----
                if rank == 0:
                    total = total_tensor.item()
                    final_metrics = {
                        f"eval/{dataset_name}_recall_{k}": hits_tensor[i].item() / max(total, 1)
                        for i, k in enumerate(ks)
                    }
                    final_metrics["step"] = state.global_step

                    # ---- Log ----
                    self.trainer.log(final_metrics)

                    print(
                        f"\n[Custom eval @ step {state.global_step}] "
                        f"{final_metrics}"
                    )

            return control


def train(model, tokenizer, train_dataset, eval_dataset, gen_eval_dataset, params):
    print(f"@@@ total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    NUM_WORKERS = 1

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_{params.ADAPTOR_SAVE_DIR}_{params.RUN_NUM}",
        logging_dir=params.LOGGING_DIR,
        per_device_train_batch_size=params.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=params.ACC_STEP,
        max_steps=params.TOTAL_STEPS,
        learning_rate=params.LR,   # base LR passed to Trainer, overridden by our custom groups
        weight_decay=params.WEIGHT_DECAY,
        warmup_steps=params.WARMUP_STEPS,      # warm up for 1000 steps
        lr_scheduler_type="cosine",
        logging_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=10,
        load_best_model_at_end=False,
        eval_strategy="steps",
        eval_steps=2000,
        optim="adamw_torch",
        bf16=True,          # <<< enable bfloat16 (H100 optimized)
        fp16=False,         # optional: if you want fp16 instead
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True
    )
    
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        target_modules=[
            "q_proj",
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # model = torch.compile(model, mode="max-autotune")
    peft_model = get_peft_model(model, lora_config)

    # Freeze all base model parameters (done automatically by get_peft_model)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    collator_fn = partial(train_thinking.sft_data_collator, tokenizer=tokenizer)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator_fn
    )

    callback = GenerateEvalCallback(
        trainer=trainer,
        eval_dataset=gen_eval_dataset,
        tokenizer=tokenizer,
        eval_fn=evaluate_sequence_recall,
        eval_steps=2000,
    )
    trainer.add_callback(callback)

    if params.CHECK_POINT == 0:
        trainer.train()
    else:
        print(f"... Continue training from {params.CHECK_POINT} on node {params.RUN_NUM}")
        trainer.train(resume_from_checkpoint=f"/usr/local/google/home/stellasyan/Documents/llm_internalization/data/model/Amazon_Combined_think_sft_adaptor_{str(params.RUN_NUM)}/checkpoint-{str(params.CHECK_POINT)}")


def main():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--WARMUP_STEPS", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=32, help="Training batch size")
    parser.add_argument("--LORA_RANK", type=int, default=16, help="Rank of LoRA adaptor")
    parser.add_argument("--LORA_RATIO", type=float, default=0.1, help="LoRA adapter ratio")
    parser.add_argument("--TOTAL_STEPS", type=int, default=20000, help="Number of total training steps")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.2, help="LoRA dropout rate")
    parser.add_argument("--ADAPTOR_SAVE_DIR", type=str, default='think_sft_adaptor', help="Where to save the trained adaptor")
    parser.add_argument("--ACC_STEP", type=int, default=1, help="Gradient accumulate steps")
    parser.add_argument("--RUN_NUM", type=int, default=0, help="Run index")
    parser.add_argument("--CHECK_POINT", type=int, default=0, help="Checkpoint number")

    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Params, key, value)

    run_name = f"lr{Params.LR}_weight_decay{Params.WEIGHT_DECAY}_bs{Params.TRAIN_BATCH_SIZE}_warmup_{Params.WARMUP_STEPS}_rank{Params.LORA_RANK}_lora_ratio{Params.LORA_RATIO}_lora_dropout{Params.LORA_DROPOUT}_total_steps{Params.TOTAL_STEPS}_acc{Params.ACC_STEP}_{Params.RUN_NUM}"
    Params.LOGGING_DIR =  config.RUN_DIR / f"{config.DATA_SOURCE}_Combined_train_thinking_sft" / run_name

    print(f"!!! total_steps: {Params.TOTAL_STEPS}")
    print(vars(Params))

    # Load model and tokenizer in local device
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    save_dir = MODEL_SAVE_DIR = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_all_sid_alignment"
    # Load model to cpu first and let torchrun handle the device placement
    model, tokenizer = load_checkpoint(base_model_name, save_dir) 
    print(f"model_device: {model.device}")
    old_vocab_size = 128_256

    
    train_dataset = train_thinking.ReasoningDataset("train", "sft", ["Toys_and_Games", "Sports_and_Outdoors", "Beauty"])
    eval_dataset = train_thinking.ReasoningDataset("eval", "sft", ["Toys_and_Games", "Sports_and_Outdoors", "Beauty"])

    # train_dataset = train_thinking.ReasoningDataset("train", "sft", ["Sports_and_Outdoors"])
    # eval_dataset = train_thinking.ReasoningDataset("eval", "sft", ["Beauty"])
    # check_idx = 3
    # print(eval_dataset[check_idx])
    # print(tokenizer.decode(eval_dataset[check_idx]["input_ids"]))
    # print(tokenizer.decode([x for x in eval_dataset[check_idx]["labels"] if x != -100]))

    # eval_dataset = train_thinking.ReasoningDataset("eval", "grpo", ["Beauty"])
    # print(eval_dataset[0])
    
    gen_eval_dataset_1 = train_thinking.ReasoningDataset("eval", "grpo", ["Toys_and_Games"])
    # gen_eval_dataset_2 = train_thinking.ReasoningDataset("eval", "grpo", ["Sports_and_Outdoors"])
    # gen_eval_dataset_3 = train_thinking.ReasoningDataset("eval", "grpo", ["Beauty"])
    
    
    SEED = 411
    GEN_EVAL_SUBSET_SIZE = 5000
    rng = random.Random(SEED)   # <- LOCAL RNG (important!)
    indices = rng.sample(range(len(eval_dataset)), GEN_EVAL_SUBSET_SIZE)
    indices = sorted(indices)   # optional but recommended
    eval_dataset = Subset(eval_dataset, indices)
    print(f"---Eval dataset size: {len(eval_dataset)}")

    indices = rng.sample(range(len(gen_eval_dataset_1)), GEN_EVAL_SUBSET_SIZE)
    indices = sorted(indices)   # optional but recommended
    gen_eval_dataset_1 = Subset(gen_eval_dataset_1, indices)
    # gen_eval_dataset_2 = Subset(gen_eval_dataset_2, indices)
    # gen_eval_dataset_3 = Subset(gen_eval_dataset_3, indices)

    print(gen_eval_dataset_1[10])

    gen_eval_datasets = {
        "toys": gen_eval_dataset_1,
        # "sports": gen_eval_dataset_2,
        # "beauty": gen_eval_dataset_3
    }

    print(f"---Gen Eval dataset size: {len(eval_dataset)}")

    
    train(model, tokenizer, train_dataset, eval_dataset, gen_eval_datasets, Params)

    
if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()