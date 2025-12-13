"""
Merge LoRA adaptor and base model into one model for downstream use.
Evaluate the model too. 

Recall mean: tensor([0.0204, 0.0516, 0.0726])

$ torchrun --nproc_per_node=8 save_full_model_eval.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from torch.utils.data import DataLoader
from peft import  PeftModel
from use_all_data import train_seq_pred_aligned_phase1
from torch.utils.data import DataLoader, DistributedSampler
import os
from torch.utils.data import Subset


MODEL_INPUT_DIR = config.MODEL_DIR / "all_sid_aligned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTOR_DIR = config.MODEL_DIR / f"train_seq_pred_aligned_phase1"
# MODEL_SAVE_DIR = config.MODEL_DIR / f"merged_best_sft"
MODEL_SAVE_DIR = config.MODEL_DIR / "train_DPO" / "checkpoint-8455"


def merge_and_save_model(model_input_dir, adaptor_dir, checkpoint_dir, model_save_dir):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_input_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_input_dir)

    # Load adaptor
    model = PeftModel.from_pretrained(model, f"{adaptor_dir}/{checkpoint_dir}")

    # Merge adaptor
    model = model.merge_and_unload()

    # Save merged model
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)


def load_model(model_input_dir):
    model = AutoModelForCausalLM.from_pretrained(model_input_dir, attn_implementation="flash_attention_2")
    # model = AutoModelForCausalLM.from_pretrained(model_input_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_input_dir)

    model, tokenizer = setup_model(model, tokenizer)

    return model, tokenizer


def setup_model(model, tokenizer):
    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    return model, tokenizer


def main():
    # Initialize distributed process group if launched with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="gloo")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Load model and tokenizer
    model, tokenizer = load_model()
    model.eval()  # important for evaluation
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    # Dataset
    gen_eval_dataset = train_seq_pred_aligned_phase1.SeqGenDataset("eval")
    # gen_eval_dataset = Subset(gen_eval_dataset, range(32*8))

    # Distributed sampler splits data per rank
    sampler = DistributedSampler(gen_eval_dataset, shuffle=False) if world_size > 1 else None

    eval_loader = DataLoader(
        gen_eval_dataset,
        batch_size=8,
        sampler=sampler,
        shuffle=False if sampler is None else None,
        collate_fn=None,
        num_workers=4,  # adjust
    )

    # Evaluate locally
    recalls_local = train_seq_pred_aligned_phase1.evaluate_sequence_recall(model, tokenizer, eval_loader)
    recalls_tensor = torch.tensor(list(recalls_local.values()), device="cpu")

    # Aggregate across all ranks
    if world_size > 1:
        gather_list = [torch.zeros_like(recalls_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gather_list, recalls_tensor)
        recalls_mean = torch.mean(torch.stack(gather_list), dim=0)  # per metric mean
    else:
        recalls_mean = recalls_tensor

    if rank == 0:
        print("Recall mean:", recalls_mean)


if __name__ == "__main__":
    main()