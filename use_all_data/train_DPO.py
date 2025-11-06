from trl import DPOTrainer, DPOConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from utils import bagz_utils
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import os
import itertools
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import copy
from transformers import TrainingArguments
from datasets import Dataset as HFDataset

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_INPUT_DIR = config.MODEL_DIR / f"merged_best_sft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / f"train_DPO"
ADAPTOR_DIR = config.MODEL_DIR / f"train_seq_pred_aligned_phase1"


class Params:
    TRAIN_BATCH_SIZE = 16
    LR = 4e-4
    WEIGHT_DECAY = 1e-3
    TOTAL_STEPS = 16_000    # 13_000

    LORA_DROPOUT = 0.1     # turn to 0.3 leads to overfit, weirdly. 0.01 also overfits, 0.05 seems best
    LORA_RANK = 16      # 16 large rank overfit early
    LORA_RATIO = 1
    WARMUP_STEPS = 1000    # 2k warmups is much better than 3K warmup
    LOGGING_DIR = "/"


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_INPUT_DIR, 
        device_map="auto",  # automatically puts model layers on available GPUs
        dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INPUT_DIR)
    return model, tokenizer


def train(model, tokenizer, params):

    lora_config = LoraConfig(
        r=params.LORA_RANK,                      # rank
        lora_alpha=params.LORA_RANK * params.LORA_RATIO,
        # target_modules=["q_proj", "v_proj"],  # attention projections
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=params.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    data = {
        "chosen": [
            "I love sunny days.",
            "Pizza is my favorite food."
        ],
        "rejected": [
            "I hate sunny days.",
            "I dislike pizza."
        ]
    }
    dataset = HFDataset.from_dict(data)


    # 3. Tokenize dataset
    def tokenize(example):
        chosen = tokenizer(example["chosen"], truncation=True, padding="max_length", max_length=16)
        rejected = tokenizer(example["rejected"], truncation=True, padding="max_length", max_length=16)
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }

    tokenized_ds = dataset.map(tokenize, batched=False)


    # 5. DPO config (beta is temperature-like for preference)
    dpo_args = DPOConfig(
        output_dir="./dpo_output",
        learning_rate=1e-5,
        per_device_train_batch_size=2,  # batch size per device
        per_device_eval_batch_size=2,
        beta=0.01,       # 0.01~0.05, Set beta very small, essentially letting DPO act as pure preference-based learning. for new token-heavy task
        max_prompt_length=128,
        max_completion_length=8,
    )

    # 6. Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        data_collator= None,
        train_dataset=tokenized_ds,
        eval_dataset=tokenized_ds,
        processing_class=tokenizer,  # your tokenizer
        compute_metrics= None,
        callbacks=None,
        peft_config=lora_config
    )
    # 7. Train
    trainer.train()

def main():
    model, tokenizer = load_model()
    train(model, tokenizer, Params)


if __name__ == '__main__':
    main()