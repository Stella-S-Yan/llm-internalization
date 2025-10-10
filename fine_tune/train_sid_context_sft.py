"""Apply SFT to teach the model to use the new SID tokens.
All embeddings (new + old) are frozen, all model parameters are frozen, only train LoRA adaptors

Input: "Q: What is SemanticID[43]? A: "
Labels: [-100, -100, ..., 1, 1, 1, ...]  # mask the question tokens

The model does not learn to predict the question text. It only computes loss on the answer part. 
This is typical in SFT: you want the model to understand the question but only generate the answer. 

But this does not work well for llm-internalization as the pretrained model won't already understand the prompts. 


DDP using all GPUs available.
# Using torchrun (PyTorch >=1.10)
$ torchrun --nproc_per_node=8 train_sid_context.py
"""


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
from fine_tune import amazon_qa_template
import pandas as pd
import os


BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_MODEL_DIR = config.MODEL_DIR / "sid_aligned_model"
SAVE_MODEL_DIR = config.MODEL_DIR / "sid_context_model"
TRAIN_BATCH_SIZE = 64
EPOCHS = 300
LR = 5e-5

run_name = f"sid_context_lr{LR}_epoch{EPOCHS}_bs{TRAIN_BATCH_SIZE}"
LOGGING_DIR =  config.RUN_DIR / "sid_context_finetune" / run_name


def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    old_vocab_size = len(tokenizer)
    print("Original vocab size:", old_vocab_size)

    model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)
    
    return model, tokenizer, old_vocab_size


class SIDDataset(Dataset):
    def __init__(self, tokenizer):
        self.df = bagz_utils.read_parquet(config.META_W_SID)
        self.tokenizer = tokenizer
        self.sep_ids = tokenizer(self.tokenizer.bos_token, add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product_data = {
            "sid": row["formatted_sid"],
            "title": row['title'] if pd.notna(row['title']) else "Unknown",
            "description": row['description'] if pd.notna(row['description']) else "Unknown",
            "brand": row['brand'] if pd.notna(row['brand']) else "Unknown",
            "fine_category": row['fine_category'] if pd.notna(row['fine_category']) else "Unknown"
        }
        
        # Randomly select a template type
        template_type = random.choice(amazon_qa_template.TEMPLATE_TYPES)
        
        # Fill it with values
        prompt_templates = amazon_qa_template.TEMPLATE_GROUPS[template_type]["prompt"]
        response_templates = amazon_qa_template.TEMPLATE_GROUPS[template_type]["response"]
        prompt_template = random.choice(prompt_templates)
        response_template = random.choice(response_templates)
        
        prompt = prompt_template.format(**product_data)
        response = response_template.format(**product_data)

        prompt_ids = self.tokenizer(
            prompt,
            max_length=128,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None
        )["input_ids"]

        response_ids = self.tokenizer(
            response,
            max_length=128,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None
        )["input_ids"]

        # Concatenate
        input_ids = prompt_ids + self.sep_ids +  response_ids

        # Labels: mask prompt, keep response
        labels = [-100] * (len(prompt_ids) + len(self.sep_ids)) + response_ids
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            # "prompt": prompt,     # for debugging
            # "response": response
        }
        


class CustomTrainer(Trainer):
    def __init__(self, *args, old_vocab_size=None, custom_optimizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if old_vocab_size is None:
            raise ValueError("You must provide old_vocab_size")
        self.old_vocab_size = old_vocab_size
        self.custom_optimizer = custom_optimizer  # store the optimizer

    def create_optimizer(self):
        # Called internally by Trainer if self.optimizer is None
        if self.custom_optimizer is not None:
            self.optimizer = self.custom_optimizer
        else:
            super().create_optimizer()
        return self.optimizer

    def training_step(self, model, inputs,  *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        # backward pass
        self.accelerator.backward(loss)

        # mask old embeddings grads
        emb = model.get_input_embeddings()
        if emb is not None and emb.weight.grad is not None:
            emb.weight.grad[:self.old_vocab_size] = 0

        if model.get_output_embeddings() is not None:
            out_emb = model.get_output_embeddings()
            if out_emb.weight.grad is not None:
                out_emb.weight.grad[:self.old_vocab_size] = 0

        return loss

    
class MaskOldTokensCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, old_vocab_size, model=None):
        super().__init__(tokenizer, model=model, padding=True)
        self.old_vocab_size = old_vocab_size

    def __call__(self, features):
        batch = super().__call__(features)
        
        if "labels" in batch:
            labels = batch["labels"]
            mask_old = labels < self.old_vocab_size
            labels = labels.clone()
            labels[mask_old] = -100  # ignored in loss
            batch["labels"] = labels
        return batch
    

class EvalCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, old_vocab_size, model=None):
        super().__init__(tokenizer, model=model, padding=True)
        self.old_vocab_size = old_vocab_size

    def __call__(self, features):
        batch = super().__call__(features)
        return batch


def evaluate_sid_accuracy(model, tokenizer, old_vocab_size, eval_dataset, batch_size=8, max_new_tokens=50):
    """
    Evaluate the fine-tuned model on SID accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    
    data_collator = EvalCollator(tokenizer, old_vocab_size, model=model)
    loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, drop_last=True)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            # Generate predictions
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Convert generated ids to strings
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            label_texts = [
                tokenizer.decode([tok for tok in l if tok != -100], skip_special_tokens=True)
                for l in batch["labels"].cpu().tolist()
            ]

            # Check if gold SID appears in generated text
            for pred, label in zip(pred_texts, label_texts):
                if label in pred:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eval_prompts = [
            # "The title of sid A135 B45 C199 D0 is",   # WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette
            # "The title is 'WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette', the sid is: ",   # 'A135 B45 C199 D0'
            "The product titled 'WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette' has the semantic ID "
        ]

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs['model']
        model.eval()
        print("\n=== Generation Evaluation ===")
        for prompt in self.eval_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"Q: {prompt}\nA: {completion}\n{'-'*40}")



def get_dataset(tokenizer, train_frac=0.9):
    full_dataset = SIDDataset(tokenizer)
    train_size = int(len(full_dataset) * train_frac)
    eval_size = len(full_dataset) - train_size

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)  # reproducible
    )
    
    return train_dataset, eval_dataset


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
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


class SavePeftModelCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer=None):
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        epoch_dir = os.path.join(self.output_dir, f"epoch_{int(state.epoch)}")

        # Save only adapters
        model.save_pretrained(epoch_dir)

        # Save tokenizer once (optional)
        if self.tokenizer is not None and state.epoch == 1:
            self.tokenizer.save_pretrained(self.output_dir)

        print(f"✅ Saved LoRA adapters at {epoch_dir}")


def train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset):

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=SAVE_MODEL_DIR,
        logging_dir=LOGGING_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=100,
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
        # callbacks=[SavePeftModelCallback(output_dir=SAVE_MODEL_DIR, tokenizer=tokenizer)]
    )

    trainer.train()
    
    

    
def main():
    
    print("# GPUs: ", torch.cuda.device_count())
    
    model, tokenizer, old_vocab_size = load_model_tokenizer()

    print("Special tokens: ", tokenizer.special_tokens_map)
    print(tokenizer.tokenize("A135 B45"))
    
    train_dataset, eval_dataset = get_dataset(tokenizer, train_frac=0.9)    
    
    train(model, tokenizer, old_vocab_size, train_dataset, eval_dataset)
    
    
if __name__ == "__main__":
    main()