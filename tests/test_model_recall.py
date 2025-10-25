from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import config
from fine_tune import train_seq_pred_subseq
import torch
from torch.utils.data import DataLoader
import numpy as np

LEVEL = 1


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your adapter folder
adapter_path = config.MODEL_DIR / f"train_seq_pred_subseq_level{LEVEL}"/ "best_checkpoint"

# Load adapter config
config = PeftConfig.from_pretrained(adapter_path)

# Load base model
base_model, tokenizer, _ = train_seq_pred_subseq.load_model_tokenizer(run_test=False)

# Load adapter on top
model = PeftModel.from_pretrained(base_model, adapter_path)
model.to(DEVICE)
model.eval()

gen_eval_dataset = train_seq_pred_subseq.SeqGenDataset("eval")

k = 5
max_new_tokens = LEVEL * 2
num_beams = 20
recalls_dict = []

dataloader = DataLoader(gen_eval_dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    prompts = batch["prompt"]
    level_prompts = train_seq_pred_subseq.generate_variations_fn(prompts, choice=LEVEL)
    targets = batch["target"]
    level_targets = train_seq_pred_subseq.generate_variations_fn(targets, choice=LEVEL)

    # Tokenize batch
    inputs = tokenizer(
        level_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)

    batch_size = len(level_prompts)
    
    # Generate sequences for the batch
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=k,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Reshape outputs: (batch_size, num_return_sequences, seq_len)
    batch_outputs = outputs.view(batch_size, k, -1)

    # Decode and compute top-k recall
    for i in range(batch_size):
        prompt_len = inputs["input_ids"].size(1)
        decoded_outputs = [
            tokenizer.decode(batch_outputs[i, x, prompt_len:], skip_special_tokens=True)
            for x in range(k)
        ]

        hits = [1 if level_targets[i] in o else 0 for o in decoded_outputs]
        recalls_dict.append(int(any(hits)))

print("Recall@5: ", np.mean(recalls_dict) )