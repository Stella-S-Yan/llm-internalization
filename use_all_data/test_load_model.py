from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import os

save_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_merged_think_sft_model"
model = AutoModelForCausalLM.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
print(len(tokenizer))

print(model)