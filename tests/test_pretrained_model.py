"""
Test the behavior of a pretrained model.
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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import Trainer
from torch.utils.data import Dataset, random_split
import os
import re

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTOR_DIR = config.MODEL_DIR / "emb_lora_sentence"


def load_model_tokenizer(run_test = False):
    """
    Load base model and tokenizer, add new tokens for SID embeddings.
    """
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.bfloat16)  
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer, 


    
    
def evaluate(model, tokenizer):
    
    model.eval()

    eval_prompts = [
        # "The product is WAWO 15 Color Professionl Makeup Eyeshadow. Do you know what is it for? "
        # "What brand makes SemanticID A135 B45 C199 D0?"
        "The quick brown fox jumps over the lazy dog."
    ]

    def ask_model(prompt, max_new_tokens=50):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,    # deterministic (greedy). Use True if you want variety.
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return completion.strip()


    # 4. Run evaluation
    for q in eval_prompts:
        answer = ask_model(q)
        print(f"Q: {q}\nA: {answer}\n{'-'*40}")
    

    
def main():
    
    model, tokenizer = load_model_tokenizer()
    
    evaluate(model, tokenizer)
    
    
if __name__ == "__main__":
    main()