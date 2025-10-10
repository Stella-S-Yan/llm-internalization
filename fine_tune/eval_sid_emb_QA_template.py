"""
Check if the model can generate sid tokens
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_MODEL_DIR = config.MODEL_DIR / "learn_sid_model_st" / "checkpoint-17100"

def load_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)
    return model, tokenizer


def eval(model, tokenizer):
    
    model.eval()
    model.to(DEVICE)
    
    eval_prompts = [
        # "The title of sid A135 B45 C199 D0 is",   # WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette
        # "The title is 'WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette', the sid is: ",   # 'A135 B45 C199 D0'
        # "What is the the title of  SemanticID A135 B45 C199 D0 ?"
        "The product is WAWO 15 Color Professionl Makeup Eyeshadow. What is its semanticID? "
        # "What brand makes SemanticID A135 B45 C199 D0?"
        # "What brand makes A1 B58 C120 D0?"
        # "What brand makes SemanticID A1 B58 C120 D0?"
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

    for q in eval_prompts:
        answer = ask_model(q)
        print(f"Q: {q}\nA: {answer}\n{'-'*40}")
    

def main():
    model, tokenizer = load_model_tokenizer()
    eval(model, tokenizer)

if __name__ == "__main__":
    main()