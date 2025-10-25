"""Apply unsupervised pre-training style of fine-tuning to teach the model to use the new SID tokens."""


import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_INPUT_DIR = config.MODEL_DIR / "train_emb_sentence_cp"


def get_latest_checkpoint(output_dir):
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # sort by step number
    checkpoints = sorted(checkpoints, key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)))
    return checkpoints[-1]


def load_model_tokenizer():
    latest_ckpt = get_latest_checkpoint(MODEL_INPUT_DIR)
    print(latest_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(latest_ckpt)

    model.eval()

    return model, tokenizer
    
    
def evaluate(model, tokenizer):
    
    model.eval()

    eval_prompts = [
        "What is the title of  SemanticID A135 B45 C199 D0 ?",
        "The product is WAWO 15 Color Professionl Makeup Eyeshadow. What is its semanticID? ",
        "What brand makes SemanticID A135 B45 C199 D0?",
        "The quick brown fox jumps over the lazy dog.",
        "What brand makes A1 B58 C120 D0?",
        "What product does the identifier A1 B58 C120 D0 refers to?",
        "In our catalog, A1 B58 C120 D0 corresponds to",
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