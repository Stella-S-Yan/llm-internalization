"""
Check if the model can generate sid tokens
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_MODEL_DIR = config.MODEL_DIR / "sid_aligned_model"

def load_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)
    # optimizer = torch.optim.Adam(model.get_input_embeddings().parameters(), lr=LR)
    # optimizer.load_state_dict(torch.load(os.path.join(INPUT_MODEL_DIR, "optimizer.pt")))

    return model, tokenizer


def eval(model, tokenizer):
    
    model.eval()
    model.to(DEVICE)
    
    # prompt = "The product titled 'Concealers & Neutralizers' has the semantic ID "
    prompt = "The product titled 'Arts, Crafts & Sewing: None, Baby' has the semantic ID "


    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=4,
            do_sample=True,       # stochastic generation
            top_k=50,
            temperature=0.7,
        )

    # Decode full output
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # Extract the generated SID token
    sid_generated = generated_text[len(prompt):]  # only the new token part

    print(f"Title: Concealers & Neutralizers")
    print(f"Prompt: {prompt}")
    print(f"Generated SID token: {sid_generated}")
    print("-" * 50)

def main():
    model, tokenizer = load_model_tokenizer()
    eval(model, tokenizer)

if __name__ == "__main__":
    main()