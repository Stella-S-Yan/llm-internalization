import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def load_model_tokenizer(model_input_dir):
    latest_ckpt = get_latest_checkpoint(model_input_dir)
    print(latest_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_input_dir)
    model = AutoModelForCausalLM.from_pretrained(latest_ckpt)
    tokenizer.padding_side = "left"

    model.eval()

    return model, tokenizer