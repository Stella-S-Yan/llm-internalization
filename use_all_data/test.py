from importlib import reload
from . import merge_save_load_model
from use_all_data import train_thinking
from utils import bagz_utils
import config
from torch.utils.data import DataLoader
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_input_dir = config.MODEL_DIR / "think_model_sft"
model, tokenizer = merge_save_load_model.load_merged_model(model_input_dir)

eval_dataset = train_thinking.ReasoningDataset(
    split="eval",
    datatype="raw_text"
)
print(eval_dataset[0])


def collate_fn(batch):
    prompt_token_ids = [item["prompt_token_ids"] for item in batch]
    targets = [item["target"] for item in batch]

    prompt_token_ids = torch.nn.utils.rnn.pad_sequence(
        prompt_token_ids,
        batch_first=True, 
        padding_value=tokenizer.pad_token_id, 
        padding_side="left"
    )

    return {
        "prompt_token_ids": prompt_token_ids,
        "target": targets,
    }

eval_loader = DataLoader(
    eval_dataset,
    batch_size=2,
    num_workers=0,  # > 0 has pickle and unpickle overhead, very slow
    shuffle=False,
    collate_fn=collate_fn,
)

k = 3
for batch in eval_loader:
    prompt_token_ids = batch["prompt_token_ids"]

    batch_size = len(prompt_token_ids)

    outputs = model.generate(
        input_ids=prompt_token_ids,
        max_new_tokens=206,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=k,
        use_cache=True
    )

    batch_outputs = outputs.view(batch_size, k, -1)

    for i in range(batch_size):
        decoded_outputs = [
            tokenizer.decode(batch_outputs[i, j], skip_special_tokens=True)
            for j in range(k)
        ]

        print(decoded_outputs)

        print("here")