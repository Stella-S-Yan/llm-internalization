
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

from LLM_INTERNALIZATION import config
from LLM_INTERNALIZATION.utils import bagz_utils


class ReasoningDataset(Dataset):
    def __init__(self, split, datatype: str, sources):
        self.datatype = datatype

        self.data = []
        for src in sources:
            data_path = os.path.join(config.PROCESSED_DATA_DIR / f'{config.DATA_SOURCE}_{src}_think_data_{split}.bagz')
            self.data.extend(bagz_utils.read_record(data_path))

        # === CRITICAL FIX: FORCE DETERMINISTIC ORDER ===
        # DDP requires self.data to be identical (index-for-index) on every GPU.
        self.data.sort(key=lambda x: str(x["prompt"]))


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        record = self.data[idx]
        if self.datatype == "sft":
            return {
                "input_ids": record["input_ids"],
                "labels": record["labels"],
                "length": len(record["input_ids"])
            }
        elif self.datatype == "grpo":
            return {
                "prompt": record["prompt"],
                "solution": record["solution"],
            }
        elif self.datatype == "raw_text_vllm":  
            return {
                "prompt": {"prompt": record["prompt"], "prompt_token_ids": record["prompt_token_ids"].tolist()},
                "target": record["target"],
                "solution": record["solution"],
            }
        elif self.datatype == "raw_text":
            return {
                "prompt_token_ids": record["prompt_token_ids"],
                "target": record["target"],
            }
        elif self.datatype == "gen_eval":
            return {
                "gen_prompt": record["prompt"],
                "gen_target": record["target"]
            }
        else:
            raise ValueError(
                f"Invalid datatype '{self.datatype}'. "
                f"Expected one of: ['sft', 'grpo', 'raw_text', 'raw_text_vllm']"
            )
     

def sft_data_collator(batch, tokenizer):
    """
    Pads variable-length input_ids and labels in a batch.
    - input_ids padded with tokenizer.pad_token_id
    - labels padded with -100 (so prompts are ignored)
    Returns attention_mask automatically.
    """
    # Convert each input/label to a torch tensor
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="left"
    )

    labels = pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=-100, 
        padding_side="left"
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

