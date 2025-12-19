"""
Build reasoning data takes time. Build once and save the data to save experiment running time.
"""

from transformers import AutoTokenizer
import config
from utils import bagz_utils
from utils import merge_save_load_model
import bagz
import json
import torch
import re


PROMPT_TEMPLATE = """
<sft:think>
user {uid}: {history}
prediction:\n
{predict}
"""

TARGET_TEMPLATE = """
<hsz>{history_len}</hsz>
<hist>
    {sid_cat_list}
</hist>
<cat>{target_sid_cat}</cat>
<sid>{target_sid}</sid>{eos}
"""

GEN_PROMPT_TEMPLATE = """
<sft:think>
user {uid}: {history}
prediction:\n
<hsz>{history_len}</hsz>
<hist>
    {sid_cat_list}
</hist>
"""

GEN_TARGET_TEMPLATE = """
<cat>{target_sid_cat}</cat>
<sid>{target_sid}</sid>{eos}
"""


PATTERN = r"A\d+\s+B\d+\s+C\d+\s+D\d+"

def do_the_work(tokenizer, split):

    meta_df = bagz_utils.read_parquet(config.META_ALL_SID) 
    sid_to_cat = dict(zip(meta_df['formatted_sid'], meta_df['fine_category']))

    if split == "train":
        data_reader = bagz.Reader(config.TRAIN_DATA)
    elif split == "eval":
        data_reader = bagz.Reader(config.EVAL_DATA)
    elif split == "test":
        data_reader = bagz.Reader(config.TEST_DATA)
    elif split == "train_eval":
        data_reader = bagz.Reader(config.TRAIN_EVAL_DATA)
    
    raw_data = [json.loads(record.decode()) for record in data_reader]

    all_data = []

    for record in raw_data:
        uid = record["uid"]
        history = record["input"]
        target = record["target"]

        # Prefix UID by data source
        if config.REVIEW_TYPE == "Beauty":
            uid = f"B_{uid}"
        elif config.REVIEW_TYPE == "Toys_and_Games":
            uid = f"T_{uid}"
        elif config.REVIEW_TYPE == "Sports_and_Outdoors":
            uid = f"S_{uid}"

        # sids = [x.strip() for x in history.split(";")]
        sids = re.findall(PATTERN, history)
        cats = [sid_to_cat.get(i) for i in sids]
        history_len = len(sids)

        sid_cat_list = "\n".join(f"{cats[i]}" for i in range(len(sids)))
        target_sid_cat = sid_to_cat.get(target)

        prompt = PROMPT_TEMPLATE.format(
            uid=uid,
            history=history.strip(),
            predict=""
        ).strip()

        result = TARGET_TEMPLATE.format(
            history_len=str(history_len),
            sid_cat_list=sid_cat_list,
            target_sid_cat=target_sid_cat,
            target_sid=target,
            eos=tokenizer.eos_token
        ).strip()

        gen_prompt = GEN_PROMPT_TEMPLATE.format(
            uid=uid,
            history=history.strip(),
            history_len=str(history_len),
            sid_cat_list=sid_cat_list
        ).strip()

        gen_target = GEN_TARGET_TEMPLATE.format(
            target_sid_cat=target_sid_cat,
            target_sid=target,
            eos=tokenizer.eos_token
        ).strip()

        prompt_enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        result_enc = tokenizer(
            result,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        solution = {
            "hsz": history_len,
            "hist": cats,
            "cat": target_sid_cat,
            "sid": target,
        }


        # For SFT
        input_ids = prompt_enc["input_ids"] + result_enc["input_ids"]
        # result starts immediately after prompt
        result_start = len(prompt_enc["input_ids"])

        # attention_mask = [1] * len(input_ids)
        labels = [-100] * result_start + input_ids[result_start:]

        data = {
            "prompt": prompt,
            "prompt_token_ids": torch.tensor(prompt_enc["input_ids"]),
            "target": result,
            "solution": solution,
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "gen_prompt": gen_prompt,
            "gen_target": gen_target
        }

        all_data.append(data)

    # Save data
    torch.save(all_data, config.PROCESSED_DATA_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_think_data_{split}.pt")


def main():
    # Only need to load tokenizer
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_all_sid_alignment"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # Make sure to use fast tokenizer

    do_the_work(tokenizer, "train")
    do_the_work(tokenizer, "eval")
    # do_the_work(tokenizer, "test")
    # do_the_work(tokenizer, "train_eval")

if __name__ == "__main__":
    main()