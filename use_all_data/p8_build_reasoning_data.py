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


PROMPT_TEMPLATE = """
                User history:
                user {uid}: {history}
                Predict next semantic ID with reasoning:
            """

TARGET_TEMPLATE = """
            <hsz>{history_len}</hsz>
            <hist>
                {sid_cat_list}
            </hist>
            <cat>{target_sid_cat}</cat>
            <sid>{target_sid}</sid>{eos}
            """


def do_the_work(tokenizer, split):

    meta_df = bagz_utils.read_parquet(config.META_W_ALL_SID) 
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

        sids = [x.strip() for x in history.split(";")]
        cats = [sid_to_cat.get(i) for i in sids]
        history_len = len(sids)

        sid_cat_list = "\n".join(f"{i}: {cats[i]}" for i in range(len(sids)))
        target_sid_cat = sid_to_cat.get(target)

        prompt = PROMPT_TEMPLATE.format(
            uid=uid,
            history=history.strip()
        ).strip()

        result = TARGET_TEMPLATE.format(
            history_len=str(history_len),
            sid_cat_list=sid_cat_list,
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
            "labels": torch.tensor(labels)
        }

        all_data.append(data)

    # Save data
    torch.save(all_data, config.PROCESSED_DATA_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_think_data_{split}.pt")


def main():
    # Only need to load tokenizer
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_all_sid_alignment"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # Make sure to use fast tokenizer

    do_the_work(tokenizer, "train")
    do_the_work(tokenizer, "eval")
    # do_the_work(tokenizer, "test")
    # do_the_work(tokenizer, "train_eval")

if __name__ == "__main__":
    main()