"""
Build reasoning data efficiently with batch tokenization and multiprocessing. But it's very slow
even slower than my single process version.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformers import AutoTokenizer
import config
from utils import bagz_utils
import bagz
import json
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

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

BATCH_SIZE = 4096  # Adjust based on RAM


def build_sample(record, sid_to_cat, tokenizer):
    uid = record["uid"]
    history = record["input"]
    target = record["target"]

    sids = [x.strip() for x in history.split(";")]
    cats = [sid_to_cat.get(i) for i in sids]
    history_len = len(sids)

    sid_cat_list = "\n".join(f"{i}: {cats[i]}" for i in range(len(sids)))
    target_sid_cat = sid_to_cat.get(target)

    prompt = PROMPT_TEMPLATE.format(uid=uid, history=history.strip()).strip()
    result = TARGET_TEMPLATE.format(
        history_len=str(history_len),
        sid_cat_list=sid_cat_list,
        target_sid_cat=target_sid_cat,
        target_sid=target,
        eos=tokenizer.eos_token
    ).strip()

    return {
        "prompt": prompt,
        "result": result,
        "solution": {
            "hsz": history_len,
            "hist": cats,
            "cat": target_sid_cat,
            "sid": target,
        }
    }


def encode_batch(batch, tokenizer):
    prompts = [x["prompt"] for x in batch]
    results = [x["result"] for x in batch]

    prompt_enc = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
        padding=False
    )
    result_enc = tokenizer(
        results,
        add_special_tokens=False,
        truncation=False,
        padding=False
    )

    encoded = []
    for i, item in enumerate(batch):
        input_ids = prompt_enc["input_ids"][i] + result_enc["input_ids"][i]
        result_start = len(prompt_enc["input_ids"][i])
        labels = [-100] * result_start + input_ids[result_start:]

        encoded.append({
            "prompt": item["prompt"],
            "prompt_token_ids": torch.tensor(prompt_enc["input_ids"][i]),
            "target": item["result"],
            "solution": item["solution"],
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        })

    return encoded


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

    # --- Parallel build of prompts and results ---
    with Pool(cpu_count()) as pool:
        build_func = partial(build_sample, sid_to_cat=sid_to_cat, tokenizer=tokenizer)
        all_samples = list(tqdm(pool.imap(build_func, raw_data, chunksize=512), total=len(raw_data)))

    # --- Batch tokenization ---
    all_data = []
    for i in tqdm(range(0, len(all_samples), BATCH_SIZE), desc=f"Encoding {split}"):
        batch = all_samples[i:i + BATCH_SIZE]
        encoded = encode_batch(batch, tokenizer)
        all_data.extend(encoded)

    # Save processed data
    torch.save(all_data, config.PROCESSED_DATA_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_think_data_{split}.pt")


def main():
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_all_sid_alignment"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # Make sure to use fast tokenizer

    for split in ["train", "eval", "test", "train_eval"]:
        do_the_work(tokenizer, split)


if __name__ == "__main__":
    main()
