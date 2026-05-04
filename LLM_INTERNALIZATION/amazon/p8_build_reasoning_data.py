"""
Build reasoning data takes time. Build once and save the data to save experiment running time.
"""
import os
from transformers import AutoTokenizer
import bagz
import json
import re
from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np

from LLM_INTERNALIZATION import config
from LLM_INTERNALIZATION.utils import bagz_utils


PROMPT_TEMPLATE = """
<sft:think>
user {uid}:
{sid_cat_brand_price_list}
prediction:\n
{predict}
"""

TARGET_TEMPLATE = """
<cat>{target_cat}</cat>
<brand>{target_brand}</brand>
<price>{target_price}</price>
<sid>{target_sid}</sid>{eos}
"""

PATTERN = r"A\d+\s+B\d+\s+C\d+\s+D\d+"


def most_frequent_Ax(ids):
    if not ids: return None
    ax_list = [x.split(maxsplit=1)[0] for x in ids]
    counts = Counter(ax_list)
    if not counts: return None
    most_common = counts.most_common(2)
    best, freq = most_common[0]
    if len(most_common) > 1 and most_common[1][1] == freq: return None
    return best


def concat_categories(x):
    if isinstance(x, np.ndarray) and len(x) > 0:
        # unwrap object array: [ array([...]) ]
        if isinstance(x[0], np.ndarray):
            x = x[0]

    if isinstance(x, (list, np.ndarray)) and len(x) > 1:
        return ", ".join(map(str, x[1:]))

    return ""



def do_the_work(tokenizer, split):

    meta_df = bagz_utils.read_parquet(config.META_ALL_SID) 
    meta_df['categories_concat'] = meta_df['categories'].apply(concat_categories)
    sid_to_cat = dict(zip(meta_df['formatted_sid'], meta_df['categories_concat']))

    meta_df['brand'] = meta_df['brand'].fillna('unknown')
    sid_to_brand = dict(zip(meta_df['formatted_sid'], meta_df['brand']))

    num_bins = 10
    cut = pd.cut(meta_df["price"], bins=num_bins)
    price_left = np.array([interval.left if pd.notna(interval) else np.nan for interval in cut])
    meta_df["price_quant"] = pd.Series(price_left).fillna(-1).astype(int)
    sid_to_price = dict(zip(meta_df['formatted_sid'], meta_df['price_quant']))

    if split == "train":
        split_file = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_user_train.bagz" )
        data_reader = bagz.Reader(split_file)
    elif split == "eval":
        split_file = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_user_eval.bagz" )
        data_reader = bagz.Reader(split_file)
    elif split == "test":
        split_file = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_user_test.bagz" )
        data_reader = bagz.Reader(split_file)
    

    all_data = []

    for record_bytes in tqdm(data_reader, desc=f"Processing {split}"):
        record = json.loads(record_bytes.decode())
        uid = record["uid"]
        history = record["input"]
        target_sid = record["target"]

        sids = re.findall(PATTERN, history)
        cats = [sid_to_cat.get(i) for i in sids]
        brands = [sid_to_brand.get(i) for i in sids]
        prices = [sid_to_price.get(i) for i in sids]

        # most frequent Ax
        freq_A = most_frequent_Ax(sids)

        sid_cat_brand_price_list = "\n".join(
                f"{sid}: {cat}; {brand}; {price}"
                for sid, cat, brand, price in zip(sids, cats, brands, prices)
            )
        target_cat = sid_to_cat.get(target_sid)
        target_brand = sid_to_brand.get(target_sid)
        target_price = sid_to_price.get(target_sid)

        prompt = PROMPT_TEMPLATE.format(
            uid=uid,
            sid_cat_brand_price_list=sid_cat_brand_price_list,
            predict=""
        ).strip()

        target = TARGET_TEMPLATE.format(
            freq_A=freq_A,
            target_cat=target_cat,
            target_brand=target_brand,
            target_price=target_price,
            target_sid=target_sid,
            eos=tokenizer.eos_token
        ).strip()

        prompt_enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        result_enc = tokenizer(
            target,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        solution = {
            "freq": freq_A if freq_A else 'None',
            "cat": target_cat,
            "brand": target_brand,
            "price": target_price,
            "sid": target_sid,
            "uid": uid
        }


        # For SFT
        input_ids = prompt_enc["input_ids"] + result_enc["input_ids"]
        # result starts immediately after prompt
        target_start = len(prompt_enc["input_ids"])

        # attention_mask = [1] * len(input_ids)
        labels = [-100] * target_start + input_ids[target_start:]

        
        data = {
            "prompt": prompt,
            "prompt_token_ids": prompt_enc["input_ids"],
            "target": target,
            "solution": solution,
            "input_ids": input_ids,
            "labels": labels,
        }

        all_data.append(data)

    # Save data
    bagz_utils.save_record(all_data, config.PROCESSED_DATA_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_think_data_{split}.bagz")

def main():

    # Only need to load tokenizer
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_all_sid_alignment"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # Make sure to use fast tokenizer

    do_the_work(tokenizer, "eval")
    do_the_work(tokenizer, "test")
    do_the_work(tokenizer, "train")

if __name__ == "__main__":
    main()