"""
Build reasoning data takes time. Build once and save the data to save experiment running time.
"""

from transformers import AutoTokenizer
import config
from utils import bagz_utils
import bagz
import json
import re
from tqdm import tqdm
from collections import Counter


PROMPT_TEMPLATE = """
<sft:think>
user {uid}:
{sid_cat_list}
prediction:\n
{predict}
"""

TARGET_TEMPLATE = """
<freq>{freq_A}</freq>
<cat>{target_sid_cat}</cat>
<sid>{target_sid}</sid>{eos}
"""

PATTERN = r"A\d+\s+B\d+\s+C\d+\s+D\d+"


def most_frequent_Ax(ids):
    """
    ids: list of strings like "A3 B5 C7 D2"
    returns: most frequent Ax (e.g. "A3") or None
    """
    ax_list = [id_.split()[0] for id_ in ids]
    counts = Counter(ax_list)

    if not counts:
        return None

    most_common, freq = counts.most_common(1)[0]

    # Check if it's strictly most frequent
    if list(counts.values()).count(freq) > 1:
        return None

    return most_common


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
    

    all_data = []

    for record_bytes in tqdm(data_reader, desc=f"Processing {split}"):
        record = json.loads(record_bytes.decode())
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

        # most frequent Ax
        freq_A = most_frequent_Ax(sids)

        sid_cat_list = "\n".join(f"{sids[i]} : {cats[i]}" for i in range(len(sids)))
        target_sid_cat = sid_to_cat.get(target)

        prompt = PROMPT_TEMPLATE.format(
            uid=uid,
            sid_cat_list=sid_cat_list,
            predict=""
        ).strip()

        target = TARGET_TEMPLATE.format(
            freq_A=freq_A,
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
            target,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )

        solution = {
            "cat": target_sid_cat,
            "sid": target,
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
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_all_sid_alignment"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # Make sure to use fast tokenizer

    do_the_work(tokenizer, "train")
    do_the_work(tokenizer, "eval")
    # do_the_work(tokenizer, "test")
    # do_the_work(tokenizer, "train_eval")

if __name__ == "__main__":
    main()