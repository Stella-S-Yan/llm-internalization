"""
Build fixed reasoning data for MovieLens data. Only do it for eval & test data. 
Training data taks sampling approach.
"""


import os
# Force tokenizer parallelism off to prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing as mp
import config
from utils import bagz_utils
import bagz
import json
import re
import pandas as pd
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import itertools
import time

# --- Global Templates ---
PROMPT_TEMPLATE = """
<sft:think>
user {uid}:
{sid_year_genre_list}
prediction:\n
{predict}
"""

TARGET_TEMPLATE = """
<freq>{freq_A}</freq>
<year>{target_year}</year>
<genre>{target_genre}</genre>
<sid>{target_sid}</sid>{eos}
"""

PATTERN = re.compile(r"A\d+\s+B\d+\s+C\d+\s+D\d+")

# --- Global Variables for Workers ---
global_tokenizer = None
global_sid_to_genre = None
global_sid_to_year = None

def init_worker(model_path, metadata_path):
    """
    Worker initialization. 
    Each worker loads its own copy of data to avoid Pickling hangs.
    """
    global global_tokenizer, global_sid_to_genre, global_sid_to_year
    
    # Print PID so we know the worker is actually alive
    pid = os.getpid()
    # print(f"[Worker {pid}] Starting initialization...", flush=True)

    try:
        # 1. Load Tokenizer
        global_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # 2. Load Metadata (DataFrame)
        # Reading parquet is fast; better than pickling huge dicts across processes
        df = pd.read_parquet(metadata_path)
        df["year"] = df["Title"].str.extract(r"\((\d{4})\)")
        
        # Create lookups
        global_sid_to_genre = dict(zip(df['sid'], df['Genre']))
        global_sid_to_year = dict(zip(df['sid'], df['year']))
        
        # print(f"[Worker {pid}] Ready!", flush=True)
        
    except Exception as e:
        print(f"[Worker {pid}] CRASHED: {e}", flush=True)
        raise e

def most_frequent_Ax(ids):
    if not ids: return None
    ax_list = [x.split(maxsplit=1)[0] for x in ids]
    counts = Counter(ax_list)
    if not counts: return None
    most_common = counts.most_common(2)
    best, freq = most_common[0]
    if len(most_common) > 1 and most_common[1][1] == freq: return None
    return best

def process_batch(records_bytes):
    """Batch processing logic."""
    prompts = []
    targets = []
    solutions = []
    
    # Pre-fetch globals to local scope for slight speed bump
    tok = global_tokenizer
    s2y = global_sid_to_year
    s2g = global_sid_to_genre
    
    for rb in records_bytes:
        record = json.loads(rb.decode())
        uid = record["uid"]
        if config.REVIEW_TYPE == "1m":
            uid = f"S_{uid}"
        elif config.REVIEW_TYPE == "20m":
            uid = f"B_{uid}"
            
        history = record["input"]
        target_sid = record["target"]

        sids = PATTERN.findall(history)
        freq_A = most_frequent_Ax(sids)

        sid_year_genre_list = "\n".join(
            f"{sid} : {s2y.get(sid)}, {s2g.get(sid)}"
            for sid in sids
        )
        
        target_year = s2y.get(target_sid)
        target_genre = s2g.get(target_sid)

        prompt = PROMPT_TEMPLATE.format(
            uid=uid,
            sid_year_genre_list=sid_year_genre_list,
            predict=""
        ).strip()

        target = TARGET_TEMPLATE.format(
            freq_A=freq_A,
            target_year=target_year,
            target_genre=target_genre,
            target_sid=target_sid,
            eos=tok.eos_token
        ).strip()
        
        prompts.append(prompt)
        targets.append(target)
        solutions.append({
            "freq": freq_A,
            "year": target_year,
            "genre": target_genre,
            "sid": target_sid,
            "uid": uid
        })

    # Batch Tokenization
    prompt_encs = tok(prompts, add_special_tokens=False, truncation=False, padding=False)
    target_encs = tok(targets, add_special_tokens=False, truncation=False, padding=False)

    batch_data = []
    for i in range(len(prompts)):
        p_ids = prompt_encs["input_ids"][i]
        t_ids = target_encs["input_ids"][i]
        input_ids = p_ids + t_ids
        target_start = len(p_ids)
        labels = [-100] * target_start + input_ids[target_start:]

        batch_data.append({
            "prompt": prompts[i],
            "prompt_token_ids": p_ids,
            "target": targets[i],
            "solution": solutions[i],
            "input_ids": input_ids,
            "labels": labels,
        })
        
    return batch_data

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk: break
        yield chunk

def do_the_work_parallel(split, model_dir, metadata_path):
    # ... (Reader setup remains the same) ...
    if split == "eval": data_reader = bagz.Reader(config.EVAL_DATA)
    elif split == "test": data_reader = bagz.Reader(config.TEST_DATA)

    # 1. Count items
    try:
        total_items = len(data_reader)
    except:
        total_items = sum(1 for _ in data_reader)
        # Reset reader
        if split == "eval": data_reader = bagz.Reader(config.EVAL_DATA)
        elif split == "test": data_reader = bagz.Reader(config.TEST_DATA)

    # 2. Config
    CHUNK_SIZE = 1000
    NUM_WORKERS = min(16, mp.cpu_count() - 2) 
    
    output_path = config.PROCESSED_DATA_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_think_data_{split}.bagz"
    print(f"[{split}] Streaming to {output_path}...")

    ctx = mp.get_context('spawn')
    
    # -------------------------------------------------------------
    # STREAMING FIX: Open Writer Once
    # -------------------------------------------------------------
    with bagz.Writer(output_path) as writer:
        
        with ctx.Pool(
            processes=NUM_WORKERS, 
            initializer=init_worker, 
            initargs=(str(model_dir), str(metadata_path))
        ) as pool:
            
            chunks = chunked_iterable(data_reader, CHUNK_SIZE)
            results_iterator = pool.imap(process_batch, chunks)
            
            with tqdm(total=total_items, desc=f"Processing {split}", unit="rec") as pbar:
                for batch_data in results_iterator:
                    
                    # Use the new utility function to write this chunk immediately
                    bagz_utils.write_batch(writer, batch_data)
                    
                    pbar.update(len(batch_data))
                    
                    # Free memory immediately
                    del batch_data

    print(f"[{split}] Done.")

def main():
    model_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_combined_sid_alignment"
    metadata_path = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_sid_df.parquet")
    
    splits = ["eval", "test"]
    
    for split in splits:
        do_the_work_parallel(split, model_dir, metadata_path)

if __name__ == "__main__":
    main()