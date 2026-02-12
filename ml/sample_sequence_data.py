import torch
from torch.utils.data import Dataset
import random
import re
import pandas as pd
from collections import Counter
import os
import bisect

import config
from utils import bagz_utils


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


class SampleSeqDataset(Dataset):
    def __init__(self):
        self.eos_token = "<|eot_id|>"

        self.data = []
        
        SEQUENCE_file = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_user_sequence.bagz" )
        records = bagz_utils.read_record(SEQUENCE_file)
        print(f"--- {config.REVIEW_TYPE} # records: {len(records)}")
        for r in records:
            self.data.append((r, config.REVIEW_TYPE))

        # Read in df with sid, meta data
        file_name = os.path.join(config.PROCESSED_DATA_DIR, f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_sid_df.parquet")
        df = pd.read_parquet(file_name) 
        df["year"] = df["Title"].str.extract(r"\((\d{4})\)")

        # Create lookups
        self.sid_to_genre = dict(zip(df['sid'], df['Genre']))
        self.sid_to_year = dict(zip(df['sid'], df['year']))

    def __len__(self):
        return 3_000_000  # or any large number

    # def set_epoch(self, epoch):
        # random.seed(411 + epoch)

    def __getitem__(self, idx):
        record, review_type = random.choice(self.data)

        uid = record["reviewerID"]

        input_seq_d, target_sid = self._sample_subsequence(record)

        sids = PATTERN.findall(input_seq_d)
        if any(x != x for x in sids):
            print(f"uid")
            print("here")


        freq_A = self._most_frequent_Ax(sids)

        sid_year_genre_list = "\n".join(
            f"{sid} : {self.sid_to_year.get(sid)}, {self.sid_to_genre.get(sid)}"
            for sid in sids
        )

        target_year = self.sid_to_year.get(target_sid)
        target_genre = self.sid_to_genre.get(target_sid)

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
            eos=self.eos_token
        ).strip()

        solution = {
            "freq": freq_A if freq_A else 'None',
            "year": target_year,
            "genre": target_genre,
            "sid": target_sid,
        }

        return {
            'prompt': prompt,
            'target': target,
            'solution': solution
        }
        

    def _sample_subsequence(self, record):
        seq = record["sequence"][:-2]
        n = len(seq)

        if n < config.MIN_HISTORY_LEN + 1:
            return None, None  # or handle separately

        # 1️. sample end position uniformly
        end = random.randint(config.MIN_HISTORY_LEN, n)

        # 2️. sample window length
        L = random.randint(
            config.MIN_HISTORY_LEN,
            min(config.MAX_HISTORY_LEN, end)
        )

        # 3️. compute start so window ends at `end`
        start = end - L
        res_seq = seq[start:end]

        input_seq_d, target_seq_d = self._make_data_point(res_seq)
        return input_seq_d, target_seq_d
        


    def _make_data_point(self, seq):
        target = seq[-1]
        input = seq[:-1]

        input_seq_str = ' '.join(
            item if isinstance(item, str) else ' '.join(item)
            for item in input
        )

        return input_seq_str, target
    

    def _most_frequent_Ax(self, ids):
        if not ids: return None
        ax_list = [x.split(maxsplit=1)[0] for x in ids]
        counts = Counter(ax_list)
        if not counts: return None
        most_common = counts.most_common(2)
        best, freq = most_common[0]
        if len(most_common) > 1 and most_common[1][1] == freq: return None
        return best

    

def sample_seq_collator(batch, tokenizer):
    """
    Correct collator for batching prompt + target for causal LM.
    """
    # evaluation data
    if "input_ids" in batch[0]: 
        input_ids = [f["input_ids"] for f in batch]
        labels = [f["labels"] for f in batch]
        
        # Pad sequences to the same length (left padding can be added if needed)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100, padding_side="left")

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    # Training data
    prompts = [ex["prompt"] for ex in batch]
    targets = [ex["target"] for ex in batch]

    # Tokenize prompts
    prompt_enc = tokenizer(
        prompts,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )

    # Tokenize targets
    target_enc = tokenizer(
        targets,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )

    # Prepare input_ids, labels
    input_ids_list = []
    labels_list = []

    for p_ids, t_ids in zip(prompt_enc["input_ids"], target_enc["input_ids"]):
        input_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    # Pad sequences to the same length (left padding can be added if needed)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100, padding_side="left")

    return {
            'input_ids': input_ids,
            'labels': labels,
        }