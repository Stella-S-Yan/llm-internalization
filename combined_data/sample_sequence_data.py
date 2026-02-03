"""
Pregenerate all possible subsequences and create a fixed training dataset
"""

import config
from utils import bagz_utils
import random
from torch.utils.data import Dataset
from collections import Counter
import torch
import re
import bisect


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

PATTERN = re.compile(r"A\d+\s+B\d+\s+C\d+\s+D\d+")


class SampleSeqDataset(Dataset):
    def __init__(self):
        self.eos_token = "<|eot_id|>"
        self.users = []
        self.cum_sizes = []

        total = 0
        self.sid_mapping = {}
        self.data = []
        review_types = ["Toys_and_Games", "Sports_and_Outdoors", "Beauty"]
        for review_type in review_types:
            SEQUENCE_file = config.PROCESSED_DATA_DIR / f"Amazon_{review_type}_user_sequence.bagz"
            records = bagz_utils.read_record(SEQUENCE_file)
            print(f"--- {review_type} # records: {len(records)}")
            for r in records:
                # Training seq (remove Val and Test items)
                seq_len = len(r["sequence"]) - 2 
                
                # FIX 1: Ensure user has enough history for at least one window
                if seq_len < config.MIN_HISTORY_LEN:
                    continue

                # FIX 2: Correct count (Inclusive math)
                # If len=5, min=3 -> windows of len 3,4,5 -> Count is 3
                num_windows = seq_len - config.MIN_HISTORY_LEN + 1
                
                total += num_windows
                
                self.data.append((r, review_type))
                # Store seq_len 'n' to help debug or boundaries if needed
                self.users.append((r, review_type, seq_len)) 
                self.cum_sizes.append(total)

            meta_df = bagz_utils.read_parquet(config.META_ALL_SID) 
            sid_to_cat = dict(zip(meta_df['formatted_sid'], meta_df['fine_category']))
            self.sid_mapping[review_type] = sid_to_cat
                


    def __len__(self):
        return 2_000_000


    def __getitem__(self, idx):

        record, review_type, input_seq_d, target_sid = self._sample_subsequence()

        uid = record["id"]
        if review_type == "Beauty":
            uid = f"B_{uid}"
        elif review_type == "Toys_and_Games":
            uid = f"T_{uid}"
        elif review_type == "Sports_and_Outdoors":
            uid = f"S_{uid}"

        sids = PATTERN.findall(input_seq_d)
        if any(x != x for x in sids):
            print(f"uid")
            print("here")

        freq_A = self._most_frequent_Ax(sids)

        sid_cat_list = "\n".join(
            f"{sid} : {self.sid_mapping[review_type].get(sid)}"
            for sid in sids
        )

        target_cat = self.sid_mapping[review_type].get(target_sid)

        prompt = PROMPT_TEMPLATE.format(
            uid=uid,
            sid_cat_list=sid_cat_list,
            predict=""
        ).strip()

        target = TARGET_TEMPLATE.format(
            freq_A=freq_A,
            target_sid_cat=target_cat,
            target_sid=target_sid,
            eos=self.eos_token
        ).strip()

        solution = {
            "freq": freq_A if freq_A else 'None',
            "cat": target_cat,
            "sid": target_sid,
            "uid": uid,
        }

        return {
            'prompt': prompt,
            'target': target,
            'solution': solution
        }


    def _sample_subsequence(self):
        # 1. Sample global position
        x = random.randint(0, self.cum_sizes[-1] - 1)
        
        # 2. Find user
        i = bisect.bisect_right(self.cum_sizes, x)
        record, review_type, n = self.users[i]

        # 3. Compute local 'end' position
        prev = self.cum_sizes[i-1] if i > 0 else 0
        offset = x - prev  # 0 to (num_windows - 1)
        
        # FIX 3: Correct Start Position
        # If offset=0, we want the shortest valid window (MIN_LEN)
        # So 'end' index should be MIN_LEN
        # Example: seq=[0,1,2,3,4], MIN=3.
        # offset=0 -> end=3 -> seq[:3] -> [0,1,2] (Len 3)
        end = config.MIN_HISTORY_LEN + offset
        
        # 4. Greedy Length (Max Context)
        # Since 'end' grows as 'offset' grows, this automatically
        # creates the "Sliding Window" effect.
        # We take as much history as possible ending at 'end'.
        L = min(config.MAX_HISTORY_LEN, end)
        
        start = end - L
        
        # Slice from the training sequence (remove last 2 items)
        seq = record['sequence'][:-2]
        res_seq = seq[start:end]

        return record, review_type, *self._make_data_point(res_seq)


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