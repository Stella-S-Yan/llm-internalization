"""
Compute recall@5 if always predict the 5 most popular next items. 
Upper-bound check
"""

import config
from collections import Counter
from utils import bagz_utils
import random


def topk_sid_baseline():
    records = bagz_utils.read_record(config.USER_SEQUENCE)
    records[0]

    # Step 1: Extract the last element of asin_sequence for each user
    last_sids = [record['sequence'][-1] for record in records]

    # Step 2: Count frequency of each last ASIN
    last_sid_counter = Counter(last_sids)

    # Step 3: Get top-k most popular last ASINs
    k = 5
    top_k_last_sids = [sid for sid, _ in last_sid_counter.most_common(k)]
    print("Top-k most popular last sids:", top_k_last_sids)

    # Step 4: Compute recall@k
    # Percentage of users whose last ASIN is in the top-k
    hits = sum(1 for sid in last_sids if sid in top_k_last_sids)
    recall_at_k = hits / len(last_sids)

    print(f"Recall@{k} by sid: {recall_at_k:.4f}")


def topk_asin_baseline():
    records = bagz_utils.read_record(config.USER_SEQUENCE)
    records[0]

    # Step 1: Extract the last element of asin_sequence for each user
    last_asins = [record['asin_sequence'][-1] for record in records]

    # Step 2: Count frequency of each last ASIN
    last_asin_counter = Counter(last_asins)

    # Step 3: Get top-k most popular last ASINs
    k = 5
    top_k_last_asins = [asin for asin, _ in last_asin_counter.most_common(k)]
    print("Top-k most popular last ASINs:", top_k_last_asins)

    # Step 4: Compute recall@k
    # Percentage of users whose last ASIN is in the top-k
    hits = sum(1 for asin in last_asins if asin in top_k_last_asins)
    recall_at_k = hits / len(last_asins)

    print(f"Recall@{k} by asin: {recall_at_k:.4f}")


def random_k_baseline():
    records = bagz_utils.read_record(config.USER_SEQUENCE)
    records[0]
    # Step 1: Extract the last element of asin_sequence for each user
    last_asins = [record['asin_sequence'][-1] for record in records]

    # Step 2: Candidate pool is all last ASINs
    candidate_pool = last_asins.copy()

    # Step 3: Randomly pick top-k ASINs for each user
    k = 5
    hits = 0
    for last_asin in last_asins:
        # Randomly sample k ASINs from the pool
        random_top_k = random.sample(candidate_pool, k)
        if last_asin in random_top_k:
            hits += 1

    # Step 4: Compute recall@k
    recall_at_k = hits / len(last_asins)
    print(f"Random Recall@{k}: {recall_at_k:.4f}")

if __name__ == "__main__":
    topk_sid_baseline()
    topk_asin_baseline()
    random_k_baseline()

