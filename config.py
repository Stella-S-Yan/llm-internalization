"""Configuration for data paths and model parameters."""

import datetime
import os
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
MODEL_DIR = DATA_DIR / "model"
RUN_DIR = DATA_DIR / "runs"

DATA_SOURCE = "Lepard"
REVIEW_TYPE = "one"

# DATA_SOURCE = "Amazon"
# REVIEW_TYPE = "Sports_and_Outdoors"  # 2 "Sports_and_Outdoors"  1 "Toys_and_Games"  3 "Beauty"

time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(BASE_DIR, "data", "tensorboard", time_str)

AMAZON_REVIEW_DATASET = os.path.join(DATA_DIR, DATA_SOURCE, f"reviews_{REVIEW_TYPE}_5.json") 
AMAZON_META_DATASET = os.path.join(DATA_DIR, DATA_SOURCE, f"meta_{REVIEW_TYPE}.json")

META_NORMALIZED = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_meta_norm_df.bagz")
META_OUTSIDE_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_meta_outside_emb_df.bagz")
META_TWO_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_meta_two_emb_df.bagz")
META_ALL_SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_embed_all_text_meta_df.bagz")


META_W_ALL_EMBEDDING = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_embedding_all_text_meta_df.bagz")
META_W_ALL_TWO_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_embedding_all_two_text_meta_df.bagz")
ALL_RQVAE_CHECKPOINT_DIR= os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_all_rqvae")



REVIEW_ID_DF = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_review_id_df.bagz" )
USER_SEQUENCE = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_sequence.bagz" )
USER_SID_SEQUENCE = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_sid_sequence.bagz" )
USER_UID_SID_SEQUENCE = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_uid_sid_sequence.bagz" )


TRAIN_EVAL_DATA = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_train_eval.bagz" )
TRAIN_DATA = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_train.bagz" )
EVAL_DATA = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_eval.bagz" )
TEST_DATA = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_test.bagz" )


USER_NEG = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user_neg.bagz" )
USER2HASHED = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_user2hashed.bagz" )

META_W_EMBEDDING = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_embedding_text_meta_df.bagz")
META_W_SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_embedding_text_meta_df.bagz")
META_W_EMB_SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_two_embed_meta_df.bagz")


LEPARD_DEST_DF = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_dest_row_id_df.parquet")
LEPARD_QUOTE_DF = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_quote_passage_id_df.parquet")
LEPARD_OUTSIDE_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_outside_emb")
LEPARD_LLM_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_llm_emb")
LEPARD_TWO_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_two_emb_df.bagz")
LEPARD_SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_df.parquet")
LEPARD_TRAIN = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_train_df.parquet")
LEPARD_EVAL = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_eval_df.parquet")
LEPARD_TEST = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_test_df.parquet")


META_W_TEXT = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_text_meta_df.bagz")
RQVAE_CHECKPOINT_DIR= os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_rqvae")
TRAIN_LOSS_PLOT = os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_rqvae_train.png")
MAX_HISTORY_LEN = 20
MIN_HISTORY_LEN = 2
MAX_DEC_LEN = 1
PAD_TOKEN = 0
MAX_EVAL_LEN = 10



