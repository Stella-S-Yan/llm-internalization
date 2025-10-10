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

DATA_SOURCE = "Amazon"
REVIEW_TYPE = "Beauty"

time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(BASE_DIR, "data", "tensorboard", time_str)

AMAZON_REVIEW_DATASET = os.path.join(DATA_DIR, DATA_SOURCE, f"reviews_{REVIEW_TYPE}_5.json") 
AMAZON_META_DATASET = os.path.join(DATA_DIR, DATA_SOURCE, f"meta_{REVIEW_TYPE}.json")

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
META_W_LLAMA_EMBEDDING = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_llama_embedding_text_meta_df.bagz")
META_W_SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_embedding_text_meta_df.bagz")

SID2ITEM = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid2item.bagz")
ITEM2SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_item2sid.bagz")

TOKENIZER = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_tokenizer.bagz")
TOKENIZER_TXT = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_tokenizer.bagz")

IID2EMBEDDING = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_iid_to_embedding.bagz")

SID_MATRIX = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_matrix.bagz")
META_W_TEXT = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_text_meta_df.bagz")
RQVAE_CHECKPOINT_DIR= os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_rqvae")
RQVAE_CHECKPOINT_LLAMA_DIR= os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_rqvae_llama")
TRAIN_LOSS_PLOT = os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_rqvae_train.png")
MAX_HISTORY_LEN = 20
MIN_HISTORY_LEN = 2
MAX_DEC_LEN = 1
PAD_TOKEN = 0
