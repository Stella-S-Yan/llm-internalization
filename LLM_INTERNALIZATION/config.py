"""Configuration for data paths and model parameters."""

import os
import pathlib


os.environ["PROJECT_WORKSPACE"] = "/usr/local/google/home/stellasyan/Documents/workspace" # "/path/to/project-workspace"

BASE_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE_DIR = pathlib.Path(
    os.environ.get("PROJECT_WORKSPACE")
)
DATA_DIR = WORKSPACE_DIR / "data"
PROCESSED_DATA_DIR = WORKSPACE_DIR / "processed_data"
MODEL_DIR = WORKSPACE_DIR / "model"
RUN_DIR = WORKSPACE_DIR / "runs"

DATA_SOURCE = "MovieLens"
REVIEW_TYPE = "1m"

# DATA_SOURCE = "Lepard"
# REVIEW_TYPE = "10k"

# DATA_SOURCE = "Amazon"
# REVIEW_TYPE =  "Beauty"   


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
LEPARD_SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_df.parquet")

LEPARD_TRAIN = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_train_df.parquet")
LEPARD_EVAL = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_eval_df.parquet")
LEPARD_TEST = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_test_df.parquet")

LEPARD_50k_TRAIN = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_50k_train_df.parquet")
LEPARD_50k_EVAL = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_50k_eval_df.parquet")
LEPARD_50k_TEST = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_50k_test_df.parquet")

LEPARD_20k_EVAL = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_20k_eval_df.parquet")
LEPARD_20k_TEST = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_20k_test_df.parquet")

LEPARD_10k_EVAL = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_10k_eval_df.parquet")
LEPARD_10k_TEST = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_10k_test_df.parquet")


ML_DF = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_row_id_df.parquet")
ML_OUTSIDE_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_outside_emb")
ML_LLM_EMB = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_llm_emb")
ML_SID = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_sid_df.parquet")

ML_TRAIN = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_50k_train_df.parquet")
ML_EVAL = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_50k_eval_df.parquet")
ML_TEST = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_50k_test_df.parquet")



META_W_TEXT = os.path.join(PROCESSED_DATA_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_text_meta_df.bagz")
RQVAE_CHECKPOINT_DIR= os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_rqvae")
TRAIN_LOSS_PLOT = os.path.join(MODEL_DIR, f"{DATA_SOURCE}_{REVIEW_TYPE}_rqvae_train.png")

if DATA_SOURCE == 'Amazon':
    MAX_HISTORY_LEN = 50  # Amazon
    MIN_HISTORY_LEN = 2
elif DATA_SOURCE == 'MovieLens':
    MAX_HISTORY_LEN = 30  # MovieLens
    MIN_HISTORY_LEN = 2


if DATA_SOURCE == "Amazon":
    HP = {
        "training": {
            "total_steps": 30_000,
            "warmup_steps": 3_000,
        },
        "learning_rate_schedule": {
            "init_value": 0.0,
            "peak_value": 1e-3,
            "end_value": 1e-5,
        },
        "optimizer": {
            "type": "adamw",
            "weight_decay": 0.055,
        },
        "vqvae": {
            "num_embeddings": 256,
            "embedding_dim": 16,
            "ema_decay": 0.99,
            "commitment_cost": 1.5,
            # data_variance omitted because dynamic
        }
    }
    CODEBOOK_PCT = 0.75 
elif DATA_SOURCE == "MovieLens":
    BATCH_SIZE = 512
    LR = 1e-3
    TOTAL_STEPS = 4_000
    TEMP = 0.2
    WARM_UP = 400
    HP = {
        "training": {
            "total_steps": 20_000, #20_000,
            "warmup_steps": 2_000,
        },
        "learning_rate_schedule": {
            "init_value": 0.0,
            "peak_value": 1e-3,  
            "end_value": 5e-5,
        },
        "optimizer": {
            "type": "adamw",  # or "adagrad"
            "weight_decay": 0.055,
        },
        "vqvae": {
            "num_embeddings": 256,
            "embedding_dim": 16,
            "ema_decay": 0.99,          
            "commitment_cost": 1.5,     
        }
    }
    CODEBOOK_PCT = 0.7
elif DATA_SOURCE == "Lepard":
    BATCH_SIZE = 128
    LR = 6e-3
    TOTAL_STEPS = 8_000
    TEMP = 0.2
    WARM_UP = 400
    if REVIEW_TYPE == "20k": 
        HP = {
            "training": {
                "total_steps": 20_000, 
                "warmup_steps": 2_000,
            },
            "learning_rate_schedule": {
                "init_value": 0.0,
                "peak_value": 1e-3,  
                "end_value": 1e-5,
            },
            "optimizer": {
                "type": "adamw",  
                "weight_decay": 0.055,
            },
            "vqvae": {
                "num_embeddings": 256,
                "embedding_dim": 16,
                "ema_decay": 0.99,          # lower value makes code book adaptation faster, can cause instability, so training takes longer to converge
                "commitment_cost": 0.1,     # Increase commitment_cost will depress quant_loss
            }
        }
        CODEBOOK_PCT = 0.95
    elif REVIEW_TYPE == "10k":
        HP = {
            "training": {
                "total_steps": 20_000, 
                "warmup_steps": 3_000,
            },
            "learning_rate_schedule": {
                "init_value": 0.0,
                "peak_value": 1e-3,  
                "end_value": 1e-5,
            },
            "optimizer": {
                "type": "adamw",  
                "weight_decay": 0.055,
            },
            "vqvae": {
                "num_embeddings": 256,
                "embedding_dim": 16,
                "ema_decay": 0.99,          # lower value makes code book adaptation faster, can cause instability, so training takes longer to converge
                "commitment_cost": 0.1,     # Increase commitment_cost will depress quant_loss
            }
        }
        CODEBOOK_PCT = 0.92