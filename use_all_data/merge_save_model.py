import config
from utils import merge_save_load_model


# ----------  Merge & save think_SFT model ----------------

model_input_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_all_sid_aligned_model"
adaptor_dir = config.MODEL_DIR / f'{config.DATA_SOURCE}_{config.REVIEW_TYPE}_train_thinking_sft_adaptor'
checkpoint_dir = "checkpoint-50001"
model_save_dir = config.MODEL_DIR / f'{config.DATA_SOURCE}_{config.REVIEW_TYPE}_think_model_sft'


# ----------  Merge & save think_CRPO model ----------------

# model_input_dir = config.MODEL_DIR / "think_model_sft"
# adaptor_dir = config.MODEL_DIR / "train_think_grpo_adaptor"
# checkpoint_dir = "checkpoint-1040"
# model_save_dir = config.MODEL_DIR / "think_model_grpo"



merge_save_load_model.merge_and_save_model(
    model_input_dir=model_input_dir,
    adaptor_dir=adaptor_dir,
    checkpoint_dir=checkpoint_dir,
    model_save_dir=model_save_dir
)



