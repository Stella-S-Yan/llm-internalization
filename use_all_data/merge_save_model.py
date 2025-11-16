import config
from utils import merge_save_load_model


# Merge & save think_SFT model

model_input_dir = config.MODEL_DIR / "all_sid_aligned_model"
adaptor_dir = config.MODEL_DIR / "train_thinking_sft"
checkpoint_dir = "checkpoint-50001"
model_save_dir = config.MODEL_DIR / "think_model_sft"

merge_save_load_model.merge_and_save_model(
    model_input_dir=model_input_dir,
    adaptor_dir=adaptor_dir,
    checkpoint_dir=checkpoint_dir,
    model_save_dir=model_save_dir
)

