"""
vLLM does not support PEFT models directly, so we need to merge the adapters into the base model
before loading into vLLM.

$ python merge_save_sft_think_model.py --checkpoint_step 35000
"""
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import torch
import os
import argparse


base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
embedding_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_all_sid_alignment"

def main(check_point, run_num):

    think_sft_adaptor_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_think_sft_adaptor_{run_num}" / f"checkpoint-{check_point}"

    # Load BASE MODEL again — quantized or FP16 as desired
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # dtype=torch.bfloat16,   # or fp16, or load_in_4bit=True
        dtype=torch.float32,
    )

    # 2. Load extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_dir)

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)

    # 3. Resize embedding table
    model.resize_token_embeddings(new_vocab_size)

    # 4. Load saved new embedding weights
    new_emb = torch.load(os.path.join(embedding_dir, "new_embeddings.pt")).to(model.device)
    print(f"new_emb device: {model.device}")

    # 5. Insert the new embeddings back into the table
    with torch.no_grad():
        model.get_input_embeddings().weight[old_vocab_size:] = new_emb

    print(f"Restored model with extended vocab ({new_vocab_size} tokens)")

    # 6. Load Think SFT adaptor
    adapter_model = PeftModel.from_pretrained(model, think_sft_adaptor_dir)

    # Merge adapter weights as vLLM does not support PEFT models directly
    merged_model = adapter_model.merge_and_unload()  # returns standard HF model

    # --- redundant, but can keep ----
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("--- Tokenizer: ", tokenizer.pad_token, tokenizer.pad_token_id)
    model.config.pad_token = tokenizer.pad_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side='left'

    print(model.config)
    # ------------------------------------
    print(f"Restored think SFT model with extended vocabulary")

    save_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_{config.REVIEW_TYPE}_merged_think_sft_model_{run_num}"
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__=="__main__":
    # 1. Create parser
    parser = argparse.ArgumentParser(description="Example script with parameters")

    # 2. Add arguments
    parser.add_argument("--RUN_NUM", type=int, default=0, help="Run index")
    parser.add_argument("--CHECK_POINT", type=int, default=0, help="Step of checkpoining")

    # 3. Parse arguments
    args = parser.parse_args()

    # args.CHECK_POINT = 290000
    # 4. Use arguments

    main(args.CHECK_POINT, args.RUN_NUM)