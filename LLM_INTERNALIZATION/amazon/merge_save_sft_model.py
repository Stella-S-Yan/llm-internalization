"""
vLLM does not support PEFT models directly, so we need to merge the adapters into the base model
before loading into vLLM.

$ python merge_save_sft_model.py --checkpoint_step 4000
"""
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import torch
import os
import argparse


base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
embedding_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_all_sid_alignment"



def main(checkpoint_step):
    seq_pred_adaptor_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_train_seq_pred_aligned_phase1" / f"checkpoint-{checkpoint_step}"
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

    # 6. Load Seq Pred adaptor
    adapter_model = PeftModel.from_pretrained(model, seq_pred_adaptor_dir)

    # Merge adapter weights as vLLM does not support PEFT models directly
    merged_model = adapter_model.merge_and_unload()  # returns standard HF model

    print(f"Restored seq pred model with extended vocabulary")

    save_dir = config.MODEL_DIR / f"{config.DATA_SOURCE}_Combined_merged_seq_pred_model"
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__=="__main__":
    # 1. Create parser
    parser = argparse.ArgumentParser(description="Example script with parameters")

    # 2. Add arguments
    parser.add_argument("--checkpoint_step", type=int, required=True, help="Step of checkpoining")

    # 3. Parse arguments
    args = parser.parse_args()

    # 4. Use arguments

    main(args.checkpoint_step)


