import os
import torch
import logging
import psutil
import multiprocessing
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils import bagz_utils  # your existing utils
import config      # your config for META_OUTSIDE_EMB

logger = logging.getLogger(__name__)

def gen_embedding(meta_df):
    logger = logging.getLogger(__name__)

    # Restrict to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    model_name = "thenlper/gte-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # evaluation mode

    # RAM & CPU info
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    logger.debug(f"Total RAM: {ram_gb:.2f} GB")
    logger.debug(f"Logical cores: {multiprocessing.cpu_count()}")

    # Embedding generation
    logger.debug(f"--- Num of items to embed: {meta_df.shape[0]}")
    batch_size = 1024  # adjust based on GPU memory
    embeddings = []
    max_len = 512  # GTE-large max sequence length

    for i in tqdm(range(0, len(meta_df), batch_size), desc="Embedding batches"):
        batch_texts = meta_df["formatted_text"].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # mean pooling over tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden_states = outputs.last_hidden_state  # [B, seq_len, H]
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            seq_lengths = attention_mask.sum(dim=1)
            batch_embeds = sum_hidden / seq_lengths
            batch_embeds = torch.nn.functional.normalize(batch_embeds, dim=1)
            embeddings.extend(batch_embeds.cpu().tolist())

    # Save embeddings to dataframe
    meta_df["gte_embed"] = embeddings
    bagz_utils.save_parquet(meta_df, config.META_OUTSIDE_EMB)
    logger.debug(meta_df.head(3))




def do_the_work():
    meta_df = bagz_utils.read_parquet(config.META_NORMALIZED)
    gen_embedding(meta_df)


if __name__=="__main__":

    do_the_work()