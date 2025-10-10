import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import math
from utils import bagz_utils
from generative_rec import config
import pandas as pd
from utils import bagz_utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# CONFIG
# -------------------------
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or your pretrained LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 150
LR = 5e-5
TEMP = 0.03

SAVE_DIR = "sid_aligned_model"



class SIDDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        A_emb = torch.tensor(row["embedding"], dtype=torch.float)   # precomputed A
        sid_text = row["formatted_sid"]                             # e.g. "A67 B111 C56 D0"
        sid_tokens = sid_text.strip().split()     
        sid_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in sid_tokens]
        return {
            "A_emb": A_emb,
            "sids": torch.tensor(sid_ids, dtype=torch.long)
        }


def evaluate_retrieval(model, dataset):
    model.eval()
    emb = model.get_input_embeddings()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    A_all, C_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            A_batch = batch["A_emb"]                # (B,H)
            sid_ids = batch["sids"].to(DEVICE)   # (B,L)
            C_batch = emb(sid_ids).mean(dim=1).cpu()

            A_all.append(A_batch)
            C_all.append(C_batch)

    A_all = torch.cat(A_all, dim=0)
    C_all = torch.cat(C_all, dim=0)

    # Normalize
    A_norm = F.normalize(A_all, dim=1)
    C_norm = F.normalize(C_all, dim=1)

    sims = A_norm @ C_norm.T
    preds = sims.argmax(dim=1)
    labels = torch.arange(len(dataset))
    
    # Top-k accuracy
    accs = {}
    for k in [1, 5, 10]:
        topk = sims.topk(k, dim=1).indices
        correct = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
        accs[f"top{k}"] = correct
    
    
    return accs
    

def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

    # 2. Load the base LLaMA model (original checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        # torch_dtype="auto",
        torch_dtype=torch.float32,
    )

    # 3. Resize the token embeddings to match the tokenizer (includes your new SID tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 4. Load the fine-tuned LoRA + embedding weights
    model = PeftModel.from_pretrained(model, SAVE_DIR, device_map="auto")

    # 5. Put in eval mode
    model.eval()
    return model, tokenizer
    

def main():
    
    model, tokenizer = load_model_tokenizer()
    model.to(DEVICE)
    
    df = bagz_utils.read_parquet(config.META_W_EXT_EMBEDDING)
    print(df.head(3))
    dataset = SIDDataset(df, tokenizer)
    
    acc = evaluate_retrieval(model, dataset)
    print("Retrieval Accuracies: ", accs)
    
        
if __name__ == "__main__":
    main()
