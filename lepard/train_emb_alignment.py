"""
Tune the embeddings of the new vocabulary, such that the embedding of a sid is closer to the embedding
of its text description.
"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import bagz_utils
import config
from utils import bagz_utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import os


MODEL_NAME = "meta-llama/Llama-3.2-1B"  #"meta-llama/Llama-3.2-1B-Instruct"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / "sid_aligned_model"
LOG_DIR = config.RUN_DIR / "sid_alignment"
BATCH_SIZE = 1024
EPOCHS = 20_000     # plateau at epoch 2k
LR = 5e-4
TEMP = 0.07


# Create an informative run name
RUN_NAME = f"sid_align_lr{LR}_temp{TEMP}_epoch{EPOCHS}_bs{BATCH_SIZE}"


def load_model_tokenizer(run_test: False):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)  # FP16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    old_vocab_size = len(tokenizer)
    print("Original vocab size:", old_vocab_size)
    prefix_tokens = [f"{prefix}{i}" for prefix in "ABCD" for i in range(256)]
    tokenizer.add_tokens(prefix_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print("Updated vocab size:", len(tokenizer))
    
    if run_test:
        text1 = "A157 B141 C28 D0"
        tokens = tokenizer.tokenize(text1)
        print("tokens: ", tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(ids)
        text_back = tokenizer.convert_tokens_to_string(tokens)
        print("text_back_from_tokens: ", text_back)
        tokens_back = tokenizer.convert_ids_to_tokens(ids)
        print("id_back_to_tokens: ", tokens_back)
        text_back = tokenizer.convert_tokens_to_string(tokens_back)
        print("id_back_from_text: ", text_back)
        
        token_id = old_vocab_size  # index you want to check
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token at index {token_id}: {token_str}")    # <A0>
        
    return model, tokenizer, old_vocab_size


def evaluate(model, tokenizer):
    model.eval()
    eval_prompts = [
        "What is the title of  SemanticID A135 B45 C199 D0 ?",
        "The product is WAWO 15 Color Professionl Makeup Eyeshadow. What is its semanticID? ",
        "What brand makes SemanticID A135 B45 C199 D0?",
        "The quick brown fox jumps over the lazy dog.",
        "What brand makes A1 B58 C120 D0?",
        "What product does the identifier A1 B58 C120 D0 refers to?",
        "In our catalog, A1 B58 C120 D0 corresponds to",
    ]

    def ask_model(prompt, max_new_tokens=50):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,    # deterministic (greedy). Use True if you want variety.
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return completion.strip()


    # 4. Run evaluation
    for q in eval_prompts:
        answer = ask_model(q)
        print(f"Q: {q}\nA: {answer}\n{'-'*40}")


class SIDDataset(Dataset):
    def __init__(self, tokenizer, split):
        self.task_name = "[SID_TASK]"
        self.tokenizer = tokenizer
        if split == "train":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_TRAIN)
        elif split == "eval":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_DEV)
        elif split == "test":
            self.df = bagz_utils.read_parquet(config.LEPARD_W_SID_TEST)

        self.data = self.df[['formatted_dest_sid', 'formatted_source_sid']].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        input = self.data[idx][0]
        target = self.data[idx][1]

        sequence = self.task_name + " " + input + " " + target
        
        seq_enc = self.tokenizer(
            sequence,
            add_special_tokens=False,
            truncation=True,
            max_length=512, 
            padding=False
        )

        input_ids = seq_enc["input_ids"]

        # --- Labels ---
        mask_start = max(0, len(input_ids) - 8)
        labels = [-100] * mask_start + input_ids[mask_start:]

        # Ensure labels same length as input_ids
        labels = labels[:len(input_ids)]

        return {
            "A_emb": input_ids,
            "sid_text": labels 
        }



class SIDRetrievalEvaluator:
    def __init__(self, dataset, model, tokenizer, device="cuda"):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_sid_embedding(self, sid_texts, normalize=True):
        """Embed a batch of SID texts using the model (last token embedding)."""
        inputs = self.tokenizer(
            sid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            last_indices = inputs["attention_mask"].sum(dim=1) - 1
            embeddings = hidden_states[torch.arange(len(sid_texts), device=self.device), last_indices]

        if normalize:
            embeddings = F.normalize(embeddings, dim=1)

        return embeddings

    def evaluate(self, topk=(1, 5, 10), num_negatives=99, sample_size=100):
        N = len(self.dataset)
        # sample `sample_size` indices without replacement
        indices = random.sample(range(N), min(sample_size, N))
        retrieval_hits = {f"top{k}_acc": 0 for k in topk}
        alignment_scores = []

        for i in indices:
            sample = self.dataset[i]
            pos_emb = sample["A_emb"].to(self.device).unsqueeze(0)  # [1, H]
            sid_text = [sample["sid_text"]]

            # SID embedding
            sid_emb = self.get_sid_embedding(sid_text)  # [1, H]

            # Cosine similarity alignment
            alignment = F.cosine_similarity(sid_emb, pos_emb).item()
            alignment_scores.append(alignment)

            # Sample negatives
            all_indices = torch.randperm(N)
            all_indices = all_indices[all_indices != i][:num_negatives].tolist()  # convert to ints
            neg_embs = torch.stack([self.dataset[j]["A_emb"] for j in all_indices]).to(self.device)

            candidates = torch.cat([pos_emb, neg_embs], dim=0)  # [num_negatives+1, H]

            sims = torch.matmul(sid_emb, candidates.T).squeeze(0)
            topk_indices = sims.topk(max(topk)).indices
            for k in topk:
                if 0 in topk_indices[:k]:
                    retrieval_hits[f"top{k}_acc"] += 1

        N_float = float(N)
        recall_results = {k: v / N_float for k, v in retrieval_hits.items()}
        mean_alignment = sum(alignment_scores) / N_float

        return mean_alignment, recall_results



def save_model(model, tokenizer, optimizer):
    save_dir = f"{MODEL_SAVE_DIR}"
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))


def load_checkpoint(save_dir):
    model = AutoModelForCausalLM.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    optimizer = torch.optim.Adam(model.get_input_embeddings().parameters(), lr=LR)
    optimizer.load_state_dict(torch.load(os.path.join(save_dir, "optimizer.pt")))



def train_sid_embeddings(model, dataset, tokenizer, old_vocab_size, writer):
    model.train()

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Use grad masking to only update new embeddings
    emb = model.get_input_embeddings()
    emb.weight.requires_grad = True

    optimizer = torch.optim.Adam([emb.weight], lr=LR)

    model.to(DEVICE)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    global_step = 0

    # Instantiate evaluator
    evaluator = SIDRetrievalEvaluator(dataset, model, tokenizer, device=DEVICE)

    best_loss = 100
    for epoch in range(EPOCHS):
        for batch in dataloader:
            # A_norm: target description embeddings (frozen), already normalized
            A_norm = batch["A_emb"].to(DEVICE)  # [B, H]
            sid_text = batch["sid_text"]       # list of SID strings

            # Tokenize SID strings
            inputs = tokenizer(sid_text, return_tensors="pt", padding=True,
                               truncation=True, max_length=10).to(DEVICE)

            # Forward pass (frozen model)
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, H]

            # Use last non-pad token for pooling (consistent with reference embeddings)
            last_indices = inputs["attention_mask"].sum(dim=1) - 1  # [B]
            C_batch = hidden_states[torch.arange(hidden_states.size(0)), last_indices]
            
            # Normalize embeddings
            C_norm = F.normalize(C_batch, dim=1)

            # Contrastive loss
            logits1 = (A_norm @ C_norm.T) / TEMP
            logits2 = (C_norm @ A_norm.T) / TEMP
            labels = torch.arange(logits1.size(0), device=DEVICE)

            loss1 = F.cross_entropy(logits1, labels)
            loss2 = F.cross_entropy(logits2, labels)
            loss = (loss1 + loss2) / 2

            optimizer.zero_grad()
            loss.backward()

            # Mask: zero out grads for old tokens
            with torch.no_grad():
                if emb.weight.grad is not None:
                    emb.weight.grad[:old_vocab_size] = 0

            optimizer.step()

            # Evaluation
            if global_step > 0 and global_step % 50 == 0:
                # log
                print(f"Step {global_step}- train/loss: {loss.item()}")

                model.eval()
                with torch.no_grad():
                    mean_alignment, recall = evaluator.evaluate(topk=(1, 5, 10), num_negatives=99)
                print(f"Step {global_step}: Alignment={mean_alignment:.4f}, Recall@1={recall['top1_acc']:.4f}, Recall@5={recall['top5_acc']:.4f}")
                writer.add_scalar("eval/alignment", mean_alignment, global_step)
                writer.add_scalar("eval/recall@1", recall["top1_acc"], global_step)
                writer.add_scalar("eval/recall@5", recall["top5_acc"], global_step)
                writer.add_scalar("eval/recall@10", recall["top10_acc"], global_step)
                model.train()
                
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
        # Save checkpoint
        if loss.item() < best_loss:
            best_loss = loss.item()
            if epoch > 300 and epoch % 10 == 0:
                save_model(model, tokenizer, optimizer)


def main():
    
    writer = SummaryWriter(log_dir=f"{LOG_DIR}/{RUN_NAME}")

    model, tokenizer, old_vocab_size = load_model_tokenizer(run_test=True)
    evaluate(model, tokenizer)
    
    # print(model.config.pad_token_id)  # likely None
    
    # df = bagz_utils.read_parquet(config.META_W_SID)
    # print(df.head(3))
    # dataset = SIDDataset(df, tokenizer)
    
    # # Train
    # train_sid_embeddings(model, dataset, tokenizer, old_vocab_size, writer)

    # # Save fine tuned model
    # model.save_pretrained(MODEL_SAVE_DIR)
    # # Save tokenizer (with new SID tokens)
    # tokenizer.save_pretrained(MODEL_SAVE_DIR)
    
    
if __name__ == "__main__":
    main()
