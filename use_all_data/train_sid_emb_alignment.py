"""
Tune the embeddings of the new vocabulary, such that the embedding of a sid is closer to the embedding
of its text description.

Have very few trainable parameters (1024 * hidden_dim). 
The objective (contrastive loss) is smooth and well-behaved. 
The model's other weights are frozen, so the optimization surface doesn't shift. 
So a complex LR schedul is not necessary, but a gental warmup and/or decay can still help stabilize early updates and avoid overshooting
"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from utils import bagz_utils
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import os
import itertools

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = config.MODEL_DIR / "all_sid_aligned_model"
LOG_DIR = config.RUN_DIR / "all_sid_alignment"
BATCH_SIZE = 1024
TOTAL_STEPS = 12_000     # plateau at epoch 2k
LR =  1e-3         #  1e-3 best, but then overfit 
TEMP = 0.1     # high temperature: smoother distribution, softer gradients
SCALE = 0.01    # Best


# Create an informative run name
RUN_NAME = f"sid_align_lr{LR}_temp{TEMP}_total_steps{TOTAL_STEPS}_bs{BATCH_SIZE}_scale{SCALE}"


def load_model_tokenizer(run_test: False):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)  # FP16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    
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


class SIDDataset(Dataset):
    def __init__(self):
        self.data = bagz_utils.read_parquet(config.META_W_ALL_SID.value)[["llama_embedding", "formatted_sid"]].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "A_emb": torch.tensor(item[0], dtype=torch.float32),
            "sid": item[1]
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

    def evaluate(self, topk=(1, 5, 10), num_negatives=99, sample_size=1000, batch_size=64):
        """
        Batched evaluation of SID embeddings against text embeddings.
        Returns Recall@K, mean alignment, and NDCG@K.

        Args:
            topk: tuple of top-K for recall
            num_negatives: number of negative text embeddings per SID
            sample_size: number of SIDs to evaluate
            batch_size: batch size for SID embedding computation
        Returns:
            mean_alignment: float, 1.0: perfect alignment, 0: orthogonal unrelated, -1: opposite direction
            recall_results: dict
            ndcg_results: dict
        """
        N = len(self.dataset)
        # Sample evaluation indices
        eval_indices = random.sample(range(N), min(sample_size, N))

        # Fixed negative pool
        all_indices = list(range(N))
        random.shuffle(all_indices)
        neg_pool = all_indices[:num_negatives]
        neg_embs = torch.stack([self.dataset[j]["A_emb"] for j in neg_pool]).to(self.device)

        retrieval_hits = {f"top{k}_acc": 0 for k in topk}
        alignment_scores = []
        ndcg_accum = {f"ndcg@{k}": 0.0 for k in topk}

        # Process in batches
        for i in range(0, len(eval_indices), batch_size):
            batch_indices = eval_indices[i:i+batch_size]
            B = len(batch_indices)

            # Positive embeddings
            pos_embs = torch.stack([self.dataset[j]["A_emb"] for j in batch_indices]).to(self.device)

            # SID embeddings
            sid_texts = [self.dataset[j]["sid"] for j in batch_indices]
            sid_embs = self.get_sid_embedding(sid_texts)  # [B, H]

            # Alignment
            batch_alignment = F.cosine_similarity(sid_embs, pos_embs, dim=1)
            alignment_scores.extend(batch_alignment.cpu().tolist())

            # Candidates: positive + negatives
            candidates_per_sample = torch.cat([
                pos_embs.unsqueeze(1),                     # [B, 1, H]
                neg_embs.unsqueeze(0).expand(B, -1, -1)   # [B, num_neg, H]
            ], dim=1)  # [B, num_neg+1, H]

            sims = torch.bmm(sid_embs.unsqueeze(1), candidates_per_sample.transpose(1, 2)).squeeze(1)  # [B, num_neg+1]

            # Top-K
            maxk = max(topk)
            topk_vals, topk_inds = sims.topk(maxk, dim=1)

            # Recall@K
            for k in topk:
                hits = (topk_inds[:, :k] == 0).sum().item()  # 0 is positive
                retrieval_hits[f"top{k}_acc"] += hits

            # NDCG@K
            for k_val in topk:
                relevance = (topk_inds[:, :k_val] == 0).float()  # [B, k_val]
                discounts = torch.log2(torch.arange(2, k_val + 2, device=relevance.device).float())
                dcg = (relevance / discounts).sum(dim=1)
                idcg = torch.tensor([1.0], device=relevance.device)
                ndcg_accum[f"ndcg@{k_val}"] += dcg.sum().item()

        N_float = float(len(eval_indices))
        recall_results = {k: v / N_float for k, v in retrieval_hits.items()}
        mean_alignment = sum(alignment_scores) / N_float
        ndcg_results = {k: v / N_float for k, v in ndcg_accum.items()}

        return mean_alignment, recall_results, ndcg_results



def save_model(model, tokenizer, optimizer, epoch=None, global_step=None):
    save_dir = MODEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step
    }, os.path.join(save_dir, "training_state.pt"))


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
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=RandomSampler(dataset),  # ensures shuffle across all files
        num_workers=4,
        pin_memory=True,
        drop_last=False
        )


    # Instantiate evaluator
    evaluator = SIDRetrievalEvaluator(dataset, model, tokenizer, device=DEVICE)

    best_recall = 0.0
    data_iter = itertools.cycle(dataloader)  # infinite iterator over dataloader

    for global_step in range(TOTAL_STEPS):
        if global_step == 400:
            for g in optimizer.param_groups:
                g["lr"] *= SCALE

        batch = next(data_iter)  # sample one batch per step
        
        # A_norm: target description embeddings (frozen), already normalized
        A_norm = batch["A_emb"].to(DEVICE)  # [B, H]
        sid_text = batch["sid"]       # list of SID strings

        # Tokenize SID strings
        inputs = tokenizer(sid_text, return_tensors="pt", padding=True,
                            truncation=True, max_length=8).to(DEVICE)

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
                mean_alignment, recall, ndcg = evaluator.evaluate(topk=(1, 5, 10), num_negatives=99)
            print(f"Step {global_step}: Alignment={mean_alignment:.4f}, Recall@1={recall['top1_acc']:.4f}, Recall@5={recall['top5_acc']:.4f}, Recall@10={recall['top10_acc']:.4f}")
            writer.add_scalar("eval/alignment", mean_alignment, global_step)
            writer.add_scalar("eval/recall@1", recall["top1_acc"], global_step)
            writer.add_scalar("eval/recall@5", recall["top5_acc"], global_step)
            writer.add_scalar("eval/recall@10", recall["top10_acc"], global_step)
            writer.add_scalar("eval/ndcg@1", ndcg["ndcg@1"], global_step)
            writer.add_scalar("eval/ndcg@5", ndcg["ndcg@5"], global_step)
            writer.add_scalar("eval/ndcg@10", ndcg["ndcg@10"], global_step)
            model.train()

            if recall["top10_acc"] > best_recall:
                best_recall = recall["top10_acc"]
                save_model(model, tokenizer, optimizer)
                print(f"model saved for recal@10 = {best_recall}")
            
        writer.add_scalar("train/loss", loss.item(), global_step)

        # Save checkpoint
        # if global_step > 2000 and global_step % 500 == 0:
            # if loss.item() < best_loss:
            #     best_loss = loss.item()
            #     save_model(model, tokenizer, optimizer)




def main():
    
    writer = SummaryWriter(log_dir=f"{LOG_DIR}/{RUN_NAME}")

    model, tokenizer, old_vocab_size = load_model_tokenizer(run_test=True)
    
    dataset = SIDDataset()
    
    # Train
    train_sid_embeddings(model, dataset, tokenizer, old_vocab_size, writer)
    
    
if __name__ == "__main__":
    main()
