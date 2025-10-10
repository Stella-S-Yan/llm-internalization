from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    def __init__(self):
        self.pad_token = "PAD"
        self.unk_token = "UNK"
        self.special_tokens = [self.pad_token, self.unk_token]
        self.token2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

    @abstractmethod
    def tokenize(self, text):
        """Convert raw text into list of tokens"""
        pass

    @property
    def vocab_size(self):
        return len(self.token2id)

    @abstractmethod
    def build_vocab(self, texts):
        """Create vocabulary from dataset"""
        pass
    
    def pad_sequence(self, ids: List[int], max_length: int):
        pad_id = self.token2id[self.pad_token]
        padded = ids[:max_length] + [pad_id] * (max_length - len(ids))
        return padded[:max_length]
    
    def pad_batch(self, batch_ids: List[List[int]], max_length: int, dynamic_padding: bool=False):
        """
        Right-pad a batch of sequences (list of list of token ids).
        Pads each sequence to the maximum length in the batch or self.max_len.
        Dynamic padding is not suitable for JAX, as it change batch data shape all the time
        """
        if dynamic_padding:
            max_length = min(max(len(ids) for ids in batch_ids), max_length)

        return [self.pad_sequence(ids, max_length) for ids in batch_ids]

    
    def decode(self, ids):
        return [self.id2token.get(idx, self.unk_token) for idx in ids]


class SimpleWhitespaceTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.bos_token = "BOS"
        self.eos_token = "EOS"
        self.special_tokens.extend([self.bos_token, self.eos_token])
        self.token2id[self.bos_token] = len(self.token2id)
        self.token2id[self.eos_token] = len(self.token2id)
        self.id2token = {idx: token for token, idx in self.token2id.items()}

    def tokenize(self, text: str):
        """Tokenizes text by whitespace."""
        return text.strip().split()

    def build_vocab(self, texts):
        """Builds vocabulary by processing all texts."""
        idx = len(self.token2id)
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token2id:
                    self.token2id[token] = idx
                    self.id2token[idx] = token
                    idx += 1

    
    def encode(self, text, append_eos: bool=False):
        """Encodes text to token ids with BOS and EOS tokens."""
        tokens = self.tokenize(text)
        res = []
        if append_eos:
            res = [self.token2id.get(token, self.token2id[self.unk_token]) for token in tokens] + [self.token2id[self.eos_token]]
        else:
            res = [self.token2id.get(token, self.token2id[self.unk_token]) for token in tokens]
        return res        

    def encode_batch(self, text: List[str], append_eos: bool=False) -> List[List[int]]:
        return [self.encode(text, append_eos) for text in text]

    
    def decode(self, ids: List[int]) -> str:
        return ' '.join([self.id2token.get(idx, self.unk_token) for idx in ids])
    
    
    def decode_batch(self, ids: List[List[int]]) -> List[str]:
        """Decodes token ids back to text, keeping BOS and EOS.
        Supports both a list of ids and a list of list of ids.
        """
        return [self.decode(sublist) for sublist in ids]


    def encode_and_pad(self, text: List[str], max_length: int, append_eos: bool=False):
        encoded = self.encode_batch(text, append_eos)
        padded = self.pad_batch(encoded, max_length)
        return padded
    
    @property
    def eos_token_id(self):
        return self.token2id.get(self.eos_token)
    
    @property
    def bos_token_id(self):
        return self.token2id.get(self.bos_token)
    
    @property
    def pad_token_id(self):
        return self.token2id.get(self.pad_token)
