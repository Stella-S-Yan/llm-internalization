import config
from utils import tokenizers_rec
from utils import bagz_utils


def build_tokenizer():
  """Builds, saves, and returns a tokenizer for recommendation system tokens.

  This function creates a vocabulary consisting of user IDs and prefixed item
  tokens (A, B, C, D). It then builds a `SimpleWhitespaceTokenizer` with this
  vocabulary and saves the tokenizer object and its vocabulary to files
  specified in `config.TOKENIZER` and `config.TOKENIZER_TXT`, respectively.

  Returns:
    A `tokenizers_rec.SimpleWhitespaceTokenizer` instance.
  """
  user_ids = [f"user_{i}" for i in range(2000)]
  prefix_tokens = [f"{prefix}{i}" for prefix in "ABCD" for i in range(256)]
  vocab = prefix_tokens + user_ids

  tokenizer = tokenizers_rec.SimpleWhitespaceTokenizer()
  tokenizer.build_vocab(vocab)
  bagz_utils.save_object(tokenizer, config.TOKENIZER)

  return tokenizer


def load_tokenizer():
  tokenizer = bagz_utils.read_object(config.TOKENIZER)
  return tokenizer
