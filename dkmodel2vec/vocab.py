from dkmodel2vec.config import VOCAB_SIZE
from tokenizers import Tokenizer
import regex
from collections import Counter
from tqdm import tqdm
import pandas as pd
from tokenizers import normalizers


def tokenize_by_word(texts: list[str]) -> Counter:
    """Split texts by word. This is the vocabulary used later on."""
    tokenizer_regex = regex.compile(r"\w+")

    tokens = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        if text:
            tokens.extend(tokenizer_regex.findall(text.lower()))

    return tokens


def create_vocabulary(texts: list[str], vocab_size=VOCAB_SIZE) -> list[str]:
    """Create the vocabulary that is used as input to distillation. Vocabulary size is limited to VOCAB_SIZE"""
    words = tokenize_by_word(texts)
    word_counts = Counter(words)
    vocabulary = [word for word, _ in word_counts.most_common(vocab_size)]
    return vocabulary


def lower_case_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """Modify the normalizer to lower-case any text before tokenizing. This reduces vocab size."""
    if tokenizer.backend_tokenizer.normalizer is not None:
        # Keep existing normalizers and add lowercase
        tokenizer.backend_tokenizer.normalizer = normalizers.Sequence(
            [tokenizer.backend_tokenizer.normalizer, normalizers.Lowercase()]
        )
    else:
        # No existing normalizer, just add lowercase
        tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()

    return tokenizer
