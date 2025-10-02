from dkmodel2vec.config import VOCAB_SIZE
from tokenizers import Tokenizer
import regex
from collections import Counter
from tqdm import tqdm
import pandas as pd
from tokenizers import normalizers
import copy
from nltk.stem import SnowballStemmer 
from model2vec.tokenizer.tokenizer import find_eos_bos
from typing import cast

from tokenizers import Tokenizer

from transformers import PreTrainedTokenizerFast

from model2vec.tokenizer.datamodels import Token
from dkmodel2vec.constants import STEMMER_LANGUAGE
from dkmodel2vec.config import STEM
import logging

logger = logging.getLogger(__name__)


def tokenize_by_word(texts: list[str]) -> Counter:
    """Split texts by word. This is the vocabulary used later on."""
    tokenizer_regex = regex.compile(r"\w+")

    tokens = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        if text:
            tokens.extend(tokenizer_regex.findall(text.lower()))

    return tokens

def create_vocabulary(
    texts: list[str], 
    vocab_size: int = VOCAB_SIZE,
    stem: bool = STEM,
    stemmer_language: str = STEMMER_LANGUAGE
) -> list[str]:
    """
    Create the vocabulary that is used as input to distillation. 
    Vocabulary size is limited to vocab_size.
    
    Args:
        texts: List of text strings to build vocabulary from
        vocab_size: Maximum vocabulary size
        stem: Whether to apply stemming before counting
        stemmer_language: Language for stemmer (if stem=True)
    
    Returns:
        List of vocabulary tokens
    """
    logger.info("Creating vocabulary")
    words = tokenize_by_word(texts)
    
    if stem:
        stemmer = SnowballStemmer(stemmer_language)
        words = [stemmer.stem(word) for word in words]
    
    word_counts = Counter(words)
    vocabulary = [word for word, _ in word_counts.most_common(vocab_size)]
    if stem and len(vocabulary) < vocab_size:
        logger.info(f"Vocabulary reduced to {len(vocabulary)} tokens after stemming (requested: {vocab_size})")
    else:
        logger.info(f"Vocabulary created with {len(vocabulary)} tokens")

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


def add_instruction_tokenizer(tokenizer: Tokenizer, instruction: str) -> Tokenizer:
    """Add an instruction that gets prepended to any text before tokenizing.

    This approach preserves all existing normalizers and preprocessing.
    The instruction is added AFTER other normalizers (like lowercasing).
    Returns a copy of the tokenizer that can be serialized with standard tooling.
    """

    tokenizer_copy = copy.deepcopy(tokenizer)

    instruction_prepender = normalizers.Prepend(instruction + " ")

    if tokenizer_copy.backend_tokenizer.normalizer is not None:
        # Keep existing normalizers and add instruction prepending at the END
        tokenizer_copy.backend_tokenizer.normalizer = normalizers.Sequence(
            [tokenizer_copy.backend_tokenizer.normalizer, instruction_prepender]
        )
    else:
        # No existing normalizer, just add instruction prepending
        tokenizer_copy.backend_tokenizer.normalizer = instruction_prepender

    return tokenizer_copy


def turn_tokens_into_ids_with_instruction(
    tokens: list[Token],
    base_tokenizer: PreTrainedTokenizerFast,
    unk_token: str | None,
    instruction: str | None = None,
) -> list[list[int]]:
    """
    Convert a list of Token objects to their corresponding token ID sequences.

    :param tokens: List of Token objects to convert
    :param base_tokenizer: The base tokenizer to use for converting tokens to IDs
    :param unk_token: The string form of the unk token.
    :param instruction: Optional instruction to prepend to text before tokenization
    :return: List of token IDs corresponding to the input tokens
    """

    unk_id = (
        None if unk_token is None else base_tokenizer.convert_tokens_to_ids(unk_token)
    )
    prefix, suffix = find_eos_bos(base_tokenizer)

    # Create instruction tokenizer if instruction is provided
    if instruction is not None:
        encoding_tokenizer = add_instruction_tokenizer(base_tokenizer, instruction)
    else:
        encoding_tokenizer = base_tokenizer

    token_ids: list[list[int]] = []
    for token in tokens:
        if token.is_internal:
            # Careful. Any incorrect tokens will just get `[UNK]``, so this could go horribly wrong
            # Cast because return type is wrong.
            token_id: int = (
                cast(int, encoding_tokenizer.convert_tokens_to_ids(token.form)) or 0
            )
            # Explicitly check and warn if `unk_id` appears, but don't crash.
            if unk_id is not None and token_id == unk_id and token.form != unk_token:
                logger.warning(f"Token {token.form} was set to unk. This is wrong.")
            token_ids.append([*prefix, token_id, *suffix])
        else:
            token_ids.append(encoding_tokenizer.encode(token.form))

    return token_ids


def turn_tokens_into_llm2mvec_input(
    tokens: list[Token],
    base_tokenizer: PreTrainedTokenizerFast,
    unk_token: str | None,
    instruction: str | None = None,
) -> list[list[str]] | list[str]:
    """
    Convert a list of Token objects to their corresponding input for LLM2vec.

    :param tokens: List of Token objects to convert
    :param base_tokenizer: The base tokenizer to use for converting tokens to IDs
    :param unk_token: The string form of the unk token.
    :param instruction: Optional instruction to prepend to text before tokenization
    :return: List of token IDs corresponding to the input tokens
    """

    unk_id = (
        None if unk_token is None else base_tokenizer.convert_tokens_to_ids(unk_token)
    )

    token_sentences: list[list[str]] | list[str] = []
    for token in tokens:
        if token.is_internal:
            # Careful. Any incorrect tokens will just get `[UNK]``, so this could go horribly wrong
            token_id = base_tokenizer.convert_tokens_to_ids(token.form)
            # Explicitly check and warn if `unk_id` appears, but don't crash.
            if unk_id is not None and unk_id == token_id and token.form != unk_token:
                logger.warning(f"Token {token.form} was set to unk. This is wrong.")
        if instruction is None:
            token_sentences.append(token.form)
        else:
            token_sentences.append([instruction, token.form])

    return token_sentences
