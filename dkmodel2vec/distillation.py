from __future__ import annotations

from collections import Counter
import logging
import os
import re
from typing import Optional, cast

import numpy as np
from huggingface_hub import model_info
from tokenizers import Tokenizer
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from model2vec.distill.inference import (
    PCADimType,
    create_embeddings,
    post_process_embeddings,
)
from model2vec.distill.utils import select_optimal_device
from model2vec.model import StaticModel
from model2vec.quantization import DType, quantize_embeddings
from model2vec.tokenizer import (
    clean_and_create_vocabulary,
    replace_vocabulary,
    turn_tokens_into_ids,
)
from dkmodel2vec.vocab import turn_tokens_into_ids_with_instruction
from model2vec.tokenizer.datamodels import Token


logger = logging.getLogger(__name__)


def distill_from_model_and_corpus(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: list[str] | None = None,
    instruction: str | None = None,
    corpus: list[str] | None = None,
    device: str | None = None,
    pca_dims: PCADimType = 256,
    sif_coefficient: float | None = None,
    apply_zipf=False,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    quantize_to: DType | str = DType.Float16,
    use_subword: bool | None = None,
) -> StaticModel:
    """
    Distill a staticmodel from a sentence transformer and a vocabulary.

    This function creates a set of embeddings from a sentence transformer. It does this by doing either
    a forward pass for all subword tokens in the tokenizer, or by doing a forward pass for all tokens in a passed vocabulary.

    If you pass through a vocabulary, we create a custom word tokenizer for that vocabulary and sort the combined internal and
    external tokens into a single token space sorted by frequency.

    If you don't pass a vocabulary, we use the model's tokenizer directly.

    :param model: The model to use.
    :param tokenizer: The tokenizer to use.
    :param vocabulary: The vocabulary to use. If this is None, we use the model's vocabulary.
    :param instruction: Instruction text if using embedding model that requires encoder. The intruction is prepended every token but the output tokenizer is not modified.
    :param corpus: The texts used to estimate frequency of tokens (both internal and external). If this is None, the ordering
    of the tokens will be internal tokens, then external tokens.
    :param device: The device to use.
    :param pca_dims: The number of components to use for PCA.
        If this is None, we don't apply PCA.
        If this is 'auto', we don't reduce dimensionality, but still apply PCA.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
        If the pattern is so general that it removes all tokens, we throw an error. If the pattern can't be compiled into a valid regex, we also throw an error.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param use_subword: DEPRECATED: If this is not set to None, we show a warning. It doesn't do anything.
    :return: A StaticModel
    :raises: ValueError if the vocabulary is empty after preprocessing.

    """
    if use_subword is not None:
        logger.warning(
            "The `use_subword` parameter is deprecated and will be removed in the next release. It doesn't do anything."
        )
    quantize_to = DType(quantize_to)
    backend_tokenizer = tokenizer.backend_tokenizer
    sif_coefficient, token_remove_regex = _validate_parameters(
        apply_zipf, sif_coefficient, token_remove_pattern
    )

    if vocabulary is None:
        vocabulary = []

    device = select_optimal_device(device)

    n_tokens_before = len(vocabulary)
    # Clean the vocabulary by removing duplicate tokens and tokens that are in the internal vocabulary.
    all_tokens, backend_tokenizer = clean_and_create_vocabulary(
        tokenizer, vocabulary, token_remove_regex=token_remove_regex
    )
    n_tokens_after = len([token for token in all_tokens if not token.is_internal])
    if n_tokens_before:
        logger.info(
            f"Adding {n_tokens_after} tokens to the vocabulary. Removed {n_tokens_before - n_tokens_after} tokens during preprocessing."
        )

    if not all_tokens:
        raise ValueError(
            "The vocabulary is empty after preprocessing. Please check your token_remove_pattern."
        )

    unk_token = cast(Optional[str], tokenizer.special_tokens_map.get("unk_token"))
    pad_token = cast(Optional[str], tokenizer.special_tokens_map.get("pad_token"))

    # Weird if to satsify mypy

    if unk_token is None:
        unk_token = cast(
            Optional[str], [token for token in tokenizer.get_vocab() if token == ","][0]
        )
        print(
            "The unknown token is not set. Hardcoding it. This is a workaround to allow encoding of more texts without error."
        )
        unk_token_obj = Token(
            form="[UNK]", normalized_form="[UNK]", is_subword=False, is_internal=False
        )
        all_tokens = all_tokens + [unk_token_obj]
    if pad_token is None:
        if unk_token is not None:
            pad_token = unk_token
            print(
                "The pad token is not set. Setting it to the unk token. This is a workaround for models that don't have a pad token."
            )
        else:
            pad_token = unk_token or all_tokens[0].form
            print(
                "The pad token is not set. Setting it to the first token in the vocabulary. This is a workaround for models that don't have a pad token."
            )

    # Replace the vocabulary in the tokenizer with the new vocabulary to eventually hold the static embeddings
    backend_tokenizer_replaced_vocab = replace_vocabulary(
        backend_tokenizer,
        all_tokens,
        unk_token=unk_token,
        pad_token=pad_token,
    )

    if corpus is not None:
        token_counts = estimate_token_frequencies(
            backend_tokenizer=backend_tokenizer_replaced_vocab,
            corpus_texts=corpus,
            batch_size=1000,
        )
        sorted_all_tokens = sort_tokens_by_frequency(
            token_counts=token_counts, all_tokens=all_tokens
        )
    else:
        sorted_all_tokens = all_tokens

    logger.info(f"Creating embeddings for {len(all_tokens)} tokens")

    # Convert tokens to IDs
    if instruction is not None:
        print("""Adding instruction to tokens. 
              The output tokenizer is however not modified. 
              Remember you will need an encoder without instructions as well.""")

    token_ids = turn_tokens_into_ids_with_instruction(
        tokens=sorted_all_tokens,
        base_tokenizer=tokenizer,
        unk_token=unk_token,
        instruction=instruction,
    )

    # Create the embeddings
    embeddings = create_embeddings(
        tokenized=token_ids,
        model=model,
        device=device,
        pad_token_id=tokenizer.get_vocab()[pad_token],
    )

    # Post process the embeddings by applying PCA but do NOT use Zipf weighting (set sif_coefficient to None)
    embeddings = post_process_embeddings(
        np.asarray(embeddings), pca_dims, sif_coefficient=None
    )

    # Weigh each embedding with 1/freq, where freq is the count of each particular token
    embeddings = weigh_by_freq(token_counts=token_counts, embeddings=embeddings)

    # Quantize the embeddings.
    embeddings = quantize_embeddings(embeddings, quantize_to)

    model_name = getattr(model, "name_or_path", "")

    config = {
        "model_type": "model2vec",
        "architectures": ["StaticModel"],
        "tokenizer_name": model_name,
        "apply_pca": pca_dims,
        "apply_zipf": False,  # hard coded
        "sif_coefficient": sif_coefficient,
        "hidden_dim": embeddings.shape[1],
        "seq_length": 1000000,  # Set this to a high value since we don't have a sequence length limit.
        "normalize": True,
    }

    if os.path.exists(model_name):
        # Using a local model. Get the model name from the path.
        model_name = os.path.basename(model_name)
        language = None
    else:
        # Get the language from the model card.
        try:
            info = model_info(model_name)
            language = (
                info.cardData.get("language", None)
                if info.cardData is not None
                else None
            )
        except Exception as e:
            # NOTE: bare except because there's many reasons this can fail.
            logger.warning(
                f"Couldn't get the model info from the Hugging Face Hub: {e}. Setting language to None."
            )
            language = None

    return StaticModel(
        vectors=embeddings,
        tokenizer=backend_tokenizer_replaced_vocab,
        config=config,
        base_model_name=model_name,
        language=language,
        normalize=True,
    )


def _validate_parameters(
    apply_zipf: bool | None,
    sif_coefficient: float | None,
    token_remove_pattern: str | None,
) -> tuple[float | None, re.Pattern | None]:
    """
    Validate the parameters passed to the distillation function.

    :param apply_zipf: DEPRECATED: This parameter used to control whether Zipf is applied.
        Zipf weighting is now controlled by the sif_coefficient parameter. If this is set to None, no weighting is applied.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value >= 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
    :return: The SIF coefficient to use.
    :raises: ValueError if the regex can't be compiled.

    """
    if apply_zipf is not None:
        logger.warning(
            "The `apply_zipf` parameter is deprecated and will be removed in the next release. "
            "Zipf weighting is applied based on the sif_coefficient parameter. If this is set to None, "
            "no weighting is applied."
        )
        if apply_zipf and sif_coefficient is None:
            logger.warning(
                "You set apply_zipf to True, but sif_coefficient is None. Setting sif_coefficient to 1e-4."
            )
            sif_coefficient = 1e-4
        elif not apply_zipf:
            logger.warning(
                "Because you set apply_zipf to False, we ignore the sif_coefficient parameter."
            )
            sif_coefficient = None

    if sif_coefficient is not None:
        if not 0 < sif_coefficient < 1.0:
            raise ValueError("SIF coefficient must be a value > 0 and < 1.0.")

    token_remove_regex: re.Pattern | None = None
    if token_remove_pattern is not None:
        try:
            token_remove_regex = re.compile(token_remove_pattern)
        except re.error as e:
            raise ValueError(f"Couldn't compile the regex pattern: {e}")

    return sif_coefficient, token_remove_regex


def estimate_token_frequencies(
    backend_tokenizer: PreTrainedTokenizerFast,
    corpus_texts: list[str],
    batch_size=1000,
    sample_size=None,
):
    """Estimate token frequencies using backend tokenizer"""

    token_counts = Counter()

    if sample_size:
        import random

        corpus_texts = random.sample(list(corpus_texts), sample_size)

    # Process in batches
    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i : i + batch_size]

        # Ensure all items are strings and filter out empty/None
        batch_strings = []
        for text in batch:
            if text is not None and isinstance(text, str) and text.strip():
                batch_strings.append(text.strip())

        if not batch_strings:  # Skip empty batches
            continue

        try:
            # Backend tokenizer encode_batch method
            encoded_batch = backend_tokenizer.encode_batch(batch_strings)

            # Count all token IDs in batch
            for encoding in encoded_batch:
                token_counts.update(encoding.ids)

        except Exception as e:
            print(f"Error processing batch at index {i}: {e}")
            # Fall back to individual processing for this batch
            for n, text in enumerate(batch_strings):
                try:
                    encoding = backend_tokenizer.encode(text)
                    token_counts.update(encoding.ids)
                except Exception as e_single:
                    print(
                        f"Error processing index {i + n} with text: {text}: {e_single}"
                    )
                    continue

        if i % (batch_size * 10) == 0:
            print(f"Processed {i} texts...")

    return token_counts


def sort_tokens_by_frequency(token_counts: Counter, all_tokens: list[Token]):
    """Tokenize corpus and sort tokens according to frequency. This is useful for weighing tokens."""
    seen_token_ids = [n[0] for n in token_counts.most_common()]
    seen_set = set(seen_token_ids)

    sorted_tokens = [all_tokens[token_id] for token_id in seen_token_ids] + [
        token for i, token in enumerate(all_tokens) if i not in seen_set
    ]
    return sorted_tokens


def weigh_by_freq(embeddings: np.array, token_counts: Counter):
    """Assuming embeddings ordered by token_count. Scale each embedding by frequency of token (word)."""
    total = token_counts.total()
    weights_of_seen_tokens = np.asarray(
        [total / count_n for _, count_n in token_counts.most_common()]
    )
    norm_weights = weights_of_seen_tokens / np.abs(np.max(weights_of_seen_tokens))
    weights = np.ones(embeddings.shape[0], dtype=float)
    weights[: norm_weights.shape[0]] = norm_weights
    embeddings *= weights[:, None]
    return embeddings
