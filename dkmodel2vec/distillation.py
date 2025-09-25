from __future__ import annotations

import inspect
from collections import Counter
import logging
import os
import re
from typing import Optional, cast

from tqdm import tqdm
import numpy as np
from huggingface_hub import model_info
from tokenizers import Tokenizer
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from sklearn.decomposition import PCA

import torch
from dkmodel2vec.config import FALLBACK_UNK_TOKEN
from torch.nn.utils.rnn import pad_sequence
from dkmodel2vec.vocab import add_instruction_tokenizer
from dkmodel2vec.logging import log_memory_usage
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
from model2vec.distill.inference import _encode_mean_using_model
from dkmodel2vec.vocab import (
    turn_tokens_into_ids_with_instruction,
    turn_tokens_into_llm2mvec_input,
)
from model2vec.tokenizer.datamodels import Token

from llm2vec.llm2vec import LLM2Vec

_DEFAULT_BATCH_SIZE = 256


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
    logger.info("Distilling model...")
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
            Optional[str],
            [token for token in tokenizer.get_vocab() if token == FALLBACK_UNK_TOKEN][
                0
            ],
        )
        logger.info(
            "The unknown token is not set. Hardcoding it. This is a workaround to allow encoding of more texts without error."
        )
        unk_token_obj = Token(
            form="[UNK]", normalized_form="[UNK]", is_subword=False, is_internal=False
        )
        all_tokens = all_tokens + [unk_token_obj]
    if pad_token is None:
        if unk_token is not None:
            pad_token = unk_token
            logger.info(
                "The pad token is not set. Setting it to the unk token. This is a workaround for models that don't have a pad token."
            )
        else:
            pad_token = unk_token or all_tokens[0].form
            logger.info(
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
    #        sorted_all_tokens = sort_tokens_by_frequency(
    #            token_counts=token_counts, all_tokens=all_tokens
    #        )
    else:
        token_counts = Counter()
    #        sorted_all_tokens = all_tokens

    logger.info(f"Creating embeddings for {len(all_tokens)} tokens")

    # Convert tokens to IDs
    if instruction is not None:
        logger.info("""Adding instruction to tokens. 
              The output tokenizer is however not modified. 
              Remember you will need an encoder without instructions as well.""")

    token_ids = turn_tokens_into_ids_with_instruction(
        tokens=all_tokens,
        base_tokenizer=tokenizer,
        unk_token=unk_token,
        instruction=instruction,
    )

    # Create the embeddings
    log_memory_usage("Before embedding")
    embeddings = create_embeddings_memory_efficient(
        tokenized=token_ids,
        model=model,
        device=device,
        pad_token_id=tokenizer.get_vocab()[pad_token],
    )
    log_memory_usage("Before postprocessing")
    # Post process the embeddings by applying PCA but do NOT use Zipf weighting (set sif_coefficient to None)
    embeddings = post_process_embeddings(
        np.asarray(embeddings), pca_dims, sif_coefficient=None
    )
    log_memory_usage("Before weighing")

    weights = calculate_weights(
        backend_tokenizer_replaced_vocab, token_counts=token_counts
    )

    embeddings = weight_embeddings(weights=weights, embeddings=embeddings)

    log_memory_usage("Before quantization")

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
        "normalize": False,  # avoid normalization or our weighting will have been in vain
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
    log_memory_usage("Before Staticmodel")

    return StaticModel(
        vectors=embeddings,
        tokenizer=backend_tokenizer_replaced_vocab,
        config=config,
        base_model_name=model_name,
        language=language,
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
            encoded_batch = backend_tokenizer.encode_batch(batch_strings)

            for encoding in encoded_batch:
                token_counts.update(encoding.ids)

        except Exception as e:
            logger.info(f"Error processing batch at index {i}: {e}")
            for n, text in enumerate(batch_strings):
                try:
                    encoding = backend_tokenizer.encode(text)
                    token_counts.update(encoding.ids)
                except Exception as e_single:
                    logger.info(
                        f"Error processing index {i + n} with text: {text}: {e_single}"
                    )
                    continue

        if i % (batch_size * 10) == 0:
            logger.info(f"Processed {i} texts...")

    return token_counts


def sort_tokens_by_frequency(token_counts: Counter, all_tokens: list[Token]):
    """Tokenize corpus and sort tokens according to frequency. This is useful for weighing tokens."""
    seen_token_ids = [n[0] for n in token_counts.most_common()]
    seen_set = set(seen_token_ids)

    sorted_tokens = [all_tokens[token_id] for token_id in seen_token_ids] + [
        token for i, token in enumerate(all_tokens) if i not in seen_set
    ]
    return sorted_tokens


def normalize_embeddings(embeddings: np.array):
    """Apply l2 normalization to embeddings. Add an infinitesimal to avoid dividing by zero."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-12)

def reduce_dimensions(embeddings: np.array, pca_dims: int | None, token_counts: Counter)->np.array:
    """Reduce embeddings dimensions by keeping the principal pca_dims dimension.
    The principal components are found from the space spanned by the embeddings with finite token_counts 
    (embeddings vectors without a single count are ignored).  
    """
    if pca_dims is not None:
            used_tokens = [token_id for token_id, token_count in token_counts.most_common() if token_count>=1]
            used_subspace = embeddings.take(used_tokens, axis = 0)
            if len(used_subspace)==0:
                logger.warning(f"None of the tokens in the have a count higher than zero. This is likely a mistake or there is no corpus. Will fall back to using all dimensions. ")
                used_subspace = embeddings
                print( "fkjlaskdj")
            if pca_dims == "auto":
                pca_dims = used_subspace.shape[1]
            if pca_dims > used_subspace.shape[1]:
                logger.warning(
                    f"PCA dimension ({pca_dims}) is larger than the number of dimensions in the used subspace ({used_subspace.shape[1]}). "
                    "Applying PCA, but not reducing dimensionality. Is this is not desired, please set `pca_dims` to None. "
                    "Applying PCA will probably improve performance, so consider just leaving it."
                )
                pca_dims = embeddings.shape[1]
            if pca_dims >= embeddings.shape[0]:
                logger.warning(
                    f"PCA dimension ({pca_dims}) is larger than the number of tokens in the vocabulary ({embeddings.shape[0]}). Not applying PCA."
                )
            elif pca_dims <= embeddings.shape[1]:
                logger.info(f"Applying PCA with n_components {pca_dims}")

                orig_dims = embeddings.shape[1]

                p = PCA(n_components=pca_dims, svd_solver="full")
                p.fit(used_subspace)
                reduced_embeddings = p.transform(embeddings)

                if embeddings.shape[1] < orig_dims:
                    explained_variance_ratio = np.sum(p.explained_variance_ratio_)
                    explained_variance = np.sum(p.explained_variance_)
                    logger.info(f"Reduced dimensionality from {orig_dims} to {embeddings.shape[1]}.")
                    logger.info(f"Explained variance ratio: {explained_variance_ratio:.3f}.")
                    logger.info(f"Explained variance: {explained_variance:.3f}.")

    return reduced_embeddings


def weight_embeddings(weights: dict[int, float], embeddings: np.array):
    """Apply weighting in the order corresponding to the embeddings (sorted by token id)."""
    weights_sorted = [s[1] for s in sorted(weights.items(), key=lambda x: x[0])]
    weights_array = np.array(weights_sorted).reshape(len(weights), 1)

    return weights_array * embeddings


def calculate_weights(
    backend_tokenizer: PreTrainedTokenizerFast,
    token_counts: Counter,
    sif_coefficient: float = 1e-3,
) -> dict[int, float]:
    """Calculate the weight of each token from token frequency using SIF weighting.
    If there is no occurence of the token in token counts, then assume a maximum frequency corresponding 
    to the least frequent token and the weight of all tokens will be the same. """
    _, ids = zip(*sorted(backend_tokenizer.get_vocab().items(), key=lambda x: x[1]))
    total = token_counts.total()
    if total == 0:
        min_freq = 1
        total = 1
    else:
        min_freq = token_counts.most_common()[-1][1]
    weights = {
        id_n: sif_coefficient
        / (sif_coefficient + token_counts.get(id_n, min_freq) / total)
        for id_n in ids
    }
    return weights


def create_embeddings_memory_efficient(
    model: PreTrainedModel,
    tokenized: list[list[int]],
    device: str,
    pad_token_id: int,
) -> np.ndarray:
    """
    Create output embeddings for a bunch of tokens using a pretrained model.

    This version is more memory efficient than the version in model2vec
    because there are no duplicates generated. Instead the embeddings
    are allocated directly in the final sorted array in each batch.

    It does a forward pass for all tokens passed in `tokens`.

    :param model: The model to use.
        This should be a transformers model.
    :param tokenized: All tokenized tokens.
    :param device: The torch device to use.
    :param pad_token_id: The pad token id. Used to pad sequences.
    :return: The output embeddings.
    """
    model = model.to(device)

    num_sequences = len(tokenized)
    out_weights = None

    add_token_type_ids = "token_type_ids" in inspect.getfullargspec(model.forward).args

    # Sort by length for efficient batching
    lengths = np.asarray([len(sequence) for sequence in tokenized])
    sort_order = np.argsort(lengths)
    sorted_tokenized = [tokenized[i] for i in sort_order]

    pbar = tqdm(total=len(sorted_tokenized), desc="Encoding tokens", unit=" tokens")

    for batch_idx in range(0, len(sorted_tokenized), _DEFAULT_BATCH_SIZE):
        batch_end = min(batch_idx + _DEFAULT_BATCH_SIZE, len(sorted_tokenized))
        batch = [torch.Tensor(x).long() for x in sorted_tokenized[batch_idx:batch_end]]

        # Skip empty batches
        if not batch:
            continue

        encoded = {}
        encoded["input_ids"] = pad_sequence(
            batch, batch_first=True, padding_value=pad_token_id
        ).to(device)
        encoded["attention_mask"] = (encoded["input_ids"] != pad_token_id).to(device)

        if add_token_type_ids:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"]).to(
                device
            )

        # Add error handling for the model forward pass
        try:
            batch_embeddings = _encode_mean_using_model(model, encoded).cpu().numpy()
        except Exception as e:
            logging.info(f"Error in model forward pass for batch {batch_idx}: {e}")
            # Create dummy embeddings with NaNs for debugging
            if out_weights is not None:
                embedding_dim = out_weights.shape[1]
                batch_embeddings = np.full((len(batch), embedding_dim), np.nan)
            else:
                # If this is the first batch and it fails, we can't continue
                raise e

        if out_weights is None:
            embedding_dim = batch_embeddings.shape[1]
            out_weights = (
                np.full(  # Use np.full instead of np.empty to initialize with NaN
                    (num_sequences, embedding_dim), np.nan, dtype=batch_embeddings.dtype
                )
            )

        # Place each embedding directly in its final position
        for i, embedding in enumerate(batch_embeddings):
            # Map from sorted position back to original position
            sorted_idx = batch_idx + i

            # Add bounds checking
            if sorted_idx >= len(sort_order):
                logging.info(
                    f"Warning: sorted_idx {sorted_idx} exceeds sort_order length {len(sort_order)}"
                )
                continue

            original_idx = sort_order[sorted_idx]

            # Add bounds checking for original index
            if original_idx >= num_sequences:
                logging.info(
                    f"Warning: original_idx {original_idx} exceeds num_sequences {num_sequences}"
                )
                continue

            # Check if embedding contains NaN values
            if np.isnan(embedding).any():
                logging.info(
                    f"Warning: NaN values in embedding for original_idx {original_idx}, sorted_idx {sorted_idx}"
                )

            out_weights[original_idx] = embedding

        pbar.update(len(batch))

    pbar.close()

    # Check for any remaining NaN rows
    nan_rows = np.isnan(out_weights).all(axis=1)
    if nan_rows.any():
        nan_indices = np.where(nan_rows)[0]
        logger.info(
            f"Warning: {len(nan_indices)} rows still contain all NaN values: {nan_indices}"
        )

        # Debug: check if these correspond to specific input sequences
        for idx in nan_indices[:5]:  # Show first 5 for debugging
            logger.info(
                f"NaN row {idx}: original sequence length = {len(tokenized[idx])}"
            )

    out_weights = np.nan_to_num(out_weights)
    return out_weights


def distill_from_llm2vec_and_corpus(
    model: LLM2Vec,
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: list[str] | None = None,
    instruction: str | None = None,
    corpus: list[str] | None = None,
    device: str | None = None,
    pca_dims: PCADimType = 256,
    focus_pca: bool = False,
    sif_coefficient: float | None = None,
    prenormalize_embeddings: bool = False,
    prePCAnormalize_embeddings: bool = False,
    normalize_embeddings: bool = False,
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
    :param prePCAnormalize_embeddings: Normalize input to the PCA
    :param prenormalize_embeddings: Normalize input to weighting
    :param normalize_embeddings: Whether to normalize the aggregated embeddings at inference. 
    :param pca_dims: The number of components to use for PCA.
        If this is None, we don't apply PCA.
        If this is 'auto', we don't reduce dimensionality, but still apply PCA.
    :param focus_pca: 
        If this is True, the dimension reduction will only take embedding vectors into account that have a finite count. 
        If set to false, will use all embedding vectors, also those corresponding to tokens that are not even represented in the corpus.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
        If the pattern is so general that it removes all tokens, we throw an error. If the pattern can't be compiled into a valid regex, we also throw an error.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param use_subword: DEPRECATED: If this is not set to None, we show a warning. It doesn't do anything.
    :return: A StaticModel
    :raises: ValueError if the vocabulary is empty after preprocessing.

    """
    logger.info("Distilling model...")
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
            Optional[str],
            [token for token in tokenizer.get_vocab() if token == FALLBACK_UNK_TOKEN][
                0
            ],
        )
        logger.info(
            "The unknown token is not set. Hardcoding it. This is a workaround to allow encoding of more texts without error."
        )
        unk_token_obj = Token(
            form="[UNK]", normalized_form="[UNK]", is_subword=False, is_internal=False
        )
        all_tokens = all_tokens + [unk_token_obj]
    if pad_token is None:
        if unk_token is not None:
            pad_token = unk_token
            logger.info(
                "The pad token is not set. Setting it to the unk token. This is a workaround for models that don't have a pad token."
            )
        else:
            pad_token = unk_token or all_tokens[0].form
            logger.info(
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
    #        sorted_all_tokens = sort_tokens_by_frequency(
    #            token_counts=token_counts, all_tokens=all_tokens
    #        )
    else:
        token_counts = Counter()
    #        sorted_all_tokens = all_tokens

    logger.info(f"Creating embeddings for {len(all_tokens)} tokens")

    # Convert tokens to IDs
    if instruction is not None:
        logger.info("""Adding instruction to tokens. 
              The output tokenizer is however not modified. 
              Remember you will need an encoder without instructions as well.""")

    token_input = turn_tokens_into_llm2mvec_input(
        tokens=all_tokens,
        base_tokenizer=model.tokenizer,  # use the built-in tokenizer and not our lower-case version
        unk_token=unk_token,
        instruction=instruction,
    )

    # Create the embeddings
    log_memory_usage("Before embedding")

    embeddings = model.encode(token_input)
    # embeddings = create_embeddings_memory_efficient(
    #     tokenized=token_ids,
    #     model=model,
    #     device=device,
    #     pad_token_id=tokenizer.get_vocab()[pad_token],
    # )
    log_memory_usage("Before postprocessing")

    if prePCAnormalize_embeddings:
        embeddings = normalize_embeddings(embeddings)

    if focus_pca:
        embeddings = reduce_dimensions(np.asarray(embeddings), pca_dims=pca_dims, token_counts=token_counts)
    else:    
        # Manually set sif_coefficient to None here to move weigting to seperate function
        embeddings = post_process_embeddings(
            np.asarray(embeddings), pca_dims, sif_coefficient=None
        )
    log_memory_usage("Before weighing")

    weights = calculate_weights(
        backend_tokenizer_replaced_vocab,
        token_counts=token_counts,
        sif_coefficient=sif_coefficient,
    )
    if prenormalize_embeddings:
        embeddings = normalize_embeddings(embeddings)
    embeddings = weight_embeddings(weights=weights, embeddings=embeddings)

    log_memory_usage("Before quantization")

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
        "normalize": normalize_embeddings,
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
    log_memory_usage("Before Staticmodel")

    return StaticModel(
        vectors=embeddings,
        tokenizer=backend_tokenizer_replaced_vocab,
        config=config,
        normalize=normalize_embeddings,
        base_model_name=model_name,
        language=language,
    )
