# test_llm2vec_import.py
from pathlib import Path
import pytest
from llm2vec import LLM2Vec
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from dkmodel2vec.llm_loader import load_base_model
from dkmodel2vec.models import LlamaModelWrapper
from dkmodel2vec.distillation import distill_from_model_and_corpus
from model2vec import StaticModel

import pytest
import numpy as np
from datasets import Dataset
from model2vec import StaticModel
from dkmodel2vec.evaluator import (
    evaluate_classification,
    evaluate_model,
)
from dkmodel2vec.models import LlamaModelWrapper
from dkmodel2vec.distillation import distill_from_model_and_corpus
from dkmodel2vec.constants import PREDICTION_COLUMN

def test_llm2vec_import():
    """Test that LLM2Vec class can be imported successfully."""
    try:
        from llm2vec import LLM2Vec

        assert LLM2Vec is not None
        assert callable(LLM2Vec)
    except ImportError as e:
        pytest.fail(f"Failed to import LLM2Vec: {e}")


def test_llm2vec_class_structure():
    """Test that LLM2Vec has expected attributes and methods."""
    from llm2vec import LLM2Vec

    assert isinstance(LLM2Vec, type)

    expected_methods = ["encode", "from_pretrained"]
    for method in expected_methods:
        assert hasattr(LLM2Vec, method), f"LLM2Vec should have {method} method"


def test_model2vec_version():
    """Test that we are using a version of model2vec which supports BPT tokenizers."""
    import model2vec

    assert int(model2vec.__version__.split(".")[1]) >= 6


@pytest.fixture(scope="session")
def tiny_llm2vec():
    """
    Load SmolLM2-135M once per test session.
    This avoids reloading the model for every test.
    """
    print("\n🔄 Loading SmolLM2-135M ...")

    model = LLM2Vec.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        device_map="cpu",
        torch_dtype=torch.float16,
        pooling_mode="mean",
        max_length=32,
    )

    return model


@pytest.fixture(scope="session")
def sample_peft_config():
    """Create dummy peft config to test that we can load a dummy model."""
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    return peft_config


@pytest.fixture(scope="session")
def sample_base_model_name():
    """Specify base model name for sample model."""
    return "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="session")
def sample_tokenizer(sample_base_model_name):
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(sample_base_model_name)

    # hot-fix as LLM2vec requires a tokenizer with padding token and padding_side
    if tokenizer.pad_token is None:
        # Instead of using eos_token (which might be an added token),
        # use a simple vocabulary token for padding
        vocab = tokenizer.get_vocab()
        for candidate in ["!", ".", ",", "?", ":", ";", "#", "$", "%"]:
            if candidate in vocab:
                tokenizer.pad_token = candidate
                break
        else:
            # Fallback to eos_token if no punctuation found
            tokenizer.pad_token = tokenizer.eos_token

    # hot-fix since pooling is only implemented for padding_side == "left"
    if tokenizer.padding_side is None or tokenizer.padding_side == "right":
        tokenizer.padding_side = "left"

    return tokenizer


@pytest.fixture(scope="session")
def tiny_fine_tuned_llm2vec_model(sample_peft_config, sample_tokenizer):
    """Test that the actual loading of a fine-tuned LLM2Vec model is succesful."""
    base_model = load_base_model(base_model_name_or_path="HuggingFaceTB/SmolLM2-135M")
    peft_model = PeftModel(model=base_model, peft_config=sample_peft_config)
    model = peft_model.merge_and_unload()  # This can take several minutes on cpu
    l2v = LLM2Vec(model=model, tokenizer=sample_tokenizer, pooling_mode="mean")

    return l2v


@pytest.fixture
def sample_texts():
    """Short test texts."""
    return ["Hi", "Hello", "Test"]


def test_tokenizer_import(sample_tokenizer):
    """Check the the type is right."""
    assert sample_tokenizer.pad_token is not None
    assert sample_tokenizer.padding_side is not None


def test_sample_peft_config(sample_peft_config):
    assert sample_peft_config is not None


# Tests using the cached model fixture
def test_llm2vec_model_loads(tiny_llm2vec):
    """Test that the model loads successfully (uses cached model)."""
    assert tiny_llm2vec is not None
    assert hasattr(tiny_llm2vec, "encode")


# Tests using the cached model fixture
def test_fine_tuned_llm2vec_model_loads(tiny_fine_tuned_llm2vec_model):
    """Test that the model loads successfully (uses cached model)."""
    assert tiny_fine_tuned_llm2vec_model is not None
    assert hasattr(tiny_fine_tuned_llm2vec_model, "encode")


def test_llm2vec_encode_texts(tiny_fine_tuned_llm2vec_model, sample_texts):
    """Test encoding one text (uses cached model)."""
    embeddings = tiny_fine_tuned_llm2vec_model.encode(sample_texts)

    assert embeddings is not None
    assert len(embeddings) == len(sample_texts)
    assert len(embeddings[0]) > 0


def test_llama_model_wrapper(tiny_fine_tuned_llm2vec_model):
    """Test that the LlamaModelWrapper properly handles token_type_ids."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model.model)
    print(f"Wrapped model type: {type(wrapped_model)}")
    print(f"Original model type: {type(tiny_fine_tuned_llm2vec_model.model)}")

    # Test the wrapper directly
    dummy_input = torch.tensor([[1, 2, 3]]).to(device)
    dummy_attention = torch.ones_like(dummy_input).to(device)
    dummy_token_types = torch.zeros_like(dummy_input).to(device)

    wrapped_model(
        input_ids=dummy_input,
        attention_mask=dummy_attention,
        token_type_ids=dummy_token_types,
    )

    # Test that .to() returns the wrapper
    wrapped_model_device = wrapped_model.to(device)
    assert type(wrapped_model_device) == type(wrapped_model), (
        "to() method should return wrapper"
    )

    # Test that device property works
    assert wrapped_model.device is not None


def test_model2vec_distillation(tiny_fine_tuned_llm2vec_model, sample_texts):
    """Test that we can actually do distillation with the basic LLM2Vec model."""
    from model2vec.distill import distill_from_model

    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)

    m2v_model = distill_from_model(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        pca_dims=256,
    )
    embeddings = m2v_model.encode(sample_texts)
    assert embeddings.shape

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_name = "test"
    m2v_model.save_pretrained(models_dir / model_name)


def test_adding_vocabulary(tiny_fine_tuned_llm2vec_model):
    from model2vec.distill import distill_from_model

    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)

    vocab = [
        "Helloworld",
        "theseunique",
        "tokensarenotlikely",
        "tobeinthestandardvocab",
    ]
    m2v_model = distill_from_model(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        vocabulary=vocab,
        pca_dims=256,
    )

    embeddings = m2v_model.encode(vocab)

    assert embeddings.shape[0] == len(vocab)
    for vocab_n in vocab:
        assert len(m2v_model.tokenize([vocab_n])) == 1


def test_custom_distillation(tiny_fine_tuned_llm2vec_model):
    from dkmodel2vec.distillation import distill_from_model_and_corpus

    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)
    texts = [
        "This Helloworld text is quite unique.",
        "This text contain the 'theseunique' token",
    ]
    vocab = [
        "Helloworld",
        "theseunique",
        "tokensarenotlikely",
        "tobeinthestandardvocab",
    ]
    m2v_model = distill_from_model_and_corpus(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        vocabulary=vocab,
        corpus=texts,
        pca_dims=256,
    )

    embeddings = m2v_model.encode(vocab)
    assert embeddings.shape[0]
    tokenizer = m2v_model.tokenizer
    assert len(tokenizer.encode("Helloworld").ids) == 1
    assert (
        tokenizer.encode("Helloworld").ids[0]
        in tokenizer.encode("This Helloworld text").ids
    )
    assert len(tokenizer.encode("tobeinthestandardvocab2").ids) > 1


def test_custom_distillation_with_instruction(tiny_fine_tuned_llm2vec_model):
    from dkmodel2vec.distillation import distill_from_model_and_corpus

    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)
    texts = [
        "This Helloworld text is quite unique.",
        "This text contain the 'theseunique' token",
    ]
    vocab = [
        "Helloworld",
        "theseunique",
        "tokensarenotlikely",
        "tobeinthestandardvocab",
    ]
    m2v_model = distill_from_model_and_corpus(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        vocabulary=vocab,
        instruction="PREPEND THIS TO MY TEXT:",
        corpus=texts,
        pca_dims=256,
    )

    embeddings = m2v_model.encode(vocab)
    assert embeddings.shape[0]
    tokenizer = m2v_model.tokenizer
    assert len(tokenizer.encode("Helloworld").ids) == 1
    assert (
        tokenizer.encode("Helloworld").ids[0]
        in tokenizer.encode("This Helloworld text").ids
    )
    assert len(tokenizer.encode("tobeinthestandardvocab2").ids) > 1


@pytest.fixture
def sample_dataset():
    """Short test examples"""
    from datasets import Dataset, DatasetDict

    querys = ["It's dangerous to go alone!", "I like green apples"]
    positives = [
        "Danger is imminent when solo. ",
        "My favourite fruit is sweet and round",
    ]
    negatives = ["I like cats", "Walking is slow"]
    languages = ["danish"] * len(querys)
    dataset = Dataset.from_dict(
        {
            "query": querys,
            "positive": positives,
            "negative": negatives,
            "language": languages,
        }
    )
    dataset = DatasetDict({"train": dataset})
    return dataset


def test_sample_dataset(sample_dataset):
    assert sample_dataset["train"].num_rows > 1


@pytest.fixture
def sample_model(fixture="session"):
    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    return model


def test_compute_distances(sample_model, sample_dataset):
    from dkmodel2vec.evaluator import compute_distances

    model = sample_model
    dataset = sample_dataset
    pos_dists, neg_dists = compute_distances(encoder=model, dataset=dataset["train"])
    # for the test cases, all the positive cases have shorter distances
    assert all(pos_dists < neg_dists)


def test_evaluate_classification():
    from dkmodel2vec.evaluator import evaluate_classification

    pos_dists = np.array([0, 1, 0.5])
    neg_dists = np.array([1, 1.1, 0.2])
    predictions = (pos_dists < neg_dists).astype(int)

    results = evaluate_classification(
        predictions=predictions, ground_truth=np.ones_like(predictions)
    )
    assert results["accuracy"] - 2 / 3.0 < 10 ** (-3)
    assert results["precision"] - 1 < 10 ** (-3)
    assert results["recall"] - 2 / 3.0 < 10 ** (-3)
    assert (results["predictions"] == [1, 1, 0]).all()
    assert (results["ground_truth"] == [1, 1, 1]).all()


def test_bm25(sample_tokenizer):
    from dkmodel2vec.evaluator import predict_bm25

    q = "CAT"
    pos = "CAT CAT CAT CAT "
    neg = "This is not it here either longer text"
    example = {"query": q, "positive": pos, "negative": neg}
    example_output = predict_bm25(example, sample_tokenizer)
    assert example_output["bm25_prediction"] == 1


def test_sentence_transformer_predict():
    from dkmodel2vec.data_loader import load_data
    from dkmodel2vec.evaluator import (
        predict_sentence_transformer,
        load_sentence_transformer,
    )

    model = load_sentence_transformer(model_name="minishlab/potion-base-8M")
    ds = load_data()
    ds = ds.filter(
        lambda example: True if example["has_positive_and_negative"] else False
    )
    batch_size = 1000
    ds = ds.map(
        lambda batch: predict_sentence_transformer(batch, model),
        batched=True,
        batch_size=batch_size,
    )


def test_create_vocabulary():
    from dkmodel2vec.vocab import create_vocabulary

    test_input = ["I like cats", "i LIKE likE CATS caTS CAts", "dogs"]
    vocab = create_vocabulary(test_input)
    assert vocab == ["cats", "like", "i", "dogs"]


# def test_token_ordering():
#     from model2vec.tokenizer.tokenizer import clean_and_create_vocabulary, replace_vocabulary
#     from dkmodel2vec.vocab import lower_case_tokenizer
#     from dkmodel2vec.distillation import estimate_token_frequencies, sort_tokens_by_frequency,calculate_weights
#     tokenizer = AutoTokenizer.from_pretrained( "jealk/llm2vec-scandi-mntp-v2")
#     tokenizer = lower_case_tokenizer(tokenizer)
#     vocab = ["unlikelytokenforsure1", "unlikelytokenforsure2", "unlikelytokenforsure3"]
#     corpus = ["unlikelytokenforsure1", "unlikelytokenforsure1 unlikelytokenforsure2 unlikelytokenforsure1 unlikelytokenforsure2 unlikelytokenforsure3"]
#     backend_tokenizer = tokenizer.backend_tokenizer
#     all_tokens, backend_tokenizer = clean_and_create_vocabulary(
#     tokenizer, vocab, token_remove_regex=None
#     )

#     backend_tokenizer_replaced_vocab = replace_vocabulary(
#         backend_tokenizer,
#         all_tokens, ",", "-"    )

#     token_counts = estimate_token_frequencies(
#             backend_tokenizer=backend_tokenizer_replaced_vocab,
#             corpus_texts=corpus,
#             batch_size=1000,
#         )


#     assert 1


def test_calculate_weights():
    from collections import Counter
    from dkmodel2vec.distillation import calculate_weights

    # Mock tokenizer with simple vocab
    class MockTokenizer:
        def get_vocab(self):
            return {"hello": 0, "world": 1, "rare": 2}

    tokenizer = MockTokenizer()

    token_counts = Counter({0: 10, 1: 5})  # token ids as keys

    weights = calculate_weights(tokenizer, token_counts)

    assert len(weights) == 3  # Should have weights for all 3 tokens
    assert all(w > 0 for w in weights.values())  # All weights should be positive
    assert weights[2] == weights[1]  # Unseen token gets same weight as least frequent
    assert weights[1] > weights[0]  # Less frequent tokens have higher weights


def test_weight_embeddings():
    from dkmodel2vec.distillation import weight_embeddings

    weights = {0: 1.0, 2: 0.1, 1: 0.01}
    embeddings = np.ones((len(weights), 256))
    weighted_embeddings = weight_embeddings(weights=weights, embeddings=embeddings)
    assert all(
        [
            (weighted_embeddings[key] - value).sum() < 0.000001
            for key, value in weights.items()
        ]
    )

    # test_debug_accuracy.py




@pytest.fixture
def debug_dataset():
    """Create a simple dataset where positive should clearly be closer than negative."""
    queries = ["I love cats", "Dogs are great pets", "Red apples are sweet"]
    positives = [
        "Cats are amazing animals",  # semantically similar to query
        "I have a wonderful dog",  # semantically similar to query
        "Sweet red fruits",  # semantically similar to query
    ]
    negatives = [
        "Mathematics is difficult",  # completely unrelated
        "Space exploration",  # completely unrelated
        "Computer programming",  # completely unrelated
    ]

    dataset = Dataset.from_dict(
        {
            "query": queries,
            "positive": positives,
            "negative": negatives,
            "has_positive_and_negative": [True] * len(queries),
        }
    )
    return dataset


def test_debug_accuracy_with_known_good_model(debug_dataset):
    """Test with a known working sentence transformer to verify our evaluation logic."""
    from sentence_transformers import SentenceTransformer

    # Use a small but working sentence transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    queries = debug_dataset["query"]
    positives = debug_dataset["positive"]
    negatives = debug_dataset["negative"]

    # Encode all texts
    query_embeds = model.encode(queries)
    pos_embeds = model.encode(positives)
    neg_embeds = model.encode(negatives)

    # Compute distances (same logic as your evaluator)
    pos_distances = np.linalg.norm(query_embeds - pos_embeds, axis=1)
    neg_distances = np.linalg.norm(query_embeds - neg_embeds, axis=1)

    print(f"Positive distances: {pos_distances}")
    print(f"Negative distances: {neg_distances}")
    print(f"Pos < Neg: {pos_distances < neg_distances}")

    # Make predictions
    predictions = (pos_distances < neg_distances).astype(int)

    # Evaluate
    results = evaluate_classification(predictions, np.ones_like(predictions))

    print(f"Accuracy with sentence transformer: {results['accuracy']}")
    print(f"Predictions: {predictions}")

    # This should have reasonable accuracy (> 0.5) with our clear examples
    assert results["accuracy"] > 0.5, (
        f"Expected accuracy > 0.5, got {results['accuracy']}"
    )


def trained_embeddings_do_not_explode(tiny_fine_tuned_llm2vec_model):
    """Test embeddings of words in the added vocabulary do not explode."""

    # Create the exact same setup as your main code
    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)

    # Simple test case
    texts = ["hello world", "goodbye world", "testing"]
    vocabulary = ["hello", "world", "goodbye", "testing"]

    # This mirrors your distill_from_model_and_corpus call
    m2v_model = distill_from_model_and_corpus(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        vocabulary=vocabulary,
        corpus=texts,
        pca_dims=256,
    )

    # Test the exact evaluation pattern
    dataset = Dataset.from_dict(
        {
            "query": ["hello", "apple"],
            "positive": ["hello world", "apple pie"],
            "negative": ["goodbye", "pear"],
            "has_positive_and_negative": [True, True],
        }
    )

    # Use your exact evaluation function
    result_dataset = evaluate_model(
        dataset=dataset,
        model=m2v_model,
        instruction_model=m2v_model,
        log_perf=False,  # Don't log to MLflow in test
    )

    predictions = result_dataset[PREDICTION_COLUMN]

    assert all(p == 1 for p in predictions)


def test_validation_of_parameters():
    from dkmodel2vec.distillation import _validate_parameters
    token_remove_pattern = r"\[unused\d+\]|\b\w*[A-Z]\w*\b"

    sif_coefficient, token_remove_regex = _validate_parameters(
        apply_zipf=False, sif_coefficient=1e-3, token_remove_pattern=token_remove_pattern
    )

    assert sif_coefficient is None
    assert token_remove_regex


def test_turn_tokens_into_llm2mvec_input(sample_tokenizer):
    from dkmodel2vec.vocab import turn_tokens_into_llm2mvec_input
    from model2vec.tokenizer.datamodels import Token

    unk_token = sample_tokenizer.special_tokens_map["unk_token"]

    tokens = [
        Token(form=".", normalized_form=".", is_subword=False, is_internal=True),
        Token(
            form="VERYUNLIKELYTOKEN",
            normalized_form="VERYUNLIKELYTOKEN",
            is_subword=False,
            is_internal=True,
        ),
        Token(
            form=unk_token,
            normalized_form=unk_token,
            is_internal=True,
            is_subword=False,
        ),
    ]
    ll2vec_input = turn_tokens_into_llm2mvec_input(
        tokens=tokens, tokenizer=sample_tokenizer, unk_token=unk_token
    )
    assert len(ll2vec_input) == 3


def test_turn_tokens_into_llm2mvec_input_with_instructions(sample_tokenizer):
    from dkmodel2vec.vocab import turn_tokens_into_llm2mvec_input
    from model2vec.tokenizer.datamodels import Token

    unk_token = sample_tokenizer.special_tokens_map["unk_token"]

    tokens = [
        Token(form=".", normalized_form=".", is_subword=False, is_internal=True),
        Token(
            form="VERYUNLIKELYTOKEN",
            normalized_form="VERYUNLIKELYTOKEN",
            is_subword=False,
            is_internal=True,
        ),
        Token(
            form=unk_token,
            normalized_form=unk_token,
            is_internal=True,
            is_subword=False,
        ),
    ]
    llm2vec_input = turn_tokens_into_llm2mvec_input(
        tokens=tokens,
        tokenizer=sample_tokenizer,
        unk_token=unk_token,
        instruction="Here is the prepend",
    )
    assert len(llm2vec_input) == 3
    assert all([len(inp) == 2 for inp in llm2vec_input])

def test_strip_upper_case():
    from dkmodel2vec.distillation import _validate_parameters
    from model2vec.tokenizer import clean_and_create_vocabulary

    token_remove_pattern = r"\[unused\d+\]"
    word_contains_upper_case_pattern = r"\b\w*[A-Z]\w*\b"
    token_remove_pattern = r"|".join([token_remove_pattern,word_contains_upper_case_pattern] )
    
    sif_coefficient, token_remove_regex = _validate_parameters(
        apply_zipf=False, sif_coefficient=1e-3, token_remove_pattern=token_remove_pattern
    )
    tokenizer = AutoTokenizer.from_pretrained("jealk/llm2vec-scandi-mntp-v2")
    vocabulary = ["kongen", "konger"]
    
    all_tokens, backend_tokenizer = clean_and_create_vocabulary(
        tokenizer, vocabulary, token_remove_regex=token_remove_regex
    )

    assert len([t for t in all_tokens if token_remove_regex.match(t.form) ]) == 0
    positives = ["kfaK", "KKK", "Mkjl", "ĠkfaK"] # should get matched
    negatives = ["Ġkfa", "kkk", "æ"] # should NOT get matched
    assert len([t for t in positives if token_remove_regex.match(t) ]) == len(positives) 
    assert len([t for t in negatives if token_remove_regex.match(t) ]) ==0

def test_strip_exotic():
    from dkmodel2vec.distillation import _validate_parameters
    from model2vec.tokenizer import clean_and_create_vocabulary

    token_remove_pattern = r"\[unused\d+\]"
    word_contains_upper_case_pattern = r"\b\w*[A-Z]\w*\b"
    # Match tokens with exotic chars, except 'Ġ' and '#' followed by only normal chars
    contains_exotic_token_pattern = r'^(?!Ġ[a-zA-ZæøåÆØÅ0-9.,\s]*$)(?!<\|end_of_text\|>$).*[^a-zA-ZæøåÆØÅ0-9.,\s]'
    token_remove_pattern = r"|".join([token_remove_pattern,word_contains_upper_case_pattern, contains_exotic_token_pattern] )

    sif_coefficient, token_remove_regex = _validate_parameters(
        apply_zipf=False, sif_coefficient=1e-3, token_remove_pattern=token_remove_pattern
    )
    tokenizer = AutoTokenizer.from_pretrained("jealk/llm2vec-scandi-mntp-v2")
    vocabulary = ["kongen", "konger"]
    
    all_tokens, backend_tokenizer = clean_and_create_vocabulary(
        tokenizer, vocabulary, token_remove_regex=token_remove_regex
    )
    matches = [t for t in all_tokens if token_remove_regex.match(t.form) ]
    assert len(matches) == 0
    positives = ["hello!",       # Contains ! (not normal) - MATCH
        "test@email",   # Contains @ (not normal) - MATCH  
        "café™",        # Contains ™ (not normal) - MATCH
        "Ġhello!", # contains ! - Match
        "Ġtest@email",  # contains @ - Match
        "#unlikelytoken___", # contains ___ - MATCH
        "#en" #contains # which is not an indicator of a continued word in BPE - So it should MATCH

    ]
    negatives = [
        "<|end_of_text|>", # special case
        "Ġhello", # starts with Ġ, rest are normal
        "hello",        # All normal - no match
        "hello world",  # Contains space (normal) - no match
        "hello123",     # All normal - no match
        "æøå",  # All normal - no match
        ]        

    assert len([t for t in positives if token_remove_regex.match(t) ]) == len(positives) 
    assert len([t for t in negatives if token_remove_regex.match(t) ]) ==0

def test_strip_uncommon():
    from dkmodel2vec.distillation import _validate_parameters
    from model2vec.tokenizer import clean_and_create_vocabulary

    token_remove_pattern = r"\[unused\d+\]"
    word_contains_upper_case_pattern = r"\b\w*[A-Z]\w*\b"
    contains_exotic_token_pattern = r'^(?!Ġ[a-zA-ZæøåÆØÅ0-9.,\s]*$)(?!<\|end_of_text\|>$).*[^a-zA-ZæøåÆØÅ0-9.,\s]'
    contains_uncommon_pattern = r'^\d{2,}$|^Ġ{2,}.*|^\.|^Ġ.*\d.*\d|(?=.*[a-zA-Z])(?=.*\d)'
    token_remove_pattern = r"|".join([token_remove_pattern, word_contains_upper_case_pattern, contains_exotic_token_pattern, contains_uncommon_pattern])

    sif_coefficient, token_remove_regex = _validate_parameters(
        apply_zipf=False, sif_coefficient=1e-3, token_remove_pattern=token_remove_pattern
    )
    tokenizer = AutoTokenizer.from_pretrained("jealk/llm2vec-scandi-mntp-v2")
    vocabulary = ["kongen", "konger"]
    
    all_tokens, backend_tokenizer = clean_and_create_vocabulary(
        tokenizer, vocabulary, token_remove_regex=token_remove_regex
    )
    
    matches = [t for t in all_tokens if token_remove_regex.match(t.form)]
    assert len(matches) == 0
    
    positives = [
        "11",           # Two digits - MATCH
        "090",          # Multiple digits - MATCH
        "4200",         # Multiple digits - MATCH
        "999999",       # Multiple digits - MATCH
        "ĠĠ4200",       # Starts with two Ġ - MATCH
        "ĠĠhello",      # Starts with two Ġ - MATCH
        "ĠĠĠtest",      # Starts with three Ġ - MATCH
        ".pdf",         # Starts with period - MATCH
        ".txt",         # Starts with period - MATCH
        ".config",      # Starts with period - MATCH
        "test123",      # Mixed letters and digits - MATCH
        "a11",          # Mixed letters and digits - MATCH
        "hello123world", # Mixed letters and digits - MATCH
        "Ġ4200",        # Starts with Ġ and contains 2+ digits - MATCH
        "Ġ1test2",      # Starts with Ġ and contains 2+ digits - MATCH
        "Ġ123",         # Starts with Ġ and contains 2+ digits - MATCH
    ]
    
    negatives = [
        "1",            # Single digit - no match
        "9",            # Single digit - no match
        "Ġ4",           # Starts with Ġ but only one digit - no match
        "Ġhello",       # Only one Ġ, no digits - no match
        "hello",        # Normal word - no match
        "pdf",          # Doesn't start with period - no match
        "test",         # Letters only - no match
    ]

    assert len([t for t in positives if token_remove_regex.match(t)]) == len(positives)
    assert len([t for t in negatives if token_remove_regex.match(t)]) == 0
    
def test_reduce_dimensions():
    from dkmodel2vec.distillation import reduce_dimensions
    from collections import Counter
    np.random.seed(79)
    embeddings = np.random.random((100, 10))
    counts = {0: 1, 3: 5, 5: 0 , 10: 10}
    token_counts = Counter(counts)
    pca_dims = 2
    reduced_embeds = reduce_dimensions(embeddings=embeddings, pca_dims=pca_dims, token_counts = token_counts)
    assert reduced_embeds.shape == (embeddings.shape[0], pca_dims) 


# def test_add_embeddings():
#     from dkmodel2vec.distillation import reduce_dimensions
#     from collections import Counter
#     np.random.seed(79)
#     embeddings = np.random.random((100, 10))
#     counts = {0: 1, 3: 5, 5: 0 , 10: 10}
#     token_counts = Counter(counts)
#     pca_dims = 2
#     reduced_embeds = reduce_dimensions(embeddings=embeddings, pca_dims=pca_dims, token_counts = token_counts)
#     assert reduced_embeds.shape == (embeddings.shape[0], pca_dims) 

def test_create_corpus():
    from dkmodel2vec.retrieval import create_corpus
    raw = {
        "idx": [0, 1,2], 
        "positive": ["hallo", None, "hej"], 
        "negative" : [None, "no", "nix"], 
        "split" : ["train", "val", "test"]}
    corpus = create_corpus(raw, columns = ["positive", "negative"], add_columns=["split"])
    long_form = {"query_idx" : [0,1, 2,2 ],
                  "document" : ["hallo", "no", "hej", "nix"], 
                  "column" : ["positive", "negative", "positive", "negative"], 
                  "split" : ["train", "val", "test", "test"]}
    assert corpus == long_form

def test_get_mapping_from_query_to_corpus():
    from dkmodel2vec.retrieval import get_mapping_from_query_to_corpus


def test_get_mapping_from_query_to_corpus():
    from dkmodel2vec.retrieval import get_mapping_from_query_to_corpus
    """Test mapping between query and corpus for positive examples."""   
    flat_corpus = Dataset.from_dict({
        'query_idx': [1, 2, 3],
        'document': ['doc1', 'doc2', 'doc3'],
        'column': ['positive', 'negative', 'positive']
    })
    result = get_mapping_from_query_to_corpus(flat_corpus)
    assert result == {1: 0, 3: 2}


def test_add_recall():
    from dkmodel2vec.retrieval import add_recall
    """Test recall calculation at different thresholds."""
    
    # Test: correct document at position 0 (found at all thresholds)
    example = {
        'corpus_idx': 42,
        'retrieved_document_idx': [42, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    }
    result = add_recall(example, thresholds=[5, 10])
    assert result['recall@5'] == 1
    assert result['recall@10'] == 1
    
    # Test: correct document at position 7 (only found at threshold 10+)
    example = {
        'corpus_idx': 99,
        'retrieved_document_idx': [1, 2, 3, 4, 5, 6, 7, 99, 9, 10]
    }
    result = add_recall(example, thresholds=[5, 10])
    assert result['recall@5'] == 0
    assert result['recall@10'] == 1
    
    # Test: correct document not in retrieved list (miss at all thresholds)
    example = {
        'corpus_idx': 999,
        'retrieved_document_idx': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    result = add_recall(example, thresholds=[5, 10])
    assert result['recall@5'] == 0
    assert result['recall@10'] == 0
    
    # Test: custom thresholds
    example = {
        'corpus_idx': 50,
        'retrieved_document_idx': [1, 10, 50]
    }
    result = add_recall(example, thresholds=[1, 2, 3])
    assert result['recall@1'] == 0
    assert result['recall@2'] == 0
    assert result['recall@3'] == 1

def test_text_length_filtering():
    from dkmodel2vec.config import DANISH_INSTRUCTION
    from dkmodel2vec.utils import check_fits_length, add_instruction_to_text
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jealk/llm2vec-scandi-mntp-v2")
    
    # Test data with short and long texts
    batch = {
        "document": ["Denne tekst er laaaaaaang..." * 1000, "kort", "kort uden instr"], 
        "column": ["query", "query", "positive"]
    }
    max_length = 800
    doc_max_length = 400
    
    # Test 1: Check length filtering
    length_check = check_fits_length(batch, tokenizer, max_length, doc_max_length)
    
    assert length_check['fits_length'][0] == False, "Long text should not fit"
    assert length_check['fits_length'][1] == True, "Short text should fit"
    assert length_check['fits_length'][2] == True, "Short text should fit"
    
    # Test 2: Filter to only texts that fit
    filtered_batch = {
        "document": [batch['document'][i] for i, fits in enumerate(length_check['fits_length']) if fits],
        "column": [batch['column'][i] for i, fits in enumerate(length_check['fits_length']) if fits]
    }
    
    assert len(filtered_batch['document']) == 2, "Should keep 2 short texts"
    
    # Test 3: Add instructions to filtered texts
    output = add_instruction_to_text(filtered_batch, DANISH_INSTRUCTION)
    
    assert len(output['processed_text']) == 2, "Should have 2 processed texts"
    assert DANISH_INSTRUCTION in output['processed_text'][0], "Query should have instruction"
    assert DANISH_INSTRUCTION not in output['processed_text'][1], "Positive should not have instruction"
    assert "!@#$%^&*()" in output['processed_text'][0], "Should have separator"
    assert "kort" in output['processed_text'][0], "Should contain original text"


def test_add_instruction_format():
    """Test that instruction formatting matches LLM2Vec exactly."""
    from dkmodel2vec.config import DANISH_INSTRUCTION
    from dkmodel2vec.utils import add_instruction_to_text
    
    batch = {
        "document": ["test text", "another text"],
        "column": ["query", "positive"]
    }
    
    output = add_instruction_to_text(batch, DANISH_INSTRUCTION)
    
    # Verify exact format: "instruction !@#$%^&*()text"
    assert output['processed_text'][0] == f"{DANISH_INSTRUCTION.strip()} !@#$%^&*()test text"
    assert output['processed_text'][1] == "!@#$%^&*()another text"