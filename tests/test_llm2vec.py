# test_llm2vec_import.py
import pytest
from llm2vec import LLM2Vec
import torch
from transformers import AutoTokenizer

from peft import LoraConfig, TaskType, PeftModel

from dkmodel2vec.llm_loader import load_base_model, add_device_property_if_missing


# Basic import tests (no fixtures needed)
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
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"

    return tokenizer


@pytest.fixture(scope="session")
def tiny_fine_tuned_llm2vec_model(sample_peft_config, sample_tokenizer):
    """Test that the actual loading of a fine-tuned LLM2Vec model is succesful."""
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # hot-fix as LLM2vec requires a tokenizer with padding token and padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"

    base_model = load_base_model(base_model_name_or_path="HuggingFaceTB/SmolLM2-135M")
    peft_model = PeftModel(model=base_model, peft_config=sample_peft_config)
    model = peft_model.merge_and_unload()  # This can take several minutes on cpu

    l2v = LLM2Vec(model=model, tokenizer=tokenizer, pooling_mode="mean")

    return l2v


@pytest.fixture
def sample_texts():
    """Short test texts."""
    return ["Hi", "Hello", "Test"]


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


def test_model2vec_distillation(
    tiny_fine_tuned_llm2vec_model, sample_tokenizer, sample_texts
):
    """Test that we can actually do distillation with the basic LLM2Vec model."""
    from model2vec.distill import distill_from_model

    model_with_forcefully_added_device = add_device_property_if_missing(
        tiny_fine_tuned_llm2vec_model
    )
    m2v_model = distill_from_model(
        model=model_with_forcefully_added_device,
        tokenizer=sample_tokenizer,
        pca_dims=256,
        device="cpu",
    )
    embeddings = m2v_model.encode(sample_texts)
    assert embeddings.shape


#   m2v_model.save_pretrained("m2v_model")
