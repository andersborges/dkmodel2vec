# test_llm2vec_import.py
import pytest
from llm2vec import LLM2Vec
import torch


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


# SESSION-SCOPED FIXTURE - loads once per test session
@pytest.fixture(scope="session")
def tiny_llm2vec():
    """
    Load SmolLM2-135M once per test session.
    This avoids reloading the model for every test.
    """
    print("\n🔄 Loading SmolLM2-135M (session fixture - loads once)...")

    model = LLM2Vec.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        device_map="cpu",
        torch_dtype=torch.float16,
        pooling_mode="mean",
        max_length=32,
        enable_bidirectional=True,  # Enable bidirectional as requested
    )

    print("✅ SmolLM2-135M loaded and cached for session (~135MB)")
    print("ℹ️  Subsequent tests will reuse this model instance")
    return model


@pytest.fixture
def sample_texts():
    """Short test texts."""
    return ["Hi", "Hello", "Test"]


# Tests using the cached model fixture
def test_llm2vec_model_loads(tiny_llm2vec):
    """Test that the model loads successfully (uses cached model)."""
    assert tiny_llm2vec is not None
    assert hasattr(tiny_llm2vec, "encode")


def test_llm2vec_encode_texts(tiny_llm2vec, sample_texts):
    """Test encoding one text (uses cached model)."""
    embeddings = tiny_llm2vec.encode(sample_texts)

    assert embeddings is not None
    assert len(embeddings) == len(sample_texts)
    assert len(embeddings[0]) > 0
