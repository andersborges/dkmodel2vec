# test_llm2vec_import.py
from pathlib import Path
import pytest
from llm2vec import LLM2Vec
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel
import numpy as np 
from datasets import Dataset

from dkmodel2vec.llm_loader import load_base_model
from dkmodel2vec.models import LlamaModelWrapper
from dkmodel2vec.distillation import distill_from_model_and_corpus
from model2vec import StaticModel

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
    """Test that we are using a version of model2vec which supports BPT tokenizers. """
    import model2vec
    assert int(model2vec.__version__.split(".")[1])>=6

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
        for candidate in ['!', '.', ',', '?', ':', ';', '#', '$', '%']:
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
    """ Check the the type is right."""
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
    
    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model.model)
    print(f"Wrapped model type: {type(wrapped_model)}")
    print(f"Original model type: {type(tiny_fine_tuned_llm2vec_model.model)}")
    
    # Test the wrapper directly
    dummy_input = torch.tensor([[1, 2, 3]])
    dummy_attention = torch.ones_like(dummy_input)
    dummy_token_types = torch.zeros_like(dummy_input)
    
    result = wrapped_model(
        input_ids=dummy_input,
        attention_mask=dummy_attention,
        token_type_ids=dummy_token_types
    )
    
    # Test that .to() returns the wrapper
    wrapped_model_cpu = wrapped_model.to("cpu")
    assert type(wrapped_model_cpu) == type(wrapped_model), "to() method should return wrapper"
    
    # Test that device property works
    assert wrapped_model.device is not None


def test_model2vec_distillation(tiny_fine_tuned_llm2vec_model, sample_texts):
    """Test that we can actually do distillation with the basic LLM2Vec model.
    """
    from model2vec.distill import distill_from_model
    
    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)
    
    m2v_model = distill_from_model(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        pca_dims=256,
        device="cpu",
    )
    embeddings = m2v_model.encode(sample_texts)
    assert embeddings.shape

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_name =  "test"
    m2v_model.save_pretrained(models_dir / model_name)

def test_adding_vocabulary(tiny_fine_tuned_llm2vec_model): 
    from model2vec.distill import distill_from_model
    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)

    vocab = ["Helloworld", "theseunique" , "tokensarenotlikely", "tobeinthestandardvocab"]
    m2v_model = distill_from_model(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        vocabulary=vocab,
        pca_dims=256,
        device="cpu",
    )

    embeddings = m2v_model.encode(vocab)

    assert embeddings.shape[0] == len(vocab)
    for vocab_n in vocab:
        assert len(m2v_model.tokenize([vocab_n]) )== 1

def test_custom_distillation(tiny_fine_tuned_llm2vec_model): 
    from dkmodel2vec.distillation import distill_from_model_and_corpus
    wrapped_model = LlamaModelWrapper(tiny_fine_tuned_llm2vec_model)
    texts = ["This Helloworld text is quite unique.", "This text contain the 'theseunique' token"]
    vocab = ["Helloworld", "theseunique" , "tokensarenotlikely", "tobeinthestandardvocab"]
    m2v_model = distill_from_model_and_corpus(
        model=wrapped_model,
        tokenizer=tiny_fine_tuned_llm2vec_model.tokenizer,
        vocabulary=vocab,
        corpus = texts,
        pca_dims=256,
        device="cpu",
    )
    
    embeddings = m2v_model.encode(vocab)
    assert embeddings.shape[0]
    tokenizer = m2v_model.tokenizer
    assert len(tokenizer.encode("Helloworld").ids) == 1
    assert tokenizer.encode("Helloworld").ids[0] in tokenizer.encode("This Helloworld text").ids
    assert len(tokenizer.encode("tobeinthestandardvocab2").ids) > 1
    
@pytest.fixture
def sample_dataset():
    """Short test examples"""
    from datasets import Dataset, DatasetDict
    querys = ["It's dangerous to go alone!", "I like green apples"]
    positives = [  "Danger is imminent when solo. ","My favourite fruit is sweet and round"]
    negatives = ["I like cats",  "Walking is slow" ]
    languages = ["danish"] * len(querys)
    dataset = Dataset.from_dict({"query" :querys, "positive" : positives, "negative" : negatives, "language" : languages})
    dataset = DatasetDict({"train": dataset})
    return dataset

def test_sample_dataset(sample_dataset): 
    assert sample_dataset['train'].num_rows > 1 

@pytest.fixture
def sample_model(fixture = "session"):
    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    return model 


def test_compute_distances(sample_model, sample_dataset): 
    from dkmodel2vec.evaluator import compute_distances
    model = sample_model
    dataset = sample_dataset
    pos_dists, neg_dists = compute_distances(
            model, 
            dataset['train'])
    # for the test cases, all the positive cases have shorter distances
    assert all(pos_dists<neg_dists)

def test_evaluate_classification(): 
    from dkmodel2vec.evaluator import evaluate_classification
    pos_distances = np.array([0, 1, 0.5])
    neg_distances = np.array([1, 1.1, 0.2])
    
    results = evaluate_classification(pos_distances=pos_distances, neg_distances=neg_distances)
    assert results['accuracy'] - 2/3.0 < 10**(-3)
    assert results['precision'] - 1 < 10**(-3)
    assert results['recall'] - 2/3.0 < 10**(-3)
    assert (results['predictions'] == [1,1,0]).all()
    assert (results['ground_truth'] == [1,1,1]).all()

def test_bm25(sample_model):
    from dkmodel2vec.evaluator import predict_bm25
    q = "CAT"
    pos = "CAT CAT CAT CAT "
    neg = "This is not it here either longer text"
    example = {"query" : q, "positive" : pos, "negative" : neg}
    example_output = predict_bm25(example, sample_model.tokenizer)
    assert example_output['bm25_prediction'] == 1

def test_data_loader():
    from dkmodel2vec.data_loader import load_data
    ds = load_data()
    assert ds.num_rows >100
    assert True in ds['has_positive_and_negative'] and False in ds['has_positive_and_negative']

def test_create_vocabulary():
    from dkmodel2vec.vocab import create_vocabulary
    test_input = ["I like cats", "i LIKE likE CATS caTS CAts", "dogs"]
    vocab = create_vocabulary(test_input)
    assert vocab == ["cats", "like", "i", "dogs" ]