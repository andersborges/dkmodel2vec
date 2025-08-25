import os

from transformers import AutoModel, AutoTokenizer, AutoConfig
from llm2vec import LLM2Vec
import torch
from peft import PeftModel


def load_base_model(
    base_model_name_or_path: str | os.PathLike = "jealk/llm2vec-scandi-mntp-v2",
) -> AutoModel:
    # Loading base Llama model, along with custom code that enables bidirectional connections in decoder-only LLMs.
    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    return base_model


def merge_LoRA_weights_into_model(
    base_model: AutoModel, peft_model_name_or_path: str | os.PathLike
) -> AutoModel:
    """LoRA weights are merged into the base model."""

    model = PeftModel.from_pretrained(base_model, peft_model_name_or_path)
    model = model.merge_and_unload()  # This can take several minutes on cpu
    return model


def load_llm2vec_model(
    base_model_name_or_path: str | os.PathLike = "jealk/llm2vec-scandi-mntp-v2",
    supervised_model_name_or_path: str
    | os.PathLike
    | None = "jealk/TTC-L2V-supervised-2",
) -> LLM2Vec:
    """Wrapper function to load base model with LoRA weights from the same model path merged in and adding LoRA weights from other (supervised) model path."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    base_model = load_base_model(base_model_name_or_path=base_model_name_or_path)
    unsupervised_model = merge_LoRA_weights_into_model(
        base_model, base_model_name_or_path
    )
    # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
    model = PeftModel.from_pretrained(unsupervised_model, supervised_model_name_or_path)

    # Wrapper for encoding and pooling operations
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=8124)

    return l2v
