import os 

from transformers import AutoModel, AutoTokenizer, AutoConfig
from llm2vec import LLM2Vec
import torch
from peft import PeftModel

def load_llm2vec_model(base_model_name_or_path: str | os.PathLike = "jealk/llm2vec-scandi-mntp-v2", supervised_model_name_or_path: str | os.PathLike = "jealk/TTC-L2V-supervised-2")->LLM2Vec:
    """Wrapper function to load model as described in model card. """
    # Loading base Llama model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path
    )
    config = AutoConfig.from_pretrained(
        base_model_name_or_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(
        model,
        base_model_name_or_path,
    )
    model = model.merge_and_unload()  # This can take several minutes on cpu

    # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
    model = PeftModel.from_pretrained(
        model, supervised_model_name_or_path
    )

    # Wrapper for encoding and pooling operations
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=8124)

    return l2v
