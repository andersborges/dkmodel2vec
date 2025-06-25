from dkmodel2vec.llm_loader import load_llm2vec_model
import torch

from model2vec.distill import distill_from_model
from dkmodel2vec.models import LlamaModelWrapper
from pathlib import Path

if __name__ == "__main__": 
    model = load_llm2vec_model()
    wrapped_model = LlamaModelWrapper(model)
    m2v_model = distill_from_model(
        model=wrapped_model,
        tokenizer=model.tokenizer,
        pca_dims=256,
    )    

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_name =  "StaticDK_baseline"
    m2v_model.save_pretrained(models_dir / model_name)

    