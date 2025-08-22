from dkmodel2vec.llm_loader import load_llm2vec_model
import torch

from dkmodel2vec.models import LlamaModelWrapper
from dkmodel2vec.distillation import distill_from_model_and_corpus
from pathlib import Path

if __name__ == "__main__": 
    model = load_llm2vec_model()
    wrapped_model = LlamaModelWrapper(model)


    # load dataset and filter out examples without both positive and negative samples
    # start experiment
#    mlflow.set_experiment(experiment_name)


 #   for idx, fold in enumerate(CV10) (stratify on negative samples): 
        # start run
        with mlflow.start_run(run_name = idx, nested = True):
            # Log parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("split", split)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("max_samples", max_samples)

#        ds = get_ds
#        get vocab
#        only keep top 200_000 tokens
#        quantize to Int8
#        distill model on train set

#        assess performance on test set
#        model performance: 
        # log run metrics
            # compare positive and negative samples
            # metrics: accuracy, confusion matrix
        

    m2v_model = distill_from_model_and_corpus(
        model=wrapped_model,
        tokenizer=model.tokenizer,
        pca_dims=256,
    ) 

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_name =  "StaticDK_baseline"
    m2v_model.save_pretrained(models_dir / model_name)

    