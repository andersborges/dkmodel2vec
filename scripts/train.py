from collections import Counter
from dkmodel2vec.llm_loader import load_llm2vec_model
import logging
import mlflow

from dkmodel2vec.data_loader import load_data
from dkmodel2vec.config import (
    VOCAB_SIZE,
    DANISH_INSTRUCTION,
    REFERENCE_MODEL2VEC,
    SIF_COEFFICIENT,
    N_SPLITS,
)
from dkmodel2vec.constants import (
    HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN,
    DATASET_NEGATIVE_COLUMN,
    DATASET_POSITIVE_COLUMN,
    DATASET_QUERY_COLUMN,
    PREDICTION_COLUMN,
    BM25_PREDICTION_COLUMN,
)
from dkmodel2vec.vocab import create_vocabulary, lower_case_tokenizer
from dkmodel2vec.models import LlamaModelWrapper
from dkmodel2vec.distillation import distill_from_llm2vec_and_corpus
from dkmodel2vec.logging import setup_logging
from dkmodel2vec.evaluator import (
    evaluate_model,
    predict_bm25,
    evaluate_bm25,
    evaluate_sentence_transformer,
    evaluate_ensemble_model,
)
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    model = load_llm2vec_model()
    #    wrapped_model = LlamaModelWrapper(model)
    tokenizer = lower_case_tokenizer(model.tokenizer)
    model.tokenizer = tokenizer
    mlflow.set_tracking_uri("http://localhost:8000")
    mlflow.set_experiment("llm2model2vec")
    dsdk = load_data()

    dsdk = dsdk.map(lambda example: predict_bm25(example, tokenizer), num_proc=4)

    skf = StratifiedKFold(n_splits=N_SPLITS)
    splits = skf.split(np.zeros(dsdk.num_rows), dsdk["has_positive_and_negative"])

    # Create a parent run for cross-validation
    with mlflow.start_run(run_name="cross_validation"):
        # Log common parameters at the parent level
        mlflow.log_param("VOCAB_SIZE", VOCAB_SIZE)
        mlflow.log_param("weight of unseen tokens", "min")
        mlflow.log_param("dataset size", dsdk.num_rows)
        mlflow.log_param("n_splits", 10)
        mlflow.log_param("sif_coefficient", SIF_COEFFICIENT)

        for fold_n, (train_idx, test_idx) in enumerate(splits):
            with mlflow.start_run(run_name=f"fold_{fold_n}", nested=True):
                mlflow.log_param("fold_number", fold_n)
                mlflow.log_param("train set size", len(train_idx))
                mlflow.log_param("test set size", len(test_idx))

                ds_train, ds_test = dsdk.select(train_idx), dsdk.select(test_idx)
                # Get texts from all examples in fold
                texts = (
                    ds_train[DATASET_QUERY_COLUMN]
                    + ds_train[DATASET_POSITIVE_COLUMN]
                    + ds_train[DATASET_NEGATIVE_COLUMN]
                )
                vocabulary = create_vocabulary(texts, vocab_size=VOCAB_SIZE)
                m2v_model = distill_from_llm2vec_and_corpus(
                    model=model,
                    tokenizer=tokenizer,
                    vocabulary=vocabulary,
                    corpus=texts,
                    pca_dims=256,
                    apply_zipf=True,
                    sif_coefficient=SIF_COEFFICIENT,
                )

                m2v_instruct_model = distill_from_llm2vec_and_corpus(
                    model=model,
                    tokenizer=tokenizer,
                    vocabulary=vocabulary,
                    instruction=DANISH_INSTRUCTION,
                    corpus=texts,
                    pca_dims=256,
                    apply_zipf=True,
                    sif_coefficient=SIF_COEFFICIENT,
                )
                ds_test_for_eval = ds_test.filter(
                    lambda example: True
                    if example[HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN]
                    else False
                )

                ds_test_for_eval = evaluate_model(
                    dataset=ds_test_for_eval,
                    model=m2v_model,
                    instruction_model=m2v_instruct_model,
                )
                evaluate_bm25(ds_test_for_eval)
                ds_test_for_eval = evaluate_sentence_transformer(
                    dataset=ds_test_for_eval
                )
                ds_test_for_eval = evaluate_sentence_transformer(
                    model_name=REFERENCE_MODEL2VEC, dataset=ds_test_for_eval
                )
                evaluate_ensemble_model(
                    ds_test_for_eval,
                    column_names=[
                        PREDICTION_COLUMN,
                        REFERENCE_MODEL2VEC,
                        BM25_PREDICTION_COLUMN,
                    ],
                )

    # Train final model on full dataset - separate run
    with mlflow.start_run(run_name="full_model"):
        mlflow.log_param("VOCAB_SIZE", VOCAB_SIZE)
        mlflow.log_param("weight of unseen tokens", "min")
        mlflow.log_param("dataset size", dsdk.num_rows)
        mlflow.log_param("training_type", "full_dataset")
        mlflow.log_param("sif_coefficient", SIF_COEFFICIENT)

        texts = (
            dsdk[DATASET_QUERY_COLUMN]
            + dsdk[DATASET_POSITIVE_COLUMN]
            + dsdk[DATASET_NEGATIVE_COLUMN]
        )
        vocabulary = create_vocabulary(texts, vocab_size=VOCAB_SIZE)
        m2v_model = distill_from_llm2vec_and_corpus(
            model=model,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            corpus=texts,
            pca_dims=256,
            apply_zipf=True,
            sif_coefficient=SIF_COEFFICIENT,
        )
        m2v_model = distill_from_llm2vec_and_corpus(
            model=model,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            instruction=DANISH_INSTRUCTION,
            corpus=texts,
            pca_dims=256,
            apply_zipf=True,
            sif_coefficient=SIF_COEFFICIENT,
        )
        #        evaluate_model(dataset=dsdk, model=m2v_model)
        #        evaluate_bm25(dsdk)
        #        evaluate_sentence_transformer(dsdk)
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)

        model_name = "dk-llm2vec-model2vec"
        m2v_model.save_pretrained(models_dir / model_name)

        # Log the model path
        mlflow.log_param("model_save_path", str(models_dir / model_name))
