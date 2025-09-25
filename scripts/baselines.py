import argparse
import logging

from dkmodel2vec.evaluator import (
    evaluate_model,
    predict_bm25,
    evaluate_bm25,
    evaluate_sentence_transformer,
    evaluate_llm2vec_model
)
from dkmodel2vec.constants import (
    HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN,
    DATASET_NEGATIVE_COLUMN,
    DATASET_POSITIVE_COLUMN,
    DATASET_QUERY_COLUMN,
    PREDICTION_COLUMN,
    BM25_PREDICTION_COLUMN,
)
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow

from dkmodel2vec.llm_loader import load_llm2vec_model
from dkmodel2vec.data_loader import load_data
from dkmodel2vec.config import (
    VOCAB_SIZE,
    DANISH_INSTRUCTION,
    REFERENCE_MODEL2VEC,
    SIF_COEFFICIENT,
    N_SPLITS,
    RANDOM_STATE
)

from dkmodel2vec.vocab import create_vocabulary, lower_case_tokenizer
from dkmodel2vec.distillation import distill_from_llm2vec_and_corpus
from dkmodel2vec.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def evaluate_baselines(
    test_size: float = 0.1,
    random_state: int = RANDOM_STATE,
    mlflow_tracking_uri: str = "http://localhost:8000",
    mlflow_experiment: str = "llm2model2vec",
):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    # Load data and model
    logger.info("Loading data and model...")
    dsdk = load_data()
    model = load_llm2vec_model()
    tokenizer = lower_case_tokenizer(model.tokenizer)
    
    # Add BM25 predictions
    logger.info("Computing BM25 predictions...")
    dsdk = dsdk.map(lambda example: predict_bm25(example, tokenizer), num_proc=4)
    with mlflow.start_run(run_name="baselines"):
        # Split data
        logger.info(f"Splitting data with test_size={test_size}...")
        train_idx, test_idx = train_test_split(
            np.arange(dsdk.num_rows),
            test_size=test_size,
            stratify=dsdk["has_positive_and_negative"],
            random_state=random_state,
            shuffle=True
        )
        ds_train, ds_test = dsdk.select(train_idx), dsdk.select(test_idx)      
        # Prepare test set for evaluation
        ds_test_for_eval = ds_test.filter(
            lambda example: example[HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN]
        )

#        evaluate_llm2vec_model(dataset=ds_test_for_eval, model=model)

        evaluate_bm25(ds_test_for_eval)
        ds_test_for_eval = evaluate_sentence_transformer(
            dataset=ds_test_for_eval
        )
        ds_test_for_eval = evaluate_sentence_transformer(
            model_name=REFERENCE_MODEL2VEC,
            dataset=ds_test_for_eval
        )
        
        # # Evaluate ensemble
        # evaluate_ensemble_model(
        #     ds_test_for_eval,
        #     column_names=[
        #         PREDICTION_COLUMN,
        #         REFERENCE_MODEL2VEC,
        #         BM25_PREDICTION_COLUMN,
        #     ],
        # )

def main():
    parser = argparse.ArgumentParser(
    description="Train dk-llm2vec-model2vec with configurable parameters"
    )   
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:8000",
        help="MLflow tracking URI (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="llm2model2vec",
        help="MLflow experiment name (default: llm2model2vec)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed (default: 51)"
    )
    parser.add_argument(
    "--test-size",
    type=float,
    default=0.1,
    help="Test set size as fraction (default: 0.1)"
    )
    
    args = parser.parse_args()

    evaluate_baselines(
        random_state=args.random_state,
        test_size=args.test_size,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
)

if __name__ == "__main__":
    main()