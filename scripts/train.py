from collections import Counter
from dkmodel2vec.llm_loader import load_llm2vec_model

import mlflow

from dkmodel2vec.data_loader import load_data
from dkmodel2vec.config import VOCAB_SIZE, DANISH_INSTRUCTION
from dkmodel2vec.vocab import create_vocabulary, lower_case_tokenizer
from dkmodel2vec.models import LlamaModelWrapper
from dkmodel2vec.distillation import distill_from_model_and_corpus
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


if __name__ == "__main__":
    model = load_llm2vec_model()
    wrapped_model = LlamaModelWrapper(model)
    tokenizer = lower_case_tokenizer(model.tokenizer)
    wrapped_model.tokenizer = tokenizer
    # start experiment
    mlflow.set_tracking_uri("http://localhost:8000")
    mlflow.set_experiment("llm2model2vec")

    dsdk = load_data()

    dsdk = dsdk.map(lambda example: predict_bm25(example, tokenizer), num_proc=4)

    skf = StratifiedKFold(n_splits=10)
    splits = skf.split(np.zeros(dsdk.num_rows), dsdk["has_positive_and_negative"])

    for fold_n, (train_idx, test_idx) in enumerate(splits):
        with mlflow.start_run(run_name=f"fold_{fold_n}", nested=True):
            ds_train, ds_test = dsdk.select(train_idx), dsdk.select(test_idx)
            # Get texts from all examples in fold
            ds_train = ds_train.select(range(5_000))
            texts = ds_train["query"] + ds_train["positive"] + ds_train["negative"]
            vocabulary = create_vocabulary(texts, vocab_size=VOCAB_SIZE)

            m2v_model = distill_from_model_and_corpus(
                model=wrapped_model,
                tokenizer=tokenizer,
                vocabulary=vocabulary,
                corpus=texts,
                pca_dims=256,
                quantize_to="int8",
            )

            m2v_instruct_model = distill_from_model_and_corpus(
                model=wrapped_model,
                tokenizer=tokenizer,
                vocabulary=vocabulary,
                instruction=DANISH_INSTRUCTION,
                corpus=texts,
                pca_dims=256,
                quantize_to="int8",
            )
            ds_test_for_eval = ds_test.filter(
                lambda example: True if example["has_positive_and_negative"] else False
            )

            ds_test_for_eval = evaluate_model(
                dataset=ds_test_for_eval,
                model=m2v_model,
                instruction_model=m2v_instruct_model,
            )
            evaluate_bm25(ds_test_for_eval)
            ds_test_for_eval = evaluate_sentence_transformer(ds_test_for_eval)
            evaluate_ensemble_model(ds_test_for_eval)

    # train final model on full dataset
    with mlflow.start_run(run_name="full_model"):
        texts = dsdk["query"] + dsdk["positive"] + dsdk["negative"]
        vocabulary = create_vocabulary(texts, vocab_size=VOCAB_SIZE)
        m2v_model = distill_from_model_and_corpus(
            model=wrapped_model,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            corpus=texts,
            pca_dims=256,
            quantize_to="int8",
        )
        m2v_model = distill_from_model_and_corpus(
            model=wrapped_model,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            instruction=DANISH_INSTRUCTION,
            corpus=texts,
            pca_dims=256,
            quantize_to="int8",
        )
        #        evaluate_model(dataset=dsdk, model=m2v_model)
        #        evaluate_bm25(dsdk)
        #        evaluate_sentence_transformer(dsdk)
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)

        model_name = "dkllm2vec2model2vec"
        m2v_model.save_pretrained(models_dir / model_name)
