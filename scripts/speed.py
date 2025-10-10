import logging
from time import time

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from bm25s.tokenization import Tokenizer

import numpy as np
import mlflow
from dkmodel2vec.config import RANDOM_STATE
from dkmodel2vec.retrieval import add_embeddings_wrapped
from dkmodel2vec.data_loader import load_data
from dkmodel2vec.retrieval import create_fresh_corpus_and_queries
from dkmodel2vec.config import RANDOM_STATE, BEST_SENTENCE_TRANSFORMER
from dkmodel2vec.logging import setup_logging
from model2vec.model import StaticModel

setup_logging()
logger = logging.getLogger(__name__)

MODEL2VEC_PATH = "scripts/models/dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings"
CHEAP_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
batch_size_small = 1000
batch_size_large = 10000


def measure_sentence_transformer_speed(contiguous_corpus: Dataset, device: str):
    with mlflow.start_run(run_name="encoding"):
        for model_path in [ CHEAP_MODEL_PATH]:#BEST_SENTENCE_TRANSFORMER,
            with mlflow.start_run(run_name=model_path.split()[-1], nested=True):
                model = SentenceTransformer(model_path)
                for max_rows in [10_000]:
                    with mlflow.start_run(run_name=f"max_rows_{max_rows}", nested=True):
                        corpus_n = contiguous_corpus.select(range(max_rows))
                        mlflow.log_param("num_rows", corpus_n.num_rows)
                        mlflow.log_param("device", device)
                        mlflow.log_param("model", model_path)
                        t0 = time()
                        corpus_with_embeddings = add_embeddings_wrapped(
                            ds=corpus_n,
                            model=model,
                            in_column="document",
                            out_column="embeddings",
                            batch_size=batch_size_small,
                            device=device,
                        )
                        encoding_time = time() - t0
                        mlflow.log_metric("encoding_seconds", encoding_time)
                        mlflow.log_metric(
                            "embeddings_per_sec", corpus_n.num_rows / encoding_time
                        )
                        output_dim = len(corpus_with_embeddings["embeddings"][0])
                        mlflow.log_param("output_dim", output_dim)
    return


def measure_model2vec_speed(contiguous_corpus: Dataset, device:str, model_path:str):
    """Assess the speed with a larger batch size because it is sooo fast."""
    with mlflow.start_run(run_name=model_path.split()[-1], nested=True):
        model = StaticModel.from_pretrained(model_path)
        for max_rows in [1_000, 10_000, 50_000]:
            with mlflow.start_run(run_name=f"max_rows_{max_rows}", nested=True):
                corpus_n = contiguous_corpus.select(range(max_rows))
                mlflow.log_param("num_rows", corpus_n.num_rows)
                mlflow.log_param("device", device)
                mlflow.log_param("model", model_path)
                mlflow.log_param("batch_size", batch_size_large)
                t0 = time()
                corpus_with_embeddings = add_embeddings_wrapped(
                    ds=corpus_n,
                    model=model,
                    in_column="document",
                    out_column="embeddings",
                    batch_size=batch_size_large,
                    device=device,
                )
                encoding_time = time() - t0
                mlflow.log_metric("encoding_seconds", encoding_time)
                mlflow.log_metric(
                    "embeddings_per_sec", corpus_n.num_rows / encoding_time
                )
                output_dim = len(corpus_with_embeddings["embeddings"][0])
                mlflow.log_param("output_dim", output_dim)
    return


def measure_encoding_speed():
    device = "cpu"
    mlflow.set_tracking_uri("http://localhost:8000")
    mlflow.set_experiment("speed")

    dsdk = load_data()
    corpus, queries, query2corpus_mapping = create_fresh_corpus_and_queries(dsdk)
    shuffled_corpus = corpus.shuffle(seed=RANDOM_STATE)
    contiguous_corpus = shuffled_corpus.flatten_indices()
#    measure_model2vec_speed(
#        contiguous_corpus=contiguous_corpus, device=device, model_path=MODEL2VEC_PATH
#    )
    measure_sentence_transformer_speed(
        contiguous_corpus=contiguous_corpus, device=device
    )
    return


def main():
    mlflow.set_tracking_uri("http://localhost:8000")
    mlflow.set_experiment("speed")
    measure_encoding_speed()


if __name__ == "__main__":
    main()
