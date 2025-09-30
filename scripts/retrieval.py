import logging
from time import time

from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
from bm25s.tokenization import Tokenizer
import bm25s

import numpy as np
import mlflow

from dkmodel2vec.vocab import create_vocabulary, lower_case_tokenizer
from dkmodel2vec.retrieval import add_embeddings, retrieve,retrieve_bm25s, create_corpus, add_recall, get_mapping_from_query_to_corpus, add_corpus_idx
from dkmodel2vec.llm_loader import load_llm2vec_model
from dkmodel2vec.data_loader import load_data
from dkmodel2vec.config import (
    VOCAB_SIZE,
    DEFAULT_PATTERN, 
    WORD_CONTAINS_UPPER_CASE_PATTERN, 
    CONTAINS_EXOTIC_PATTERN, 
    SIF_COEFFICIENT,
    RANDOM_STATE, 
    NORMALIZE_EMBEDDINGS, 
    FOCUS_PCA, 
    E5_EMBED_INSTRUCTION
)

from dkmodel2vec.constants import (
    HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN,
    DATASET_NEGATIVE_COLUMN,
    DATASET_POSITIVE_COLUMN,
    DATASET_QUERY_COLUMN,
    PREDICTION_COLUMN,
    BM25_PREDICTION_COLUMN,
)

from dkmodel2vec.logging import setup_logging
from model2vec.model import StaticModel

setup_logging()
logger = logging.getLogger(__name__)



test_size = 0.1

mlflow_tracking_uri: str = "http://localhost:8000"
mlflow_experiment: str = "retrieval",
# Setup MLflow
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment('retrieval')

# Load data and model
logger.info("Loading data and model...")
dsdk = load_data()
model = StaticModel.from_pretrained("scripts/models/dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings")
tokenizer = model.tokenizer

# Split data
logger.info(f"Splitting data with test_size={test_size}...")
train_idx, test_idx = train_test_split(
    np.arange(dsdk.num_rows),
    test_size=test_size,
    stratify=dsdk["has_positive_and_negative"],
    random_state=RANDOM_STATE,
    shuffle=True
)
ds_test = dsdk.select(test_idx)  # ← You had a tuple here, fixed
flat_corpus = ds_test.map(
    lambda examples: create_corpus(
        examples, columns=[DATASET_POSITIVE_COLUMN, DATASET_NEGATIVE_COLUMN]
    ), 
    remove_columns=ds_test.column_names,
    batched=True,
    batch_size=500
)
queries = ds_test.map(
    lambda example: {"query": example['query'], 'idx': example['idx']}, 
    remove_columns=ds_test.column_names
)
query2corpus = get_mapping_from_query_to_corpus(flat_corpus)
queries = queries.map(lambda example: add_corpus_idx(example, query2corpus))

with mlflow.start_run(run_name="llm2model2vec"):
    mlflow.log_param("num_rows", ds_test.num_rows)
    mlflow.log_param("corpus_size", flat_corpus.num_rows)

    t0 = time()
    flat_corpus = flat_corpus.map(
        lambda examples: {"embeddings": model.encode(examples['document'])}, 
        batched=True,
        batch_size=500
    )
    t1 = time()
    mlflow.log_metric("corpus_encoding_seconds", t1-t0)

    flat_corpus.add_faiss_index(index_name="m2v", column="embeddings")    
    # remove embeddings after indexing - potentially HUGE memory savings!
    flat_corpus = flat_corpus.remove_columns(["embeddings"])
    
    mlflow.log_param("query_size", queries.num_rows)
    t0 = time()    
    queries = queries.map(
        lambda examples: add_embeddings(examples, model), 
        batched=True,
        batch_size=500
    )
    t1 = time()
    mlflow.log_metric("query_encoding_seconds", t1-t0)

    t0 = time()
    queries_with_retrieval = queries.map(
        lambda examples: retrieve(examples, flat_corpus, top_k=30), 
        batched=True,
        batch_size=100,  # smaller batches for retrieval
        remove_columns=["embeddings"]  # remove embeddings after retrieval
    )
    t1 = time()
    mlflow.log_metric("retrieval_time", t1-t0)

    queries_with_retrieval = queries_with_retrieval.map(add_recall)
    
    for c in ["recall@5", "recall@10", "recall@20", "recall@30"]:
        recall = np.mean(list(queries_with_retrieval[c]))
        mlflow.log_metric(c.replace("@", "_at_"), recall)


with mlflow.start_run(run_name="bm25s"):
    queries = ds_test.map(
        lambda example: {"query": example['query'], 'idx': example['idx']}, 
        remove_columns=ds_test.column_names
    )
    queries = queries.map(lambda example: add_corpus_idx(example, query2corpus))

    mlflow.log_param("num_rows", ds_test.num_rows)
    mlflow.log_param("corpus_size", flat_corpus.num_rows)
    t0 = time()
    corpus = list(flat_corpus['document'])

    stemmer = None
    stopwords = None
    splitter = lambda x: model.tokenizer.encode(x).tokens
    # Create a tokenizer
    tokenizer = Tokenizer(
        stemmer=stemmer, stopwords=stopwords, splitter=splitter
    )

    t0 = time()
    corpus_tokens = tokenizer.tokenize(corpus)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    queries = queries.map(
        lambda example: {'tokens': tokenizer.tokenize(example['query'])}, 
        batched=True
    )
    t1 = time()
    mlflow.log_metric("corpus_encoding_seconds", t1-t0)

    t0 = time()
    queries_with_retrieval = queries.map(
    lambda examples: retrieve_bm25s(examples, retriever, top_k=30), 
    batched=True,
    batch_size=100
    )

    t1 = time()
    mlflow.log_metric("retrieval_time", t1-t0)

    queries_with_retrieval =queries_with_retrieval.map(add_recall)
    
    for c in ["recall@5", "recall@10", "recall@20", "recall@30"]:
        recall = np.mean(list(queries_with_retrieval[c]))
        mlflow.log_metric(c.replace("@", "_at_"), recall)

with mlflow.start_run(run_name="reference_model2vec"):
    # Recreate queries dataset
    queries = ds_test.map(
        lambda example: {"query": example['query'], 'idx': example['idx']}, 
        remove_columns=ds_test.column_names
    )
    queries = queries.map(lambda example: add_corpus_idx(example, query2corpus))
    
    mlflow.log_param("num_rows", ds_test.num_rows)
    mlflow.log_param("corpus_size", flat_corpus.num_rows)
    mlflow.log_param("model_name", "minishlab/potion-base-8M")
    
    # Load reference model
    logger.info("Loading reference model2vec model...")
    reference_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    
    # Encode corpus
    t0 = time()
    flat_corpus_ref = flat_corpus.map(
        lambda examples: {"embeddings": reference_model.encode(examples['document'])}, 
        batched=True,
        batch_size=500
    )
    t1 = time()
    mlflow.log_metric("corpus_encoding_seconds", t1-t0)
    
    # Index corpus
    flat_corpus_ref.add_faiss_index(index_name="m2v", column="embeddings")    
    flat_corpus_ref = flat_corpus_ref.remove_columns(["embeddings"])
    
    # Encode queries
    mlflow.log_param("query_size", queries.num_rows)
    t0 = time()    
    queries = queries.map(
        lambda examples: add_embeddings(examples, reference_model), 
        batched=True,
        batch_size=500
    )
    t1 = time()
    mlflow.log_metric("query_encoding_seconds", t1-t0)
    
    # Retrieve
    t0 = time()
    queries_with_retrieval = queries.map(
        lambda examples: retrieve(examples, flat_corpus_ref, top_k=30), 
        batched=True,
        batch_size=100,
        remove_columns=["embeddings"]
    )
    t1 = time()
    mlflow.log_metric("retrieval_time", t1-t0)
    
    # Calculate recall
    queries_with_retrieval = queries_with_retrieval.map(add_recall)
    
    for c in ["recall@5", "recall@10", "recall@20", "recall@30"]:
        recall = np.mean(list(queries_with_retrieval[c]))
        mlflow.log_metric(c.replace("@", "_at_"), recall)


with mlflow.start_run(run_name="e5_sentence_transformer"):
    # Recreate queries dataset
    queries = ds_test.map(
        lambda example: {"query": example['query'], 'idx': example['idx']}, 
        remove_columns=ds_test.column_names
    )
    queries = queries.map(lambda example: add_corpus_idx(example, query2corpus))
    
    mlflow.log_param("num_rows", ds_test.num_rows)
    mlflow.log_param("corpus_size", flat_corpus.num_rows)
    mlflow.log_param("model_name", "intfloat/multilingual-e5-large-instruct")
    
    # Load E5 model
    logger.info("Loading E5 sentence transformer model...")
    e5_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        
    t0 = time()
    flat_corpus_e5 = flat_corpus.map(
        lambda examples: {
            "embeddings": e5_model.encode(
                [doc for doc in examples['document']],
                normalize_embeddings=True
            )
        }, 
        batched=True,
        batch_size=500
    )
    t1 = time()
    mlflow.log_metric("corpus_encoding_seconds", t1-t0)
    
    flat_corpus_e5.add_faiss_index(index_name="e5", column="embeddings")    
    flat_corpus_e5 = flat_corpus_e5.remove_columns(["embeddings"])
    
    # Encode queries with instruction
    mlflow.log_param("query_size", queries.num_rows)
    t0 = time()    
    queries = queries.map(
        lambda examples: {
            "embeddings": e5_model.encode(
                [f"Instruct: {E5_EMBED_INSTRUCTION}\nQuery: {q}" for q in examples['query']],
                normalize_embeddings=True
            )
        }, 
        batched=True,
        batch_size=500
    )
    t1 = time()
    mlflow.log_metric("query_encoding_seconds", t1-t0)
    
    # Retrieve
    t0 = time()
    queries_with_retrieval = queries.map(
        lambda examples: retrieve(examples, flat_corpus_e5, top_k=30), 
        batched=True,
        batch_size=100,
        remove_columns=["embeddings"]
    )
    t1 = time()
    mlflow.log_metric("retrieval_time", t1-t0)
    
    # Calculate recall
    queries_with_retrieval = queries_with_retrieval.map(add_recall)
    
    for c in ["recall@5", "recall@10", "recall@20", "recall@30"]:
        recall = np.mean(list(queries_with_retrieval[c]))
        mlflow.log_metric(c.replace("@", "_at_"), recall)