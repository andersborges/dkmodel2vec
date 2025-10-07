import logging
from time import time

from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
from bm25s.tokenization import Tokenizer
import bm25s

import numpy as np
import mlflow
from datasets import Dataset

from dkmodel2vec.vocab import create_vocabulary, lower_case_tokenizer
from dkmodel2vec.retrieval import add_embeddings, add_embeddings_wrapped,retrieve,retrieve_bm25s, create_corpus, add_recall, get_mapping_from_query_to_corpus, add_corpus_idx, retrieve_hybrid,retrieve_hybrid_with_reranking
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
    DATASET_NEGATIVE_COLUMN,
    DATASET_POSITIVE_COLUMN,
)

from dkmodel2vec.logging import setup_logging
from model2vec.model import StaticModel

setup_logging()
logger = logging.getLogger(__name__)


# Constants
TEST_SIZE = 0.1
MLFLOW_TRACKING_URI = "http://localhost:8000"
MLFLOW_EXPERIMENT = "retrieval"
CORPUS_COLUMNS = [DATASET_POSITIVE_COLUMN, DATASET_NEGATIVE_COLUMN]
MAX_EXAMPLES = None

def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def prepare_test_data(dsdk):
    """Split data into train and test sets."""
    logger.info(f"Splitting data with test_size={TEST_SIZE}...")
    train_idx, test_idx = train_test_split(
        np.arange(dsdk.num_rows),
        test_size=TEST_SIZE,
        stratify=dsdk["has_positive_and_negative"],
        random_state=RANDOM_STATE,
        shuffle=True
    )
    if MAX_EXAMPLES is not None:
        test_idx = test_idx[:MAX_EXAMPLES]
    
    return dsdk.select(test_idx)


def create_fresh_corpus_and_queries(ds_test:Dataset):
    """Create fresh corpus and queries datasets without caching."""
    flat_corpus = ds_test.map(
        lambda examples: create_corpus(examples, columns=CORPUS_COLUMNS), 
        remove_columns=ds_test.column_names,
        batched=True,
        batch_size=500,
        load_from_cache_file=False
    )
    
    # Add corpus index to flat_corpus
    flat_corpus = flat_corpus.add_column("idx", range(flat_corpus.num_rows))
    
    queries = ds_test.map(
        lambda example: {"query": example['query'], 'idx': example['idx'], "query_instruct" : example['query_instruct']}, 
        remove_columns=ds_test.column_names,
        load_from_cache_file=False
    )
    
    query2corpus = get_mapping_from_query_to_corpus(flat_corpus)
    queries = queries.map(
        lambda example: add_corpus_idx(example, query2corpus), 
        load_from_cache_file=False
    )
    
    return flat_corpus, queries, query2corpus

def encode_corpus_and_index(flat_corpus:Dataset, model: SentenceTransformer, index_name:str, batch_size:int=500, keep_embeddings=False):
    """Encode corpus, create FAISS index, and optionally remove embeddings."""
    t0 = time()
    flat_corpus = flat_corpus.map(
        lambda examples: {"embeddings": model.encode(examples['document'])}, 
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False
    )
    encoding_time = time() - t0
    
    flat_corpus.add_faiss_index(index_name=index_name, column="embeddings")
    
    # Only remove embeddings if keep_embeddings is False
    if not keep_embeddings:
        flat_corpus = flat_corpus.remove_columns(["embeddings"])
    
    return flat_corpus, encoding_time


def retrieve_and_calculate_recall(queries, flat_corpus, top_k=30, index_name="m2v"):
    """Perform retrieval and calculate recall metrics."""
    t0 = time()
    queries_with_retrieval = queries.map(
        lambda examples: retrieve(examples, flat_corpus, top_k=top_k, index_name=index_name), 
        batched=True,
        batch_size=100,
        remove_columns=["embeddings"],
        load_from_cache_file=False
    )
    retrieval_time = time() - t0
    
    queries_with_retrieval = queries_with_retrieval.map(
        add_recall, 
        load_from_cache_file=False
    )
    
    recall_metrics = {
        c: np.mean(list(queries_with_retrieval[c]))
        for c in ["recall@5", "recall@10", "recall@20", "recall@30"]
    }
    
    return retrieval_time, recall_metrics


def log_basic_params(ds_test:Dataset, flat_corpus:Dataset, queries:Dataset):
    """Log basic parameters to MLflow."""
    mlflow.log_param("num_rows", ds_test.num_rows)
    mlflow.log_param("corpus_size", flat_corpus.num_rows)
    mlflow.log_param("query_size", queries.num_rows)


def log_recall_metrics(recall_metrics):
    """Log recall metrics to MLflow."""
    for metric_name, value in recall_metrics.items():
        mlflow.log_metric(metric_name.replace("@", "_at_"), value)


def run_embedding_model(ds_test, model, model_name, index_name="m2v"):
    """Run evaluation for an embedding model."""
    flat_corpus, queries, _ = create_fresh_corpus_and_queries(ds_test)
    
    log_basic_params(ds_test, flat_corpus)
    mlflow.log_param("model_name", model_name)
    
    # Encode corpus
    flat_corpus, corpus_encoding_time = encode_corpus_and_index(
        flat_corpus, model, index_name=index_name
    )
    mlflow.log_metric("corpus_encoding_seconds", corpus_encoding_time)
    
    # Encode queries
    mlflow.log_param("query_size", queries.num_rows)
    queries, query_encoding_time = encode_queries(queries, model)
    mlflow.log_metric("query_encoding_seconds", query_encoding_time)
    
    # Retrieve and calculate recall
    retrieval_time, recall_metrics = retrieve_and_calculate_recall(
        queries, flat_corpus, top_k=30
    )
    mlflow.log_metric("retrieval_time", retrieval_time)
    log_recall_metrics(recall_metrics)


def run_e5_model(ds_test, e5_model):
    """Run evaluation for E5 sentence transformer with special encoding."""
    flat_corpus, queries, _ = create_fresh_corpus_and_queries(ds_test)
    
    log_basic_params(ds_test, flat_corpus)
    mlflow.log_param("model_name", "intfloat/multilingual-e5-large-instruct")
    
    # Encode corpus
    t0 = time()
    flat_corpus = flat_corpus.map(
        lambda examples: {
            "embeddings": e5_model.encode(
                examples['document'],
                normalize_embeddings=True
            )
        }, 
        batched=True,
        batch_size=500,
        load_from_cache_file=False
    )
    mlflow.log_metric("corpus_encoding_seconds", time() - t0)
    
    flat_corpus.add_faiss_index(index_name="e5", column="embeddings")
    flat_corpus = flat_corpus.remove_columns(["embeddings"])
    
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
        batch_size=500,
        load_from_cache_file=False
    )
    mlflow.log_metric("query_encoding_seconds", time() - t0)
    
    # Retrieve and calculate recall
    retrieval_time, recall_metrics = retrieve_and_calculate_recall(
        queries, flat_corpus, top_k=30, index_name="e5"  # ← Add index_name="e5"
    )

    mlflow.log_metric("retrieval_time", retrieval_time)
    log_recall_metrics(recall_metrics)


def run_bm25s(ds_test, flat_corpus, query2corpus, tokenizer_model):
    """Run evaluation for BM25s."""
    queries = ds_test.map(
        lambda example: {"query": example['query'], 'idx': example['idx']}, 
        remove_columns=ds_test.column_names,
        load_from_cache_file=False
    )
    queries = queries.map(
        lambda example: add_corpus_idx(example, query2corpus),
        load_from_cache_file=False
    )
    
    log_basic_params(ds_test, flat_corpus)
    
    # Create tokenizer and index
    t0 = time()
    corpus = list(flat_corpus['document'])
    
    tokenizer = Tokenizer(
        stemmer=None,
        stopwords=None,
        splitter=lambda x: tokenizer_model.tokenizer.encode(x).tokens
    )
    
    corpus_tokens = tokenizer.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    queries = queries.map(
        lambda example: {'tokens': tokenizer.tokenize(example['query'])}, 
        batched=True,
        load_from_cache_file=False
    )
    mlflow.log_metric("corpus_encoding_seconds", time() - t0)
    
    # Retrieve
    t0 = time()
    queries_with_retrieval = queries.map(
        lambda examples: retrieve_bm25s(examples, retriever, top_k=30), 
        batched=True,
        batch_size=100,
        load_from_cache_file=False
    )
    mlflow.log_metric("retrieval_time", time() - t0)
    
    # Calculate recall
    queries_with_retrieval = queries_with_retrieval.map(
        add_recall,
        load_from_cache_file=False
    )
    
    recall_metrics = {
        c: np.mean(list(queries_with_retrieval[c]))
        for c in ["recall@5", "recall@10", "recall@20", "recall@30"]
    }
    log_recall_metrics(recall_metrics)


def run_hybrid_model(ds_test, embedding_model, model_name, rrf_k=60):
    """Run hybrid evaluation combining embedding + BM25."""
    flat_corpus, queries, query2corpus = create_fresh_corpus_and_queries(ds_test)
    
    log_basic_params(ds_test, flat_corpus)
    mlflow.log_param("model_name", f"{model_name}_hybrid")
    mlflow.log_param("fusion_method", "RRF")
    mlflow.log_param("rrf_k", rrf_k)
    
    # Encode corpus for embeddings
    flat_corpus, corpus_encoding_time = encode_corpus_and_index(
        flat_corpus, embedding_model, index_name="m2v"
    )
    
    # Create BM25 index
    t0 = time()
    corpus = list(flat_corpus['document'])
    tokenizer = Tokenizer(
        stemmer=None,
        stopwords=None,
        splitter=lambda x: embedding_model.tokenizer.encode(x).tokens
    )
    corpus_tokens = tokenizer.tokenize(corpus)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens)
    bm25_encoding_time = time() - t0
    
    mlflow.log_metric("corpus_encoding_seconds", 
                     corpus_encoding_time + bm25_encoding_time)
    
    # Encode queries for embeddings
    queries, query_encoding_time = encode_queries(queries, embedding_model)
    
    # Tokenize queries for BM25
    queries = queries.map(
        lambda examples: {'tokens': tokenizer.tokenize(examples['query'])},
        batched=True,
        load_from_cache_file=False
    )
    mlflow.log_metric("query_encoding_seconds", query_encoding_time)
    
    # Hybrid retrieval
    t0 = time()
    queries_with_retrieval = queries.map(
        lambda examples: retrieve_hybrid(
            examples, flat_corpus, bm25_retriever, 
            top_k=30, rrf_k=rrf_k
        ),
        batched=True,
        batch_size=100,
        remove_columns=["embeddings", "tokens"],
        load_from_cache_file=False
    )
    mlflow.log_metric("retrieval_time", time() - t0)
    
    overlap_thresholds = [5, 10, 20, 30]
    overlap_metrics = {}
    for t in overlap_thresholds:
        avg_overlap = np.mean(queries_with_retrieval[f"overlap_at_{t}"])
        overlap_metrics[f"overlap_at_{t}"] = avg_overlap
        mlflow.log_metric(f"avg_overlap_at_{t}", avg_overlap)
        logger.info(f"Average overlap@{t}: {avg_overlap:.3f}")

    # Calculate recall
    queries_with_retrieval = queries_with_retrieval.map(
        add_recall,
        load_from_cache_file=False
    )
    
    recall_metrics = {
        c: np.mean(list(queries_with_retrieval[c]))
        for c in ["recall@5", "recall@10", "recall@20", "recall@30"]
    }
    log_recall_metrics(recall_metrics)
    
    return overlap_metrics, recall_metrics

def run_two_stage_model(ds_test, cheap_model, expensive_model, 
                        cheap_name, expensive_name, stage1_k=100, final_k=30):
    """Run two-stage retrieval evaluation."""
    flat_corpus, queries, _ = create_fresh_corpus_and_queries(ds_test)
    
    log_basic_params(ds_test, flat_corpus, queries)
    mlflow.log_param("model_name", f"{cheap_name}_hybrid_rerank_{expensive_name}")
    mlflow.log_param("stage1_retrievers", f"{cheap_name} + BM25")
    mlflow.log_param("stage2_reranker", expensive_name)
    mlflow.log_param("stage1_k", stage1_k)
    mlflow.log_param("final_k", final_k)
    
    # Encode corpus with BOTH cheap and expensive models at once
    t0_cheap = time()
    flat_corpus = add_embeddings_wrapped(
        ds = flat_corpus, 
        model = cheap_model, 
        in_column="document", 
        out_column="embeddings", 
        batch_size=1000)
    cheap_encoding_time = time() - t0_cheap
    
    t0_expensive = time()
    flat_corpus = add_embeddings_wrapped(
        ds = flat_corpus, 
        model = expensive_model, 
        in_column="document", 
        out_column="expensive_embeddings", 
        batch_size=500)
    
    # add FAISS index here because if dataset is transformed, link to FAISS is broken
    flat_corpus.add_faiss_index(
        column="embeddings", 
        index_name="cheap",
        string_factory="HNSW32", 
        metric_type=1
    )
    flat_corpus.add_faiss_index(index_name="cheap", column="embeddings")

    expensive_encoding_time = time() - t0_expensive
    
    # Build BM25 index
    t0 = time()
    corpus = list(flat_corpus['document'])
    tokenizer = Tokenizer(
        stemmer=None,
        stopwords=None,
        splitter=lambda x: cheap_model.tokenizer.encode(x).tokens
    )
    corpus_tokens = tokenizer.tokenize(corpus)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens)
    bm25_encoding_time = time() - t0
    
    mlflow.log_metric("stage1_corpus_encoding_seconds", cheap_encoding_time + bm25_encoding_time)
    mlflow.log_metric("stage2_corpus_encoding_seconds", expensive_encoding_time)
    
    # Encode queries with cheap model for stage 1
    t0 = time()
    queries = add_embeddings_wrapped(
        ds = queries,
        model =cheap_model, 
        in_column = "query", 
        out_column = "embeddings", 
        batch_size = 1000
    )

    queries = queries.map(
        lambda examples: {'tokens': tokenizer.tokenize(examples['query'])},
        batched=True,
        load_from_cache_file=False
    )
    cheap_query_time = time() - t0
    # Encode queries with expensive model for stage 2
    t0 = time()
    queries = add_embeddings_wrapped(
        ds = queries, 
        model = expensive_model, 
        in_column="query_instruct", 
        out_column = "expensive_embeddings", 
        batch_size = 500
    )
    expensive_query_time = time() - t0
    
    mlflow.log_metric("stage1_query_encoding_seconds", cheap_query_time)
    mlflow.log_metric("stage2_query_encoding_seconds", expensive_query_time)
    
    # Two-stage retrieval
    t0 = time()
    queries_with_retrieval = queries.map(
        lambda examples: retrieve_hybrid_with_reranking(

            examples, flat_corpus, bm25_retriever,index_name="cheap", 
            stage1_k=stage1_k, final_k=final_k, 
        ),
        batched=True,
        batch_size=400,
        remove_columns=["embeddings", "tokens", "expensive_embeddings"],
        load_from_cache_file=False
    )
    total_time = time() - t0
    mlflow.log_metric("total_retrieval_time", total_time)
    
    # Calculate recall
    queries_with_retrieval = queries_with_retrieval.map(
        add_recall,
        load_from_cache_file=False
    )
    
    recall_metrics = {
        c: np.mean(list(queries_with_retrieval[c]))
        for c in ["recall@5", "recall@10", "recall@20", "recall@30"]
    }
    log_recall_metrics(recall_metrics)
    
    return recall_metrics

def main():
    """Main execution function."""
    setup_mlflow()
    
    # Load data and model
    logger.info("Loading data and model...")
    dsdk = load_data()
    model = StaticModel.from_pretrained(
        "scripts/models/dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings"
    )
    
    # Prepare test data
    ds_test = prepare_test_data(dsdk)
    
    # # Run llm2model2vec (COMMENTED OUT - already ran)
    # with mlflow.start_run(run_name="llm2model2vec"):
    #     logger.info("Running llm2model2vec evaluation...")
    #     run_embedding_model(
    #         ds_test, 
    #         model, 
    #         model_name="llm2model2vec",
    #         index_name="m2v"
    #     )
    
    # # Prepare data for BM25s (reuse corpus from first run)
    # flat_corpus_bm25, _, query2corpus = create_fresh_corpus_and_queries(ds_test)
    
    # # Run BM25s (COMMENTED OUT - already ran)
    # with mlflow.start_run(run_name="bm25s"):
    #     logger.info("Running BM25s evaluation...")
    #     run_bm25s(ds_test, flat_corpus_bm25, query2corpus, model)
    
    # # Run reference model2vec (COMMENTED OUT - already ran)
    # with mlflow.start_run(run_name="reference_model2vec"):
    #     logger.info("Running reference model2vec evaluation...")
    #     reference_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    #     run_embedding_model(
    #         ds_test,
    #         reference_model,
    #         model_name="minishlab/potion-base-8M",
    #         index_name="m2v"
    #     )
    
    # # Run E5 sentence transformer (COMMENTED OUT - already ran)
    # with mlflow.start_run(run_name="e5_sentence_transformer"):
    #     logger.info("Running E5 sentence transformer evaluation...")
    #     e5_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    #     run_e5_model(ds_test, e5_model)
    
    # Run hybrid retrieval 
    # with mlflow.start_run(run_name="hybrid_cheap_bm25"):
    #     logger.info("Running hybrid (cheap + BM25) evaluation...")
    #     overlap_metrics, hybrid_recall = run_hybrid_model(
    #         ds_test, model, model_name="llm2model2vec"
    #     )
    #     logger.info(f"Hybrid recall@10: {hybrid_recall['recall@10']:.3f}")
    #     logger.info(f"Overlap@20: {overlap_metrics['overlap_at_20']:.3f}")

    # After hybrid evaluation
    with mlflow.start_run(run_name="two_stage_cheap_bm25_expensive"):
        logger.info("Running two-stage retrieval (cheap+BM25 → expensive reranking)...")
        # Load expensive model
        e5_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        
        two_stage_recall = run_two_stage_model(
            ds_test, 
            cheap_model=model,
            expensive_model=e5_model,
            cheap_name="llm2model2vec",
            expensive_name="e5-large",
            stage1_k=100,
            final_k=30
        )
        logger.info(f"Two-stage recall@10: {two_stage_recall['recall@10']:.3f}")



    logger.info("All evaluations complete!")

    logger.info("All evaluations complete!")
if __name__ == "__main__":
    main()