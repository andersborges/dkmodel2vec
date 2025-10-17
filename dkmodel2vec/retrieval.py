from collections import defaultdict
from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from typing import Any 
from bm25s import BM25
from llm2vec import LLM2Vec

import dkmodel2vec.constants as constants
from dkmodel2vec.constants import DATASET_CORPUS_COLUMNS

def create_corpus(examples: dict, columns:list[str] = [constants.DATASET_POSITIVE_COLUMN, constants.DATASET_NEGATIVE_COLUMN]): 
    """Create corpus in long form where every document has its own row. """
    documents = []
    
    for n in range(len(examples['idx'])): 
        for c in columns:
            if examples[c][n] is not None:
                if examples[c][n].strip() !="":
                    document_n = {"query_idx": examples['idx'][n], "document" : examples[c][n], "column" : c}
                    documents.append(document_n)

    long_form = {
        key: [doc[key] for doc in documents] 
        for key in ["query_idx", "document", "column"]
                }

    return long_form

def get_mapping_from_query_to_corpus(flat_corpus:Dataset):
    """Get mapping between query and corpus for positive examples."""
    query_idx_to_corpus_idx = {}
    for corpus_idx, row in enumerate(flat_corpus):
        if row['column'] == 'positive':
            query_idx_to_corpus_idx[row['query_idx']] = corpus_idx
    return query_idx_to_corpus_idx

def add_corpus_idx(example: dict, query_idx_to_corpus_idx: dict[int, int]):
    """Look up the corpus index using the query's idx"""
    
    example['corpus_idx'] = query_idx_to_corpus_idx.get(example['idx'], None)
    return example

def add_embeddings_wrapped(ds: Dataset, model: SentenceTransformer | LLM2Vec, in_column:str, out_column:str, batch_size:int, device:str = "cuda:0"):
    """Wrapper function to increase readability. Model must have .encode method. """

    ds = ds.map(
        lambda examples: {
            out_column: model.encode(
                examples[in_column],
                normalize_embeddings=True, 
                device=device
            )
        }, 
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False
    )
    return ds

def add_embeddings(examples: dict, model:SentenceTransformer)->dict:
    """Add embeddings to example."""
    examples['embeddings'] = model.encode(examples['query'])
    return examples

def retrieve(query_examples: dict[str, list], index: Any, top_k: int = 30, index_name: str = "m2v") -> dict:
    """Perform batch retrieval for queries on index."""
    queries = np.array(query_examples["embeddings"]).astype(np.float32)
    scores, retrieved_examples = index.get_nearest_examples_batch(
        index_name=index_name, 
        queries=queries,
        k=top_k
    )
    
    result = query_examples.copy()
    result['retrieved_scores'] = scores 
    result['retrieved_document_idx'] = [d['idx'] for d in retrieved_examples]
    result['retrieved_column_name']= [d['column'] for d in retrieved_examples]

    return result

def add_recall(example: dict, thresholds=[5, 10, 20, 30]): 
    """Add recall at different thresholds for retrieved documents. Assumes that all queries have a single correct index"""
    correct_corpus_idx = example['corpus_idx']
    
    for t in thresholds:
        top_t_docs = example['retrieved_document_idx'][:t]        
        found = correct_corpus_idx in top_t_docs
        example[f'recall@{t}'] = 1 if found else 0
    
    return example


def retrieve_bm25s(query_examples: dict[str, list],retriever: BM25, top_k: int = 30) -> dict:
    """Perform batch retrieval for queries on batch"""
    results, scores = retriever.retrieve(query_examples['tokens'], k=top_k)
    
    result = query_examples.copy()
    result['retrieved_scores'] = scores.tolist()
    result['retrieved_document_idx'] = results.tolist()
    return result

def reciprocal_rank_fusion(ranking_lists, k=60):
    """
    Fuse multiple ranking lists using RRF.
    
    Args:
        ranking_lists: List of lists, each containing doc_ids in rank order
        k: RRF parameter (default 60)
    
    Returns:
        List of doc_ids sorted by RRF score (descending)
    """
    scores = defaultdict(float)
    for ranking in ranking_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    
    # Return sorted list of doc_ids
    return [doc_id for doc_id, _ in sorted(scores.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)]


def retrieve_hybrid(query_examples, flat_corpus, bm25_retriever, 
                   top_k=30, index_name="m2v", rrf_k=60):
    """
    Retrieve using both embedding and BM25, then fuse with RRF.
    """
    # Get embedding results
    emb_results = retrieve(query_examples, flat_corpus, top_k, index_name)
    
    # Get BM25 results  
    bm25_results = retrieve_bm25s(query_examples, bm25_retriever, top_k)
    
    # Fuse with RRF for each query in batch
    fused_retrieved_idx = []
    overlap_thresholds = [5, 10, 20, 30]
    overlaps = {f'overlap_at_{t}': [] for t in overlap_thresholds}
    
    for i in range(len(emb_results['retrieved_document_idx'])):
        emb_ranking = emb_results['retrieved_document_idx'][i]
        bm25_ranking = bm25_results['retrieved_document_idx'][i]
        
        # RRF fusion
        fused_ranking = reciprocal_rank_fusion(
            [emb_ranking, bm25_ranking], 
            k=rrf_k
        )
        fused_retrieved_idx.append(fused_ranking[:top_k])
        
        # Calculate overlap at multiple thresholds
        for t in overlap_thresholds:
            emb_set = set(emb_ranking[:t])
            bm25_set = set(bm25_ranking[:t])
            overlap = len(emb_set & bm25_set) / t
            overlaps[f'overlap_at_{t}'].append(overlap)
    
    # Build result dict
    result = {
        'retrieved_document_idx': fused_retrieved_idx,
        'corpus_idx': query_examples['corpus_idx'],
        **overlaps  # Unpack all overlap columns
    }
    
    return result


def retrieve_hybrid_with_reranking(
    query_examples: dict[str, Any], 
    flat_corpus: Dataset, 
    bm25_retriever: BM25,
    index_name: str,
    stage1_k: int = 100,
    final_k: int = 30,
    rrf_k: int = 60,
) -> dict:
    """
    Two-stage retrieval using pre-computed embeddings.
    """
    import time
    import mlflow
    
    t0 = time.time()
    # STAGE 1: Cheap + BM25 hybrid retrieves top-100
    emb_results = retrieve(query_examples, flat_corpus, stage1_k, index_name)
    bm25_results = retrieve_bm25s(query_examples, bm25_retriever, stage1_k)
    stage1_time = time.time() - t0
    mlflow.log_metric("retrieval_stage1_batch_seconds", stage1_time)
    
    t0 = time.time()
    # RRF fusion for stage 1
    stage1_candidates = []
    for i in range(len(emb_results['retrieved_document_idx'])):
        emb_ranking = emb_results['retrieved_document_idx'][i]
        bm25_ranking = bm25_results['retrieved_document_idx'][i]
        
        fused_ranking = reciprocal_rank_fusion(
            [emb_ranking, bm25_ranking], 
            k=rrf_k
        )
        stage1_candidates.append(fused_ranking[:stage1_k])
    rrf_time = time.time() - t0
    mlflow.log_metric("retrieval_rrf_fusion_batch_seconds", rrf_time)
    
    t0 = time.time()
    # Extract embeddings
    expensive_embs_array = np.array(flat_corpus['expensive_embeddings'])
    extract_time = time.time() - t0
    mlflow.log_metric("retrieval_extract_embeddings_batch_seconds", extract_time)
    
    t0 = time.time()
    # STAGE 2: Rerank with pre-computed expensive embeddings
    final_rankings = []
    query_embs = np.array(query_examples['expensive_embeddings'])
    
    for i, candidate_ids in enumerate(stage1_candidates):
        candidate_embs = expensive_embs_array[candidate_ids]
        similarities = np.dot(candidate_embs, query_embs[i])
        reranked_indices = np.argsort(similarities)[::-1][:final_k]
        reranked_ids = [candidate_ids[idx] for idx in reranked_indices]
        final_rankings.append(reranked_ids)
    
    rerank_time = time.time() - t0
    mlflow.log_metric("retrieval_rerank_batch_seconds", rerank_time)
    
    return {
        'retrieved_document_idx': final_rankings,
        'corpus_idx': query_examples['corpus_idx']
    }

def create_fresh_corpus_and_queries(ds:Dataset):
    """Seperate dataset into queries and corpus datasets and create mapping between them."""
    flat_corpus = ds.map(
        lambda examples: create_corpus(examples, columns=DATASET_CORPUS_COLUMNS), 
        remove_columns=ds.column_names,
        batched=True,
        batch_size=500,
        load_from_cache_file=False
    )
    
    # Add corpus index to flat_corpus
    flat_corpus = flat_corpus.add_column("idx", range(flat_corpus.num_rows))
    
    queries = ds.map(
        lambda example: {"query": example['query'], 'idx': example['idx'], "query_instruct" : example['query_instruct']}, 
        remove_columns=ds.column_names,
        load_from_cache_file=False
    )
    
    query2corpus = get_mapping_from_query_to_corpus(flat_corpus)
    queries = queries.map(
        lambda example: add_corpus_idx(example, query2corpus), 
        load_from_cache_file=False
    )
    
    return flat_corpus, queries, query2corpus