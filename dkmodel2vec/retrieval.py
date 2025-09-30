import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from typing import Any 
from bm25s import BM25

import dkmodel2vec.constants as constants

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

def get_mapping_from_query_to_corpus(flat_corpus):
    """Get mapping between query and corpus for positive examples."""
    query_idx_to_corpus_idx = {}
    for corpus_idx, row in enumerate(flat_corpus):
        if row['column'] == 'positive':
            query_idx_to_corpus_idx[row['query_idx']] = corpus_idx
    return query_idx_to_corpus_idx

def add_corpus_idx(example: dict, query_idx_to_corpus_idx: dict[int, int]):
    # Look up the corpus index using the query's idx
    example['corpus_idx'] = query_idx_to_corpus_idx.get(example['idx'], None)
    return example



def add_embeddings(examples: dict, model:SentenceTransformer)->dict:

    examples['embeddings'] = model.encode(examples['query'])
    return examples

def retrieve(query_examples: dict[str, list], index: Any, top_k: int = 30) -> dict:
    """Perform batch retrieval for queries on index."""
    queries = np.array(query_examples["embeddings"]).astype(np.float32)
    scores, retrieved_examples = index.get_nearest_examples_batch(
        index_name="m2v", 
        queries=queries,
        k=top_k
    )
    
    result = query_examples.copy()
    result['retrieved_scores'] = scores 
    result['retrieved_document_idx'] = [d['query_idx'] for d in retrieved_examples]
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
