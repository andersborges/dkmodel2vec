import argparse
from collections import Counter
import logging
from pathlib import Path
import random
import time
import numpy as np
from pathlib import Path

import mlflow
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

from dkmodel2vec.logging import setup_logging, log_memory_usage
from dkmodel2vec.retrieval import create_corpus
from dkmodel2vec.data_loader import load_data, add_splits
from dkmodel2vec.llm_loader import load_llm2vec_model
from dkmodel2vec.config import DANISH_INSTRUCTION, RANDOM_STATE
from dkmodel2vec.constants import DATASET_QUERY_COLUMN, DATASET_POSITIVE_COLUMN, DATASET_NEGATIVE_COLUMN
from dkmodel2vec.utils import check_fits_length

setup_logging()
logger = logging.getLogger(__name__)

def format_for_encoding(batch):
    """Create [instruction, document] pairs for model.encode()"""
    documents = batch['document']
    columns = batch['column']
    
    formatted_pairs = []
    for doc, col in zip(documents, columns):
        instruction = DANISH_INSTRUCTION if col == "query" else ""
        formatted_pairs.append([instruction, doc])
    
    return {'instruction_document_pair': formatted_pairs}


def encode_and_save_incrementally(model, ds_filtered, output_path, batch_size=32, chunk_size=5000):
    """
    Encode dataset and save embeddings incrementally to disk.
    """
    
    output_path = Path(output_path)
    embeddings_dir = output_path / "embeddings_chunks"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    all_pairs = ds_filtered["instruction_document_pair"]
    total_examples = len(all_pairs)
    
    chunk_files = []
    
    for chunk_idx, chunk_start in enumerate(tqdm(range(0, total_examples, chunk_size), 
                                                  desc="Encoding chunks")):
        chunk_end = min(chunk_start + chunk_size, total_examples)
        chunk = all_pairs[chunk_start:chunk_end]
        
        # Encode chunk
        chunk_embeddings = model.encode(
            chunk,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Save chunk to disk
        chunk_file = embeddings_dir / f"chunk_{chunk_idx:04d}.npy"
        np.save(chunk_file, chunk_embeddings)
        chunk_files.append(chunk_file)
    
    logger.info("Loading and concatenating all embeddings...")
    all_embeddings = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        all_embeddings.append(np.load(chunk_file))
    
    all_embeddings = np.vstack(all_embeddings)
    embeddings_list = [embedding for embedding in all_embeddings]
    ds_filtered = ds_filtered.add_column("embedding", embeddings_list)    
    return ds_filtered


def main():
    parser = argparse.ArgumentParser(description="Featurize texts using an LLM2Vec encoder and save as Dataset.")
    parser.add_argument(
        "--max-means",
        type=int,
        default=None,
        help="The maximum number of mean embeddings to generate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=800,
        help="The max_length of the LLM2Vec model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding.",
    )
    args = parser.parse_args()
    max_means = args.max_means
    max_length_arg = args.max_length
    batch_size = args.batch_size

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("max_means", max_means)
        mlflow.log_param("max_length_arg", max_length_arg)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("danish_instruction", DANISH_INSTRUCTION)
        mlflow.log_param("encoding_batch_size", batch_size)
        
        output_path = Path(__file__).resolve().parent.parent / "features" / f"features_{max_means}_max_length_{max_length_arg}"
        mlflow.log_param("output_path", str(output_path))
        
        start_time = time.time()
        logger.info("Loading data set")
        ds = load_data()
        log_memory_usage("After loading data")
        original_dataset_size = len(ds)
        mlflow.log_metric("original_dataset_size", original_dataset_size)
        
        logger.info("Adding splits")
        ds = add_splits(ds)
        log_memory_usage("After adding splits")
        idx2split = {idx: split for idx, split in zip(ds['idx'], ds['split'])}
        
        # Log split distribution
        split_counts = Counter(ds['split'])
        for split_name, count in split_counts.most_common():
            mlflow.log_metric(f"split_{split_name}_count", count)
        
        if max_means is not None:
            logger.info("Sampling dataset")
            random.seed(RANDOM_STATE)
            num_samples = min(max_means, len(ds))
            sample_idx = random.sample(range(len(ds)), k=num_samples)
            ds = ds.select(sample_idx)
            mlflow.log_metric("sampled_dataset_size", len(ds))

        logger.info("Creating corpus...")
        corpus_start = time.time()
        ds = ds.map(
            lambda examples: create_corpus(examples, 
                                           columns=[
                                               DATASET_QUERY_COLUMN, 
                                               DATASET_NEGATIVE_COLUMN, 
                                               DATASET_POSITIVE_COLUMN
                                               ]
                                                 ), 
            batched=True,
            remove_columns=ds.column_names,
            batch_size=500,
        )
        corpus_time = time.time() - corpus_start
        mlflow.log_metric("corpus_creation_time_seconds", corpus_time)
        mlflow.log_metric("corpus_size", len(ds))
        log_memory_usage("After creating corpus")        
        logger.info("Adding splits")
        ds = ds.map(lambda example: {"split": idx2split[example['query_idx']]})

        # Load model just to get tokenizer and params
        logger.info("Loading model")
        model_load_start = time.time()
        log_memory_usage("Before loading llm2vec")
        model = load_llm2vec_model(max_length=max_length_arg)
        model_load_time = time.time() - model_load_start
        mlflow.log_metric("model_load_time_seconds", model_load_time)
        log_memory_usage("After loading llm2vec")        
        logger.info("Extracting tokenizer and params")
        tokenizer = model.tokenizer
        model_max_length = model.max_length
        doc_max_length = model.doc_max_length
        
        mlflow.log_param("model_max_length", model_max_length)
        mlflow.log_param("doc_max_length", doc_max_length)
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)
        logger.info(f"Using device: {device}")
        
        logger.info("Checking text lengths...")
        length_check_start = time.time()
        ds = ds.map(
            lambda batch: check_fits_length(batch, tokenizer, model_max_length, doc_max_length),
            batched=True,
            batch_size=10000,
            num_proc=1
        )
        length_check_time = time.time() - length_check_start
        mlflow.log_metric("length_check_time_seconds", length_check_time)
        log_memory_usage("After length check.")
        ds_filtered = ds.filter(lambda x: x['fits_length'])
        filtered_size = len(ds_filtered)
        filter_ratio = filtered_size / len(ds) if len(ds) > 0 else 0
        
        mlflow.log_metric("filtered_dataset_size", filtered_size)
        mlflow.log_metric("filter_ratio", filter_ratio)
        mlflow.log_metric("filtered_out_count", len(ds) - filtered_size)
        
        logger.info(f"Kept {filtered_size:,} / {len(ds):,} examples ({filter_ratio*100:.1f}%)")
        log_memory_usage("After filtering out long examples")
        # Create the [instruction, text] format for encoding

        logger.info("Formatting for encoding...")
        format_start = time.time()
        ds_filtered = ds_filtered.map(
            format_for_encoding,
            batched=True,
            batch_size=10000,
            num_proc=1,
        )
        format_time = time.time() - format_start
        mlflow.log_metric("formatting_time_seconds", format_time)
        log_memory_usage("After formatting for encoding.")
        # Clean up helper column
        ds_filtered = ds_filtered.remove_columns(['fits_length'])
        
        # Count column distribution before encoding
        column_counts = Counter(ds_filtered['column'])
        for col_name, count in column_counts.most_common():
            mlflow.log_metric(f"column_{col_name}_count", count)
                
        # Encode using the [instruction, document] pairs
        logger.info(f"Encoding embeddings with batch_size={batch_size}...")
        logger.info(f"dataset features: {ds_filtered.features}")
        log_memory_usage("Before encoding")

        encoding_start = time.time()        

        ds_filtered = encode_and_save_incrementally(
            model = model, 
            ds_filtered = ds_filtered, 
            output_path=output_path, 
            batch_size=batch_size, 
            chunk_size=5000)
        encoding_time = time.time() - encoding_start
        mlflow.log_metric("encoding_time_seconds", encoding_time)
        mlflow.log_metric("examples_per_second", filtered_size / encoding_time if encoding_time > 0 else 0)
        
        # Get embedding dimension
        embedding_dim = len(ds_filtered['embedding'][0])
        mlflow.log_metric("embedding_dimension", embedding_dim)
        
        # Save the filtered dataset
        save_start = time.time()
        ds_filtered.save_to_disk(output_path)
        save_time = time.time() - save_start
        mlflow.log_metric("save_time_seconds", save_time)
        
        total_time = time.time() - start_time
        mlflow.log_metric("total_time_seconds", total_time)
        mlflow.log_metric("total_time_minutes", total_time / 60)
        
        logger.info(f"Saved features to {output_path}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Encoding speed: {filtered_size / encoding_time:.2f} examples/second")



if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:8000")
    mlflow.set_experiment("featurize")

    main()