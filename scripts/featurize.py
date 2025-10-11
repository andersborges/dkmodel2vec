import argparse

import logging
from pathlib import Path
import random

from dkmodel2vec.logging import setup_logging
from dkmodel2vec.retrieval import create_corpus
from dkmodel2vec.data_loader import load_data, add_splits
from dkmodel2vec.llm_loader import load_llm2vec_model
from dkmodel2vec.config import DANISH_INSTRUCTION, RANDOM_STATE
from dkmodel2vec.constants import DATASET_QUERY_COLUMN, DATASET_POSITIVE_COLUMN, DATASET_NEGATIVE_COLUMN

setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Featurize texts using an LLM2Vec encoder and save as Dataset.")
    parser.add_argument(
        "--max-means",
        type=int,
        default=None,
        help="The maximum number of mean embeddings to generate.",
    )
    args = parser.parse_args()
    max_means = args.max_means

    output_path = Path(__file__).resolve().parent.parent / "features" / f"features_{max_means}"
    logger.info("Loading data set")
    ds = load_data()
    logger.info("Adding splits")
    ds = add_splits(ds) # train, val and test
    idx2split ={idx: split for idx, split in zip(ds['idx'], ds['split'])}
    if max_means is not None: 
        logger.info("Sampling dataset")
        random.seed(RANDOM_STATE)
        num_samples = min(max_means, len(ds))
        sample_idx = random.sample(range(len(ds)), k=num_samples)
        ds = ds.select(sample_idx)


    logger.info("Creating corpus...")
    
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
    ds = ds.map(lambda example: {"split": idx2split[example['query_idx']]})
    # for queries, add instruction, else add empty string
    ds = ds.map(lambda example: {"document": [DANISH_INSTRUCTION, example["document"]] if example['column'] == "query" else ["", example["document"]] })

    model = load_llm2vec_model()
    
    ds = ds.map(
        lambda examples: {
            "embedding": model.encode(
                examples["document"],
            )
        }, 
        batched=True,
        batch_size=16
    )
    ds.save_to_disk(output_path)



if __name__ == "__main__":
    main()