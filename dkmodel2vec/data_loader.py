from datasets import load_dataset, DatasetDict, Dataset

import numpy as np
from sklearn.model_selection import train_test_split

from dkmodel2vec.config import E5_EMBED_INSTRUCTION, DANISH_INSTRUCTION, TEST_SIZE, VAL_SIZE, RANDOM_STATE
from dkmodel2vec.constants import (
    HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN,
    DATASET_QUERY_COLUMN,
    DATASET_NEGATIVE_COLUMN,
    DATASET_POSITIVE_COLUMN,

)


def has_positive_and_negative(example: dict) -> dict:
    """Flag examples which have both a positive and negative example."""
    example[HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN] = (
        example[DATASET_POSITIVE_COLUMN] is not None
        and example[DATASET_NEGATIVE_COLUMN] is not None
    )
    return example


def get_detailed_instruct(example: dict) -> dict:
    """Construct instruction for embedding model."""
    example["query_instruct"] = (
        f"'Instruct: {E5_EMBED_INSTRUCTION}\nQuery: {example['query']}'"
    )
    return example


def get_danish_detailed_instruct(example: dict) -> dict:
    """Construct instruction for embedding model in Danish."""
    example["query_danish_instruct"] = f"{DANISH_INSTRUCTION} {example['query']}"
    return example


def load_data() -> Dataset:
    """Loads dataset and only keeps examples in Danish from the training set.
    Adds a column with an index which we need for the stratified cross validation."""
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds: DatasetDict = load_dataset("DDSC/nordic-embedding-training-data")

    dsdk = ds.filter(lambda sample: True if sample["language"] == "danish" else False)
    dsdk = dsdk["train"]
    dsdk = dsdk.add_column("idx", range(len(dsdk)))
    dsdk = dsdk.map(has_positive_and_negative, num_proc=4)
    dsdk = dsdk.map(get_detailed_instruct, num_proc=4)
    return dsdk


def add_splits(ds: Dataset)->Dataset:
    """Add train, val and test set split as a seperate column in dataset.
    Ensure even distribution 'positive/negative' columns by stratifying on 'has_positive_negative' column. """
    train_idx, test_idx = train_test_split(
        np.arange(ds.num_rows),
        test_size=TEST_SIZE,
        stratify=ds["has_positive_and_negative"],
        random_state=RANDOM_STATE,
        shuffle=True
    )
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=VAL_SIZE, 
        stratify=ds['has_positive_and_negative'][train_idx], 
        random_state=RANDOM_STATE, 
        shuffle = True
    )
    mapper = {idx : "train" for idx in train_idx}
    mapper.update({idx: "val" for idx in val_idx})
    mapper.update({idx: "test" for idx in test_idx})
    ds = ds.map(lambda example: {"split": mapper[example['idx']]})
    return ds
 