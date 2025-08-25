from datasets import load_dataset

from dkmodel2vec.config import E5_EMBED_INSTRUCTION, DANISH_INSTRUCTION


def has_positive_and_negative(example: dict) -> dict:
    """Flag examples which have both a positive and negative example."""
    example["has_positive_and_negative"] = (
        example["positive"] is not None and example["negative"] is not None
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


def load_data():
    """Loads dataset and only keeps examples in Danish from the training set.
    Adds a column with an index which we need for the stratified cross validation."""
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("DDSC/nordic-embedding-training-data")

    dsdk = ds.filter(lambda sample: True if sample["language"] == "danish" else False)
    dsdk = dsdk["train"]
    dsdk = dsdk.add_column("idx", range(len(dsdk)))
    dsdk = dsdk.map(has_positive_and_negative, num_proc = 4)
    dsdk = dsdk.map(get_detailed_instruct, num_proc = 4)
    dsdk = dsdk.map(get_danish_detailed_instruct, num_proc = 4)
    return dsdk
