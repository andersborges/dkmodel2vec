from datasets import load_dataset
from transformers import AutoTokenizer

def has_positive_and_negative(example: dict)->dict: 
    """Flag examples which have both a positive and negative example. """
    example['has_positive_and_negative'] = example['positive'] is not None and example['negative'] is not None
    return example

def load_data(): 
    """Loads dataset and only keeps examples in Danish from the training set. 
    Adds a column with an index which we need for the stratified cross validation. """
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("DDSC/nordic-embedding-training-data")

    dsdk = ds.filter(lambda sample: True if sample['language']=="danish" else False)
    dsdk = dsdk['train']
    dsdk = dsdk.add_column('idx', range(len(dsdk)))
    dsdk = dsdk.map(has_positive_and_negative)
    return dsdk