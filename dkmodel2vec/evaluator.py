import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datasets import load_dataset
from model2vec import StaticModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
from datasets import DatasetDict, Dataset
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

def predict_bm25(example: dict, tokenizer: Tokenizer)->np.array: 
    """Convert a query its corresponding tokens and get the bm25 prediction from the score across the corpus consisting of a single positive and negative candidates."""
    q_tokens = tokenizer.encode(example["query"], add_special_tokens=False).tokens
    corpus = [ts.tokens for ts in tokenizer.encode_batch([ example['negative'], example['positive']], add_special_tokens = False)]
    
    # somehow BM25 does not work with only 2 examples? so adding a third
    corpus.append(["dummy", "document", "for", "idf", "calculation"])

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q_tokens)[:2] # get rid of the dummy document
    example['bm25_prediction'] = np.argmax(scores)

    return example # 1 is positive predictione and 0 is negative prediction


#    def load_dataset(self, dataset_name: str, split: str = "test", 
#                     query_col: str = "query", pos_col: str = "positive", 
#                     neg_col: str = "negative") -> pd.DataFrame:
#         """Load and prepare the dataset."""
#         dataset = load_dataset(dataset_name, split=split)
#         df = pd.DataFrame(dataset)
        
#         # Ensure required columns exist
#         required_cols = [query_col, pos_col, neg_col]
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             raise ValueError(f"Missing columns: {missing_cols}")
            
#         return df[[query_col, pos_col, neg_col]].rename(columns={
#             query_col: "query", pos_col: "positive", neg_col: "negative"
#         })

def compute_distances(model: StaticModel, dataset: DatasetDict, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Euclidean distances between queries and positive/negative examples."""
    queries = dataset['query']
    positives = dataset['positive']
    negatives = dataset['negative']
    n_samples = len(queries)

    pos_distances = np.zeros(n_samples, dtype = "int8") 
    neg_distances = np.zeros(n_samples, dtype = "int8")
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Computing embeddings"):
        batch_end = min(i + batch_size, n_samples)
        
        batch_queries = queries[i:batch_end]
        batch_positives = positives[i:batch_end]
        batch_negatives = negatives[i:batch_end]
        
        query_embeds = model.encode(batch_queries)
        pos_embeds = model.encode(batch_positives)
        neg_embeds = model.encode(batch_negatives)
        
        # Compute Euclidean distances
        # For distance metrics, smaller values indicate higher similarity
        batch_pos_distances = np.linalg.norm(query_embeds - pos_embeds, axis=1)
        batch_neg_distances = np.linalg.norm(query_embeds - neg_embeds, axis=1)
        
        pos_distances[i:batch_end] = batch_pos_distances
        neg_distances[i:batch_end] = batch_neg_distances

    return pos_distances, neg_distances

def evaluate_classification(predictions: np.array, ground_truth) -> Dict:
    """Evaluate the classification performance. """


    # Ground truth: always positive class (since we expect pos > neg)
    ground_truth = np.ones_like(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='binary'
    )
    
    # MCC is not calculated because there is only a single label. Not sure how to think about this. 
#    mcc = matthews_corrcoef(ground_truth, predictions)
    
    conf_matrix = confusion_matrix(ground_truth, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': predictions,
        'ground_truth': ground_truth, 
    }

def log_performance(results: dict, log_prefix: str = ""):
    """Log metric, dicts and table. """
    metrics_to_log = ['accuracy', 'precision', 'recall', 'f1_score']
        
    for metric in metrics_to_log:
        if not np.isnan(results[metric]):
            mlflow.log_metric(f"{log_prefix}_metric", results[metric])

    class_report = classification_report(
        results['ground_truth'], results['predictions'], 
        output_dict=True
    )
    mlflow.log_dict(class_report, f"{log_prefix}_classification_report.json")

    conf_matrix_df = pd.DataFrame(
        results['confusion_matrix'],
        index=['Actual_0', 'Actual_1'],
        columns=['Pred_0', 'Pred_1']
    )
    mlflow.log_table(conf_matrix_df, f"{log_prefix}_confusion_matrix.json")

    return  

    
def evaluate_model(dataset: Dataset, model: StaticModel, max_samples: int) -> Dict:
    """Run the complete evaluation on the subset of the test set that contains both positive and negative examples."""
    dataset = dataset.filter(lambda example: True if example['has_positive_and_negative'] else False)
    if max_samples:
        dataset = dataset.select(range(max_samples))
    # Compute similarities
    print("Computing similarities...")
    pos_dists, neg_dists = compute_distances(
        query = dataset['query'],
        positive = dataset['positive'], 
        negative =dataset['negative'],
    )
    print("Evaluating classification performance...")
    # Classification: positive class if pos_distance < neg_distance
    predictions = (pos_dists < neg_dists).astype(int)
    results = evaluate_classification(predictions, ground_truth=np.ones_like(predictions))
    log_performance(results)

    print("Computing baseline scores with BM25")
    dataset =  dataset.map(lambda example: predict_bm25(example, model.tokenizer))    
    bm25_results = evaluate_classification(dataset['bm25_prediction'], ground_truth=np.ones_like(predictions))
    log_performance(bm25_results, log_prefix = "bm25")

    return 

